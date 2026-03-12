from __future__ import annotations

import json
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState
from one_eval.logger import get_logger

log = get_logger("BenchConfigRecommendAgent")


class BenchConfigRecommendAgent(CustomAgent):
    @property
    def role_name(self) -> str:
        return "BenchConfigRecommendAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "bench_config_recommend.system"

    @property
    def task_prompt_template_name(self) -> str:
        return "bench_config_recommend.task"

    def _pick_best_split(self, splits: List[str], preferred: str) -> str:
        if not splits:
            return preferred
        if preferred in splits:
            return preferred
        for cand in ("test", "validation", "dev", "val", "train"):
            if cand in splits:
                return cand
        fuzzy_test = [s for s in splits if "test" in s.lower()]
        if fuzzy_test:
            return fuzzy_test[0]
        fuzzy_val = [s for s in splits if "valid" in s.lower() or "dev" in s.lower()]
        if fuzzy_val:
            return fuzzy_val[0]
        return splits[0]

    def _extract_candidates(self, structure: Dict[str, Any]) -> Dict[str, List[str]]:
        candidates: Dict[str, List[str]] = {}
        subsets = structure.get("subsets", [])
        if not isinstance(subsets, list):
            return candidates
        for subset in subsets:
            if not isinstance(subset, dict):
                continue
            config = subset.get("subset")
            if not isinstance(config, str) or not config:
                continue
            raw_splits = subset.get("splits", [])
            split_names: List[str] = []
            if isinstance(raw_splits, list):
                for sp in raw_splits:
                    if isinstance(sp, dict) and isinstance(sp.get("name"), str) and sp.get("name"):
                        split_names.append(sp["name"])
                    elif isinstance(sp, str) and sp:
                        split_names.append(sp)
            if not split_names:
                split_names = ["test"]
            candidates[config] = split_names
        return candidates

    def _normalize_choice(
        self,
        config: Optional[str],
        split: Optional[str],
        candidates: Dict[str, List[str]],
    ) -> Optional[Dict[str, str]]:
        if not candidates:
            return None
        configs = list(candidates.keys())
        chosen_config = config if isinstance(config, str) and config in candidates else configs[0]
        available_splits = candidates.get(chosen_config, [])
        preferred_split = split if isinstance(split, str) and split else "test"
        chosen_split = self._pick_best_split(available_splits, preferred_split)
        return {"config": chosen_config, "split": chosen_split}

    async def run(self, state: NodeState) -> NodeState:
        benches = getattr(state, "benches", None)
        if not benches:
            return state

        llm = self.create_llm(state)

        for bench in benches:
            if not bench.meta:
                continue

            structure = bench.meta.get("structure")

            if not structure or not structure.get("ok"):
                log.warning(f"跳过 {bench.bench_name}: 无结构信息")
                continue

            repo_id = structure.get("repo_id") or bench.bench_name
            candidates = self._extract_candidates(structure)
            if not candidates:
                log.warning(f"[{repo_id}] 无可用 config/split 候选，跳过推荐")
                continue

            existing = bench.meta.get("download_config")
            if isinstance(existing, dict):
                fixed_existing = self._normalize_choice(existing.get("config"), existing.get("split"), candidates)
                if fixed_existing:
                    bench.meta["download_config"] = {
                        "config": fixed_existing["config"],
                        "split": fixed_existing["split"],
                        "reason": existing.get("reason") or "validated existing download_config",
                    }
                    if (
                        fixed_existing["config"] == existing.get("config")
                        and fixed_existing["split"] == existing.get("split")
                    ):
                        log.info(f"[{bench.bench_name}] 复用已有 download_config: {bench.meta['download_config']}")
                        continue
                    log.info(f"[{bench.bench_name}] 修正已有 download_config: {bench.meta['download_config']}")
                    continue

            simplified_structure = {"subsets": []}
            for subset in structure.get("subsets", []):
                s_info = {
                    "subset": subset.get("subset"),
                    "splits": [s.get("name") for s in subset.get("splits", [])],
                }
                simplified_structure["subsets"].append(s_info)

            msgs = [
                SystemMessage(content=self.get_prompt(self.system_prompt_template_name)),
                HumanMessage(
                    content=self.get_prompt(
                        self.task_prompt_template_name,
                        repo_id=repo_id,
                        structure_json=json.dumps(
                            simplified_structure, ensure_ascii=False, indent=2
                        ),
                    )
                ),
            ]

            try:
                resp = await llm.ainvoke(msgs)
                content = resp.content

                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                recommendation = json.loads(content)
                fixed = self._normalize_choice(
                    recommendation.get("config") if isinstance(recommendation, dict) else None,
                    recommendation.get("split") if isinstance(recommendation, dict) else None,
                    candidates,
                )
                if fixed:
                    bench.meta["download_config"] = {
                        "config": fixed["config"],
                        "split": fixed["split"],
                        "reason": "validated by structure whitelist",
                    }
                    log.info(f"[{repo_id}] 推荐配置(白名单校验后): {bench.meta['download_config']}")
                else:
                    log.warning(f"[{repo_id}] LLM 返回格式错误: {content}")

            except Exception as e:
                fallback = self._normalize_choice(None, None, candidates)
                if fallback:
                    bench.meta["download_config"] = {
                        "config": fallback["config"],
                        "split": fallback["split"],
                        "reason": "fallback by structure whitelist",
                    }
                    log.warning(f"[{repo_id}] 推荐失败，使用白名单兜底配置: {bench.meta['download_config']}, error={e}")
                else:
                    log.error(f"[{repo_id}] 推荐失败: {e}")

        return state
