from __future__ import annotations
from typing import Dict, List, Any, Optional

from huggingface_hub import DatasetCard, list_datasets

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState
from one_eval.logger import get_logger

log = get_logger("BenchResolveAgent")


class BenchResolveAgent(CustomAgent):

    @property
    def role_name(self) -> str:
        return "BenchResolveAgent"

    # 这里不需要 prompt，整个 Agent 不调用 LLM
    @property
    def system_prompt_template_name(self) -> str:
        return ""

    @property
    def task_prompt_template_name(self) -> str:
        return ""

    def _extract_query_info(self, state: NodeState) -> Dict[str, Any]:
        """只拿 domain / specific_benches，给后续逻辑用"""
        q = {}
        if isinstance(state.result, dict):
            q = state.result.get("QueryUnderstandAgent", {}) or {}

        return {
            "domain": q.get("domain") or [],
            "specific_benches": q.get("specific_benches") or [],
        }

    # -------- HF 解析逻辑（保持你原先的思路，稍微收拾一下） --------
    def _resolve_hf_bench(self, bench_name: str) -> Optional[Dict[str, Any]]:
        """
        尝试根据 bench_name 从 HuggingFace 拉取数据集信息：
        1) 直接当 repo_id 调用 DatasetCard.load
        2) 若失败，使用 list_datasets(search=bench_name) 搜索，优先匹配 id 末尾等于 bench_name 的
        找不到则返回 None
        """
        if not isinstance(bench_name, str):
            return None

        bench_name = bench_name.strip()
        if not bench_name:
            return None

        # 1) 直接当作 repo_id 尝试
        try:
            card = DatasetCard.load(bench_name)
            data = getattr(card, "data", {}) or {}
            return {
                "bench_name": bench_name,
                "hf_repo": bench_name,
                "card_text": card.text or "",
                "tags": data.get("tags", []),
                "exists_on_hf": True,
            }
        except Exception:
            pass

        # 2) 用搜索 + 后缀精确匹配
        try:
            candidates = list(list_datasets(search=bench_name, limit=10))
        except Exception as e:
            log.warning(f"[BenchResolveAgent] list_datasets(search={bench_name}) 失败: {e}")
            return None

        bench_lower = bench_name.lower()
        chosen_id = None
        for d in candidates:
            ds_id = d.id
            short_id = ds_id.split("/")[-1].lower()
            if short_id == bench_lower:
                chosen_id = ds_id
                break

        if not chosen_id and candidates:
            chosen_id = candidates[0].id

        if not chosen_id:
            return None

        try:
            card = DatasetCard.load(chosen_id)
            data = getattr(card, "data", {}) or {}
            return {
                "bench_name": bench_name,          # 用户原始名称
                "hf_repo": chosen_id,              # 真正的 HF repo_id
                "card_text": card.text or "",
                "tags": data.get("tags", []),
                "exists_on_hf": True,
            }
        except Exception as e:
            log.warning(f"[BenchResolveAgent] DatasetCard.load({chosen_id}) 失败: {e}")
            return {
                "bench_name": bench_name,
                "hf_repo": chosen_id,
                "card_text": "",
                "tags": [],
                "exists_on_hf": False,
            }

    async def run(self, state: NodeState) -> NodeState:
        log.info("[BenchResolveAgent] 执行开始")

        # 如果前一个 Agent 判定可以直接跳过，则不做任何事
        if state.temp_data.get("skip_resolve"):
            log.info("[BenchResolveAgent] skip_resolve=True，直接返回")
            return state

        info = self._extract_query_info(state)
        specific_benches: List[str] = info["specific_benches"] or []

        # 第一个 Agent 推荐的 bench 名称列表
        bench_names: List[str] = state.temp_data.get("bench_names_suggested", []) or []

        # 已有的本地 bench_info（来自 BenchNameSuggestAgent，已经把本地表匹配好）
        bench_info: Dict[str, Dict[str, Any]] = getattr(state, "bench_info", {}) or {}

        existing_keys = set(bench_info.keys())

        # ================ Step 1: 需要去 HF 解析的名称集合 ================
        names_to_resolve: List[str] = []

        # 1) LLM 推荐的 bench_names 中，本地没出现的
        for name in bench_names:
            if not name:
                continue
            if name in existing_keys:
                continue
            names_to_resolve.append(name)

        # 2) 用户强制指定的 specific_benches 里，本地没出现的
        for name in specific_benches:
            if not name:
                continue
            if name in existing_keys:
                continue
            if name not in names_to_resolve:
                names_to_resolve.append(name)

        log.info(f"[BenchResolveAgent] 需要在 HF 上解析的名称: {names_to_resolve}")

        # ================ Step 2: HF 精确解析 ================
        hf_resolved: List[Dict[str, Any]] = []

        for name in names_to_resolve:
            resolved = self._resolve_hf_bench(name)
            if not resolved:
                continue

            hf_resolved.append(resolved)

            repo_id = resolved.get("hf_repo") or name
            if repo_id not in bench_info:
                bench_info[repo_id] = {
                    "bench_name": repo_id,
                    "source": "hf_resolve",
                    "aliases": [name],     # 保留原始名称
                    "hf_meta": resolved,
                }

        # ================ Step 3: 写回 state ================
        state.benches = list(bench_info.keys())
        state.bench_info = bench_info

        state.agent_results["BenchResolveAgent"] = {
            "bench_names_input": bench_names,
            "specific_benches": specific_benches,
            "hf_resolved": hf_resolved,
        }

        log.info(f"[BenchResolveAgent] 最终 bench 列表: {state.benches}")
        return state
