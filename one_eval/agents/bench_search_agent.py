from __future__ import annotations
import json
from typing import Dict, List, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from huggingface_hub import DatasetCard, list_datasets

from one_eval.utils.bench_registry import BenchRegistry
from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState
from one_eval.logger import get_logger

log = get_logger("BenchSearchAgent")


class BenchSearchAgent(CustomAgent):

    @property
    def role_name(self) -> str:
        return "BenchSearchAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "bench_search.system"

    @property
    def task_prompt_template_name(self) -> str:
        # 这里沿用原来的 key 名称，但语义已经改成“推荐 bench 名称列表”
        return "bench_search.hf_query"

    # --------------------------------------------------------
    # 取第一步 agent 的输出（QueryUnderstandAgent）
    # --------------------------------------------------------
    def _extract_query_info(self, state: NodeState) -> Dict[str, Any]:
        """
        当前约定：state.result 形如：
        {
          "QueryUnderstandAgent": {...}
        }
        """
        if isinstance(state.result, dict):
            return state.result.get("QueryUnderstandAgent", {}) or {}
        return {}

    # --------------------------------------------------------
    # 辅助：尝试把 bench_name 映射到 HF 数据集信息
    # --------------------------------------------------------
    def _resolve_hf_bench(self, bench_name: str) -> Optional[Dict[str, Any]]:
        """
        尝试根据 bench_name 从 HuggingFace 拉取数据集信息：
        1) 先尝试把 bench_name 当成完整 repo_id 使用 DatasetCard.load
        2) 若失败，再用 list_datasets(search=bench_name)，优先匹配 id 末尾等于 bench_name 的
        找不到则返回 None
        """
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
            candidates = list_datasets(search=bench_name, limit=10)
        except Exception as e:
            log.warning(f"[BenchSearchAgent] list_datasets(search={bench_name}) 失败: {e}")
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
            # 兜底：取第一个（不一定完美，但总比没有强）
            chosen_id = candidates[0].id

        if not chosen_id:
            return None

        try:
            card = DatasetCard.load(chosen_id)
            data = getattr(card, "data", {}) or {}
            return {
                "bench_name": bench_name,
                "hf_repo": chosen_id,
                "card_text": card.text or "",
                "tags": data.get("tags", []),
                "exists_on_hf": True,
            }
        except Exception as e:
            log.warning(f"[BenchSearchAgent] DatasetCard.load({chosen_id}) 失败: {e}")
            return {
                "bench_name": bench_name,
                "hf_repo": chosen_id,
                "card_text": "",
                "tags": [],
                "exists_on_hf": False,
            }

    # --------------------------------------------------------
    # Agent 主运行逻辑
    # --------------------------------------------------------
    async def run(self, state: NodeState) -> NodeState:
        log.info("[BenchSearchAgent] 执行开始")

        q = self._extract_query_info(state)
        specific_benches: List[str] = q.get("specific_benches") or []
        domain: List[str] = q.get("domain") or []
        user_query = getattr(state, "user_query", "")

        # ================ Step 1: 本地 BenchRegistry 搜索 ================
        registry = BenchRegistry("one_eval/utils/bench_table/bench_config.json")
        local_matches = registry.search(
            specific_benches=specific_benches,
            domain=domain,
        )
        log.warning(
            f"用于检索的关键词是: specific_benches={specific_benches}, domain={domain}"
        )
        log.info(f"[BenchSearchAgent] 本地匹配到 {len(local_matches)} 个 bench")

        bench_info: Dict[str, Dict[str, Any]] = {}
        for m in local_matches:
            name = m["bench_name"]
            bench_info[name] = m

        # 如果本地 >= 3，则跳过后续推荐
        if len(local_matches) >= 3:
            state.benches = list(bench_info.keys())
            state.bench_info = bench_info
            state.agent_results["BenchSearchAgent"] = {
                "local_matches": local_matches,
                "llm_bench_names": [],
                "hf_resolved": [],
            }
            log.info("[BenchSearchAgent] 使用本地结果即可")
            return state

        # ================ Step 2: 通过 LLM 推荐 benchmark 名称列表 ================
        sys_prompt = self.get_prompt(self.system_prompt_template_name)
        task_prompt = self.get_prompt(
            self.task_prompt_template_name,
            user_query=user_query,
            domain=",".join(domain),
            local_benches=",".join(bench_info.keys()),
        )

        llm = self.create_llm(state)
        resp = await llm.call(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=task_prompt),
            ],
            bind_post_tools=False,
        )

        parsed = self.parse_result(resp.content)

        bench_names: List[str] = []
        if isinstance(parsed, dict):
            raw_list = parsed.get("bench_names") or []
            if isinstance(raw_list, list):
                bench_names = [
                    str(x).strip()
                    for x in raw_list
                    if isinstance(x, (str, int, float)) and str(x).strip()
                ]

        bench_names = list(dict.fromkeys(bench_names))  # 去重保持顺序

        log.info(f"[BenchSearchAgent] LLM 推荐 bench_names: {bench_names}")

        if not bench_names:
            # LLM 没给任何推荐，只返回本地结果
            state.benches = list(bench_info.keys())
            state.bench_info = bench_info
            state.agent_results["BenchSearchAgent"] = {
                "local_matches": local_matches,
                "llm_bench_names": [],
                "hf_resolved": [],
            }
            return state

        # ================ Step 3: 利用本地 registry 再匹配一轮（按名称/别名） ================
        for name in bench_names:
            # 如果已经在 bench_info 里，就不再查
            if name in bench_info:
                continue
            # 按名称再查一轮（不限制 domain）
            extra_matches = registry.search(
                specific_benches=[name],
                domain=None,
            )
            for m in extra_matches:
                bname = m["bench_name"]
                if bname not in bench_info:
                    bench_info[bname] = m

        # ================ Step 4: 对仍未在本地出现的名称，尝试从 HF 精确解析 ================
        hf_resolved: List[Dict[str, Any]] = []

        # 已知本地 bench 的名字集合（含 registry 的主键）
        existing_keys = set(bench_info.keys())

        for name in bench_names:
            if name in existing_keys:
                continue
            resolved = self._resolve_hf_bench(name)
            if not resolved:
                continue

            hf_resolved.append(resolved)

            # 使用 hf_repo 作为 key（更稳定），同时保留原始 bench_name 作为别名
            repo_id = resolved.get("hf_repo") or name
            if repo_id not in bench_info:
                bench_info[repo_id] = {
                    "bench_name": repo_id,
                    "source": "hf_resolve",
                    "aliases": [name],
                    "hf_meta": resolved,
                }

        # ================ Step 5: 汇总结果写回 state ================
        state.benches = list(bench_info.keys())
        state.bench_info = bench_info

        state.agent_results["BenchSearchAgent"] = {
            "local_matches": local_matches,
            "llm_bench_names": bench_names,
            "hf_resolved": hf_resolved,
        }

        log.info(f"[BenchSearchAgent] 最终 bench 列表: {state.benches}")

        return state
