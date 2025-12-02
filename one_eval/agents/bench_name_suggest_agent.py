from __future__ import annotations
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState
from one_eval.utils.bench_registry import BenchRegistry
from one_eval.logger import get_logger

log = get_logger("BenchNameSuggestAgent")


class BenchNameSuggestAgent(CustomAgent):

    @property
    def role_name(self) -> str:
        return "BenchNameSuggestAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "bench_search.system"

    @property
    def task_prompt_template_name(self) -> str:
        return "bench_search.hf_query"

    def _extract_query_info(self, state: NodeState) -> Dict[str, Any]:
        """
        从 QueryUnderstandAgent 的输出中抽取必要信息：
        state.result 约定为:
        {
          "QueryUnderstandAgent": {...}
        }
        """
        q = {}
        if isinstance(state.result, dict):
            q = state.result.get("QueryUnderstandAgent", {}) or {}

        return {
            "domain": q.get("domain") or [],
            "specific_benches": q.get("specific_benches") or [],
            "user_query": getattr(state, "user_query", ""),
        }

    async def run(self, state: NodeState) -> NodeState:
        log.info("[BenchNameSuggestAgent] 执行开始")

        info = self._extract_query_info(state)
        domain: List[str] = info["domain"]
        specific_benches: List[str] = info["specific_benches"]
        user_query: str = info["user_query"]

        # ================ Step 1: 本地 BenchRegistry 搜索 ================
        registry = BenchRegistry("one_eval/utils/bench_table/bench_config.json")
        local_matches = registry.search(
            specific_benches=specific_benches,
            domain=domain,
        )

        # log.warning(
        #     f"[BenchNameSuggestAgent] 检索关键词: specific_benches={specific_benches}, domain={domain}"
        # )
        log.info(f"[BenchNameSuggestAgent] 本地匹配到 {len(local_matches)} 个 bench")

        bench_info: Dict[str, Dict[str, Any]] = {
            m["bench_name"]: m for m in local_matches
        }

        # 如果本地 >= 3，认为已经足够，不再调用 LLM 推荐
        if len(local_matches) >= 3:
            state.benches = list(bench_info.keys())
            state.bench_info = bench_info
            state.agent_results["BenchNameSuggestAgent"] = {
                "local_matches": local_matches,
                "bench_names": [],
                "skip_resolve": True,
            }
            # 标记后续 BenchResolveAgent 可以直接跳过
            state.temp_data["skip_resolve"] = True
            log.info("[BenchNameSuggestAgent] 本地 bench >= 3，跳过推荐")
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
        raw_list = []
        if isinstance(parsed, dict):
            raw_list = parsed.get("bench_names") or []

        bench_names: List[str] = []
        for x in raw_list:
            if isinstance(x, (str, int, float)):
                name = str(x).strip()
                if name and name.lower() not in {"none", "null", "-"}:
                    bench_names.append(name)

        # 去重保持顺序
        bench_names = list(dict.fromkeys(bench_names))

        log.info(f"[BenchNameSuggestAgent] LLM 推荐 bench_names: {bench_names}")

        # 把本地匹配 + 推荐名称写入中间状态，方便后续 BenchResolveAgent 使用
        state.bench_info = bench_info
        state.temp_data["bench_names_suggested"] = bench_names

        state.agent_results["BenchNameSuggestAgent"] = {
            "local_matches": local_matches,
            "bench_names": bench_names,
            "skip_resolve": False,
        }

        return state
