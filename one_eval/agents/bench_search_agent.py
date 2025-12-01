from __future__ import annotations
from typing import Dict, Any, List

from one_eval.core.agent import CustomAgent
from one_eval.utils.prompts import prompt_registry
from one_eval.bench.bench_registry import BenchRegistry
from one_eval.toolkits.hf_search_tool import hf_search_tool
from one_eval.logger import get_logger

log = get_logger("BenchSearchAgent")


class BenchSearchAgent(CustomAgent):
    """
    Step 2 Agent:负责搜索 benchmark。
    - 先查本地 bench 表
    - 若结果不足 → 自动调用 hf_search_tool(LLM 再过滤与识别)
    """

    def __init__(self, tool_manager=None, model_name="gpt-4o"):
        super().__init__(
            tool_manager=tool_manager,
            model_name=model_name,
            react_mode=True,  # !! Step2 必须启用，因为需要 function_call
        )

        # === 配置 prompt 名 ===
        self.system_prompt_template_name = "bench_search.system"
        self.task_prompt_template_name = "bench_search.task"

        # === 默认加载本地 bench registry ===
        self.registry = BenchRegistry(
            config_path="one_eval/utils/bench_table/bench_config.json"
        )

    async def run(self, state) -> Any:
        log.info("BenchSearchAgent 开始执行")

        # 从 QueryUnderstandAgent 读取第 1 步解析结果
        q = state.agent_results.get("query_understand", {})

        specific = q.get("specific_benches", []) or []
        domain = q.get("domain", []) or []

        # =========================================
        # 1) 本地 bench_registry 匹配
        # =========================================
        local_results = self.registry.search(
            specific_benches=specific,
            domain=domain,
        )

        log.info(f"本地匹配结果数量: {len(local_results)}")

        # 若本地结果 >= 3，则直接返回（不调用 HF）
        if len(local_results) >= 3:
            state.agent_results["bench_search"] = local_results
            log.info("BenchSearchAgent 结束（使用本地匹配）")
            return state

        # =========================================
        # 2) 准备调用工具：hf_search_tool（function-call）
        # =========================================
        system_prompt = self.get_prompt(self.system_prompt_template_name)
        task_prompt = self.get_prompt(
            self.task_prompt_template_name,
            user_query=state.user_query,
            local_candidates=str(local_results),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        # LLM 自动选择是否调用 hf_search_tool
        llm = self.create_llm(state)
        response = await llm.call(messages, bind_post_tools=True)

        # 若模型没有使用工具，强制 fallback 为 local-only
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        if not tool_calls:
            log.info("模型未调用 hf_search_tool，用本地结果即可")
            state.agent_results["bench_search"] = local_results
            return state

        # =========================================
        # 3) 执行工具
        # =========================================
        tool_call = tool_calls[0]
        args = tool_call["function"]["arguments"]
        import json
        args = json.loads(args)

        hf_results = hf_search_tool.func(**args)

        # =========================================
        # 4) 让模型对 HF 搜索结果进行过滤、识别 benchmark
        # =========================================
        second_prompt = [
            {
                "role": "system",
                "content": (
                    "你是一个 benchmark 识别器，会从 HF 搜索结果中挑选属于真正 benchmark 的数据集。"
                    "根据 README、tags、meta 判断其是否属于 benchmark。只输出 JSON。"
                )
            },
            {
                "role": "user",
                "content": f"HF 搜索结果：{hf_results}\n\n请返回识别出的 benchmark 的 repo id 列表（JSON）。"
            }
        ]

        resp2 = await llm.call(second_prompt, bind_post_tools=False)

        try:
            final_ids = json.loads(resp2.content)
        except:
            final_ids = []

        # 组合结果（local + hf）
        final_list = local_results + [
            {
                "bench_name": rid,
                "hf_repo": rid,
                "source": "hf_search"
            }
            for rid in final_ids
        ]

        state.agent_results["bench_search"] = final_list
        log.info("BenchSearchAgent 执行结束")

        return state
