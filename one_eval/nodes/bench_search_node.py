from __future__ import annotations

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.agents.bench_name_suggest_agent import BenchNameSuggestAgent
from one_eval.agents.bench_resolve_agent import BenchResolveAgent
from one_eval.logger import get_logger

log = get_logger("BenchSearchNode")


class BenchSearchNode(BaseNode):
    """
    Step 2 Node:
    1) BenchNameSuggestAgent：根据需求推荐 benchmark 名称列表
    2) BenchResolveAgent：将名称映射为本地 / HF 可用的 benchmark 配置
    """

    def __init__(self):
        self.name = "BenchSearchNode"

    async def run(self, state: NodeState) -> NodeState:
        log.info(f"[{self.name}] 节点开始执行")

        # 1) 先通过 LLM 推荐 bench 名称 + 本地匹配（可能直接满足）
        suggest_agent = BenchNameSuggestAgent(
            tool_manager=None,
            model_name="gpt-4o",
        )
        state = await suggest_agent.run(state)

        # 2) 若需要，对推荐名称做本地表 + HF 精确解析
        resolve_agent = BenchResolveAgent(
            tool_manager=None,
            model_name="gpt-4o",
        )
        state = await resolve_agent.run(state)

        log.info(f"[{self.name}] 执行结束，最终 bench 数量: {len(getattr(state, 'benches', []))}")
        return state
