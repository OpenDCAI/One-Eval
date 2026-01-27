from __future__ import annotations

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.nodes.bench_name_suggest_node import BenchNameSuggestNode
from one_eval.agents.bench_resolve_agent import BenchResolveAgent
from one_eval.logger import get_logger

log = get_logger("BenchSearchNode")


class BenchSearchNode(BaseNode):
    """
    Step 2 Node:
    1) BenchNameSuggestNode：根据需求检索推荐 benchmark 名称列表（TF-IDF 或 RAG）
    2) BenchResolveAgent：将名称映射为本地 / HF 可用的 benchmark 配置
    """

    def __init__(self, use_rag: bool = False):
        """
        Args:
            use_rag: 是否使用 RAG 模式进行 benchmark 检索，默认 False 使用 TF-IDF
        """
        self.name = "BenchSearchNode"
        self.use_rag = use_rag

    async def run(self, state: NodeState) -> NodeState:
        # 1) 通过 BenchNameSuggestNode 检索 benchmark（不调用 LLM）
        suggest_node = BenchNameSuggestNode(use_rag=self.use_rag)
        state = await suggest_node.run(state)

        # 2) 若需要，对推荐名称做本地表 + HF 精确解析
        resolve_agent = BenchResolveAgent(
            tool_manager=None,
            model_name="gpt-4o",
        )
        state = await resolve_agent.run(state)

        log.info(f"执行结束，最终 bench 数量: {len(getattr(state, 'benches', []))}")
        return state
