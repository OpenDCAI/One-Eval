from __future__ import annotations
from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.agents.query_understand_agent import QueryUnderstandAgent
from one_eval.toolkits.tool_manager import get_tool_manager
from one_eval.logger import get_logger

log = get_logger("QueryUnderstandNode")


class QueryUnderstandNode(BaseNode):
    """
    Step 1 Node:
    调用 QueryUnderstandAgent, 将用户 NL → 结构化需求。
    """
    
    def __init__(self):
        self.name = "QueryUnderstandNode"

    async def run(self, state: NodeState) -> NodeState:
        log.info(f"[{self.name}] 节点开始执行")

        # 获取全局 ToolManager
        tm = get_tool_manager()

        # 创建 agent
        agent = QueryUnderstandAgent(
            tool_manager=tm,
            model_name="gpt-4o",
        )

        new_state = await agent.run(state)

        log.info(f"[{self.name}] 执行结束，输出结果: {new_state.result[agent.role_name]}")
        
        return new_state
