from __future__ import annotations

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.agents.score_calc_agent import ScoreCalcAgent
from one_eval.toolkits.tool_manager import get_tool_manager
from one_eval.logger import get_logger

log = get_logger("ScoreCalcNode")


class ScoreCalcNode(BaseNode):
    def __init__(self):
        self.name = "ScoreCalcNode"

    async def run(self, state: NodeState) -> NodeState:
        log.info(f"[{self.name}] 节点开始执行")

        tm = get_tool_manager()
        agent = ScoreCalcAgent(
            tool_manager=tm,
            model_name=None,
            temperature=0.0,
        )

        state = await agent.run(state)

        log.info(f"[{self.name}] 执行结束，eval_results 已更新")
        return state
