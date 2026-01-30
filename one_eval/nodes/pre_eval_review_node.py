import json
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState


class PreEvalReviewNode(BaseNode):
    def __init__(self):
        super().__init__(name="PreEvalReviewNode", tools=None)

    async def run(self, state: NodeState, config: RunnableConfig) -> Command:
        approved_ids: List[str] = getattr(state, "approved_warning_ids", []) or []
        confirm_id = f"{self.name}_confirm"
        if confirm_id in approved_ids:
            return Command(goto="DataFlowEvalNode", update={})

        payload: Dict[str, Any] = {
            "node": self.name,
            "kind": "pre_eval_review",
        }
        user_input = interrupt(payload)

        try:
            content_str = json.dumps(user_input, ensure_ascii=False)
        except Exception:
            content_str = str(user_input)

        new_approved = list(approved_ids)
        new_approved.append(confirm_id)

        update_dict: Dict[str, Any] = {
            "approved_warning_ids": new_approved,
            "waiting_for_human": False,
            "human_feedback": content_str,
        }
        return Command(goto="DataFlowEvalNode", update=update_dict)

