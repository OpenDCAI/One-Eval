from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState
from one_eval.logger import get_logger

log = get_logger("QueryUnderstandAgent")


class QueryUnderstandAgent(CustomAgent):
    """
    Step1 Agent: 理解用户自然语言需求 → 输出结构化信息
    """

    @property
    def role_name(self) -> str:
        return "QueryUnderstandAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "query_understand.system"

    @property
    def task_prompt_template_name(self) -> str:
        return "query_understand.task"


    async def run(self, state: NodeState) -> NodeState:
        user_query = getattr(state, "user_query", "")
        if not user_query:
            log.error("QueryUnderstandAgent 需要 state.user_query")
            return state
        
        # -------- 使用统一 prompt 管理 --------
        sys_prompt = self.get_prompt(self.system_prompt_template_name)
        task_prompt = self.get_prompt(
            self.task_prompt_template_name,
            user_query=user_query,
        )

        msgs = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=task_prompt),
        ]

        llm = self.create_llm(state)
        resp = await llm.call(msgs, bind_post_tools=False)
        parsed = self.parse_result(resp.content)

        state.result[self.role_name] = parsed
        log.info(f"{self.role_name} 结果: {parsed}")

        return state
