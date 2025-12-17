import inspect
import json
from typing import Callable, List, Dict, Any, Optional, Union

from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.toolkits.tool_manager import get_tool_manager
from one_eval.agents.human_in_loop_agent import HumanInLoopAgent
from one_eval.logger import get_logger

log = get_logger("InterruptNode")


class InterruptNode(BaseNode):
    def __init__(
        self,
        name: str,
        validators: List[Callable[[NodeState], Union[Optional[Dict], Any]]],
        success_node: str,
        failure_node: str = "__end__",
        rewind_nodes: Optional[List[str]] = None,
        model_name: str = "gpt-4o",
        node_docs: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name=name, tools=None)
        self.validators = validators
        self.success_node = success_node
        self.failure_node = failure_node
        self.rewind_nodes = rewind_nodes or []
        self.model_name = model_name
        # 节点说明：{"QueryUnderstandNode": "...", "BenchSearchNode": "...", ...}
        self.node_docs = node_docs or {}

    async def run(self, state: NodeState, config: RunnableConfig) -> Command:
        log.info(f"开始执行安全/人工检查...")

        approved_ids = getattr(state, "approved_warning_ids", []) or []

        for validator in self.validators:
            validator_id = f"{self.name}_validator_{validator.__name__}"

            if validator_id in approved_ids:
                log.info(f"规则 {validator_id} 已在白名单中，跳过。")
                continue

            try:
                result = validator(state)
                if inspect.iscoroutine(result):
                    check_result = await result
                else:
                    check_result = result
            except Exception as e:
                log.error(f"[{self.name}] Validator 执行出错: {e}")
                check_result = {
                    "type": "error",
                    "message": f"校验器执行异常: {str(e)}",
                }

            if not check_result:
                continue

            log.warning(f"触发拦截规则: {check_result}")

            # === 中断等待 human_input ===
            user_input = interrupt(
                {
                    "node": self.name,
                    "validator_id": validator_id,
                    "check_result": check_result,
                }
            )

            log.info(f"收到用户反馈: {user_input}")

            history = getattr(state, "llm_history", []) or []
            try:
                content_str = json.dumps(user_input, ensure_ascii=False)
            except Exception:
                content_str = str(user_input)
            history.append({"role": "user", "content": content_str})

            # --------- 构造 node_io 记录（可以按需扩展）---------
            node_io: Dict[str, Any] = {
                "agent_results": getattr(state, "agent_results", {}),
                "benches": getattr(state, "benches", []),
                "bench_info": getattr(state, "bench_info", {}),
                "task_domain": getattr(state, "task_domain", None),
                "user_query": getattr(state, "user_query", None),
            }

            # === 调 HumanInLoopAgent 决策 ===
            try:
                tm = get_tool_manager()
                hitl_agent = HumanInLoopAgent(
                    tool_manager=tm,
                    model_name=self.model_name,
                )

                decision = await hitl_agent.run(
                    state=state,
                    human_input=user_input,
                    check_result=check_result,
                    current_node=self.name,
                    allowed_nodes=self.rewind_nodes,
                    validator_id=validator_id,
                    node_docs=self.node_docs,
                    node_io=node_io,
                )

            except Exception as e:
                log.error(
                    f"HumanInLoopAgent 执行失败，走兜底拒绝逻辑: {e}",
                    exc_info=True,
                )
                return self._handle_rejection(
                    state,
                    reason=f"HumanInLoopAgent 执行失败: {str(e)}",
                    history=history,
                )

            action = decision.get("action") or "continue"
            target_node = decision.get("target_node")
            state_update = decision.get("state_update") or {}
            approve_validator = decision.get("approve_validator", True)

            new_approved_ids = list(approved_ids)
            if approve_validator and validator_id not in new_approved_ids:
                new_approved_ids.append(validator_id)

            update_dict: Dict[str, Any] = {
                "approved_warning_ids": new_approved_ids,
                "waiting_for_human": False,
                "human_feedback": content_str,
                "llm_history": history,
            }
            if isinstance(state_update, dict):
                update_dict.update(state_update)
                log.info(f"状态更新: {update_dict}")

            if action == "goto_node" and target_node:
                log.info(f"决策: 回跳到 {target_node}")
                return Command(goto=target_node, update=update_dict)

            if action == "continue":
                log.info(f"决策: 继续 -> {self.success_node}")
                return Command(goto=self.success_node, update=update_dict)

            log.warning(
                f"决策异常 action={action}, target_node={target_node}，跳转 failure_node"
            )
            return self._handle_rejection(
                state,
                reason=f"非法决策 action={action}, target_node={target_node}",
                history=history,
            )

        log.info(f"所有检查通过，跳转 -> {self.success_node}")
        return Command(goto=self.success_node, update={})

    def _handle_rejection(
        self,
        state: NodeState,
        reason: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Command:
        log.info(f"操作被拒绝/失败，跳转 -> {self.failure_node}")

        rejection_msg = {
            "role": "user",
            "content": f"操作被拦截/拒绝。原因: {reason}。请尝试其他方案或终止任务。",
        }

        if history is None:
            history = getattr(state, "llm_history", []) or []
        history = history + [rejection_msg]

        update_dict = {
            "human_feedback": reason,
            "waiting_for_human": False,
            "llm_history": history,
            "error_flag": True,
            "error_msg": reason,
        }

        return Command(
            goto=self.failure_node,
            update=update_dict,
        )
