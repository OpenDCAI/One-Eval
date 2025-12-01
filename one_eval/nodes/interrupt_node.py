import inspect
from typing import Callable, List, Dict, Any, Optional, Union
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig
from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.logger import get_logger

log = get_logger("InterruptNode")

class InterruptNode(BaseNode):
    """
    InterruptNode:
    用于在流程中插入“人机交互”或“安全检查”断点。
    
    机制：
    1. 运行 validators 检查 State。
    2. 如果触发拦截条件，在 Node 内部调用 interrupt() 挂起。
    3. 恢复后，通过返回 Command 对象控制跳转方向 (success_node 或 failure_node)。
    """

    def __init__(
        self, 
        name: str, 
        validators: List[Callable[[NodeState], Union[Optional[Dict], Any]]], 
        success_node: str,
        failure_node: str = "__end__"
    ):
        """
        Args:
            name: 节点名称
            validators: 校验函数列表。支持同步(def)或异步(async def)。
            success_node: 校验通过或用户批准后的跳转节点名称。
            failure_node: 用户拒绝后的跳转节点名称。
        """
        super().__init__(name=name, tools=None)
        self.validators = validators
        self.success_node = success_node
        self.failure_node = failure_node

    async def run(self, state: NodeState, config: RunnableConfig) -> Command:
        """
        执行校验逻辑并返回 Command 对象控制流向。
        """
        log.info(f"[{self.name}] 开始执行安全扫描...")

        # 获取已批准的警告列表 (更新 NodeState : approved_warning_ids 字段)
        approved_ids = getattr(state, "approved_warning_ids", [])

        for i, validator in enumerate(self.validators):
            # 生成一个唯一 ID 标识当前校验器
            validator_id = f"{self.name}_validator_{validator.__name__}"

            # 如果该规则已被批准，直接跳过
            if validator_id in approved_ids:
                log.info(f"[{self.name}] 规则 {validator_id} 已在白名单中，跳过。")
                continue

            try:
                result = validator(state)
                if inspect.iscoroutine(result):
                    check_result = await result
                else:
                    check_result = result
            except Exception as e:
                log.error(f"[{self.name}] Validator 执行出错: {e}")
                check_result = {"type": "error", "message": f"校验器执行异常: {str(e)}"}

            if check_result:
                log.warning(f"[{self.name}] 触发拦截规则: {check_result.get('reason', 'Unknown')}")
                
                # 触发中断
                user_decision = interrupt(check_result)
                
                # === 恢复后的逻辑 ===
                log.info(f"[{self.name}] 收到用户决策: {user_decision}")

                if isinstance(user_decision, dict) and user_decision.get("action") == "approve":
                    log.info(f"[{self.name}] 用户批准操作。正在更新状态并重新检查...")
                    
                    # 批准后，更新 State 并重启当前 Node
                    # 这样做的目的是将“批准”持久化，防止下次重跑时丢失
                    new_approved_ids = approved_ids + [validator_id]
                    
                    return Command(
                        goto=self.name,  # 重启自己
                        update={"approved_warning_ids": new_approved_ids}
                    )
                
                else:
                    # 拒绝逻辑保持不变
                    reason = "用户手动拒绝了该操作。"
                    if isinstance(user_decision, dict):
                        reason = user_decision.get("reason", reason)
                    elif isinstance(user_decision, str):
                        reason = user_decision
                    
                    return self._handle_rejection(state, reason)

        log.info(f"[{self.name}] 所有检查通过，跳转至 -> {self.success_node}")
        return Command(goto=self.success_node, update={})

    def _handle_rejection(self, state: NodeState, reason: str) -> Command:
        """
        处理拒绝逻辑：构建更新数据并跳转到 failure_node。
        """
        log.info(f"[{self.name}] 操作被拒绝，跳转至 -> {self.failure_node}")

        # 构造拒绝消息
        rejection_msg = {
            "role": "user", 
            "content": f"操作被拦截/拒绝。原因: {reason}。请尝试其他方案或终止任务。"
        }

        # 构造 State 更新字典
        updated_history = state.llm_history + [rejection_msg] if isinstance(state.llm_history, list) else [rejection_msg]

        update_dict = {
            "human_feedback": reason,
            "waiting_for_human": False,
            "llm_history": updated_history
        }

        # 返回 Command 跳转到失败节点，并应用状态更新
        return Command(
            goto=self.failure_node, 
            update=update_dict
        )
