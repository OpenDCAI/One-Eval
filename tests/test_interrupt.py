# test for InterruptNode
import asyncio
import uuid
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from one_eval.nodes.interrupt_node import InterruptNode
from one_eval.core.state import NodeState

# --- 定义两个 validators ---
async def mock_validator(state: NodeState):
    if "删除" in state.user_query:
        return {
            "id": "WARN_001", 
            "reason": "检测到高危操作：数据删除"
        }
    return None

async def sensitive_topic_validator(state: NodeState):
    if "数据库" in state.user_query:
        return {
            "id": "WARN_SENSITIVE_001",
            "type": "content_warning",
            "reason": "检测到敏感词：数据库",
            "risk_level": "MEDIUM"
        }
    return None

# 定义几个简单的节点，用于跳转测试
async def input_node(state: NodeState):
    print(f"[InputNode] 接收到输入: {state.user_query}")
    return {"user_query": state.user_query}

async def success_node(state: NodeState):
    print("[SuccessNode] 操作执行成功！")
    return {"final_result": "Success"}

async def failure_node(state: NodeState):
    print("[FailureNode] 操作被拒绝。")
    return {"final_result": "Blocked"}

# 将 validators 合并为列表，后续传给 interrupt node
ALL_VALIDATORS = [
    mock_validator,
    sensitive_topic_validator
]

# --- Graph Build --- 
# 这里用的 LangGraph 官方建图器
# 目前 One-Eval 库的 graphbuilder 以及 DataFlow-Agent 的 GenericGraphBuilder 不支持 **kwargs 传参
# 无法接收 checkpointer 以及 interrupt 函数
# 后续可做改进

def build_graph():
    workflow = StateGraph(NodeState)
    
    # 创建中断节点
    check_node = InterruptNode(
        name="security_check",
        validators=ALL_VALIDATORS,
        success_node="success_node",
        failure_node="failure_node"
    )

    workflow.add_node("input_node", input_node)
    workflow.add_node("security_check", check_node.run)
    workflow.add_node("success_node", success_node)
    workflow.add_node("failure_node", failure_node)

    workflow.add_edge("input_node", "security_check")

    workflow.set_entry_point("input_node")
    
    return workflow.compile(checkpointer=MemorySaver())

# --- 测试函数 ---
async def run_test():
    app = build_graph()
    
    # 唯一的 Thread ID，用于记忆状态
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"启动工作流 (Thread: {thread_id})")
    print("=" * 50)

    # 第一阶段：运行直到触发中断
    await app.ainvoke(
        NodeState(user_query="请帮我删除数据库。"), 
        config=config
    )

    # 检查当前状态
    # while 循环用于检测所有的 validators 的意见
    while True:
        # 获取当前快照
        snapshot = app.get_state(config)

        # 如果没有下一步 (next)，说明流程结束
        if not snapshot.next:
            print("\n流程全部完成！")
            # 这里可以打印最终结果
            final_state = snapshot.values
            print("-" * 30)
            print(f"最终执行结果: {final_state.get('final_result')}")
            # 这里在 NodeState 中添加了字段 approved_warning_ids
            # 目的是用于统计 InterruptNode 中人工同意通过的 validator 名单，避免被重复执行 
            print(f"最终白名单 (Approved IDs): {final_state.get('approved_warning_ids')}")
            print(f"完整 State: {final_state}")
            print("-" * 30)
            break

        # 否则，进程中断，获取中断详情
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            interrupt_value = snapshot.tasks[0].interrupts[0].value
            # value 可能是 dict 或 string 
            reason = interrupt_value.get('reason') if isinstance(interrupt_value, dict) else str(interrupt_value)
            
            print(f"\n工作流已暂停 (节点: {snapshot.next})")
            print(f"拦截原因: {reason}")

            # 人机交互：获取用户输入
            user_input = input("请输入是否同意 (y/n) > ")
            action = "approve" if user_input.strip().lower() == "y" else "decline"

            print(f"发送指令: {action} ...")
            
            # 恢复执行 (Resume)
            # 注意：这里不需要接收返回值，循环回到开头会再次检查 get_state
            await app.ainvoke(
                Command(resume={"action": action}),
                config=config
            )
        else:
            # 若有 next 但没有 interrupts，通常是系统处于调度间隙
            print("等待调度中...")
            break

if __name__ == "__main__":
    asyncio.run(run_test())
