import asyncio
import uuid
from typing import Dict, Any
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder  
from one_eval.nodes.interrupt_node import InterruptNode
from one_eval.nodes.query_understand_node import QueryUnderstandNode
from one_eval.agents.query_understand_agent import QueryUnderstandAgent
from one_eval.toolkits.tool_manager import get_tool_manager
from one_eval.logger import get_logger

log = get_logger("QueryUnderstandNode")
# === 修改 QueryUnderstandNode 类 ===
class QueryUnderstandNode(QueryUnderstandNode):
    def __init__(self) -> None:
        super().__init__()
    
    async def run(self, state: NodeState) -> dict:  
        log.info(f"[{self.name}] 节点开始执行")

        tm = get_tool_manager()

        # 初始化 Agent
        agent = QueryUnderstandAgent(
            tool_manager=tm,
            model_name="gpt-4o", # 或从配置读取
        )

        processed_state = await agent.run(state)
        
        # 提取 Agent 的原始输出字典
        raw_result = getattr(processed_state, "result", {}) or {}

        log.info(f"[{self.name}] Agent 原始输出: {raw_result}")

        # 关键修改：返回字典以触发 LangGraph 状态更新
        # 手动把 Agent 的字段 (key) 映射到 NodeState 的字段 (key)
        return {
            # 映射：Agent 叫 'model_path' -> State 叫 'target_model'
            "target_model": raw_result.get("model_path", []), 
            
            # 映射：Agent 叫 'domain' -> State 叫 'domain'
            "domain": raw_result.get("domain", []),
            
            # 映射：Agent 叫 'specific_benches' -> State 叫 'specific_benches'
            # (这是给 BenchSearchNode 用的关键字段)
            "specific_benches": raw_result.get("specific_benches", []),
            
            # 保留原始结果以备查验
            "result": raw_result 
        }

# === 定义 Validator ===
def expensive_model_validator(state: NodeState):
    """
        规则：如果 target_model 包含昂贵模型，返回警告信息。
    """
    # 定义敏感名单
    EXPENSIVE_MODELS = {"gpt-4-expensive", "claude-3-opus"}
    
    # 获取当前请求的模型列表
    current_models = getattr(state, "target_model", []) or []
    
    # 遍历检查
    for model in current_models:
        if model in EXPENSIVE_MODELS:
            # 发现违规！返回一个字典作为警告
            return {
                "reason": f"预算预警: 检测到使用了高成本模型 [{model}]",
                "severity": "warning",
                "model": model
            }
            
    # 一切正常，返回 None
    return None

# === 使用 GraphBuilder 构建图 ===
def build_graph():
    checkpointer = MemorySaver()
    
    # 定义节点
    query_node = QueryUnderstandNode()
    
    # 定义中断节点
    interrupt_node = InterruptNode(
        name="cost_check",
        validators=[expensive_model_validator], # 挂载刚才写的规则
        success_node="end_node",  # 批准后去哪里？ -> 结束或下一步
        failure_node="end_node"   # 拒绝后去哪里？ -> 结束（通常是终止流程）
    )
    
    # 简略定义终点
    async def end_node_func(state: NodeState):
        print("[EndNode] 流程结束")

    # 构建图
    builder = GraphBuilder(state_model=NodeState, entry_point=query_node.name)
    
    builder.add_node(query_node.name, query_node.run)
    builder.add_node(interrupt_node.name, interrupt_node.run)
    builder.add_node("end_node", end_node_func)

    # 连接边 (Query -> Check)
    builder.add_edge(query_node.name, interrupt_node.name)
    
    return builder.build(checkpointer=checkpointer)

# === 4. 运行测试 (逻辑保持不变) ===
async def run_test():
    # 初始化图
    app = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"=== 开始测试 (Thread: {thread_id}) ===")

    # --- Step 1: 触发中断 ---
    print("\nStep 1: 用户发出请求...")
    
    initial_state = NodeState(user_query="请评估 gpt-4-expensive 在 gsm-8k 上的性能")
    # initial_state = NodeState(user_query="请评估 claude-3-opus 在 gsm-8k 上的性能")
    
    async for event in app.astream(initial_state, config=config):
        pass
    
    # --- Step 2: 检查中断 ---
    snapshot = app.get_state(config)
    if snapshot.next:
        print(f"\n流程暂停于: {snapshot.next}")
        if snapshot.tasks:
            val = snapshot.tasks[0].interrupts[0].value
            print(f"警告: {val.get('reason')}")
    else:
        print("错误：未触发中断")
        return

    # --- Step 3: 批准 ---
    print("\nStep 3: 用户反馈...")
    user_input = input("请输入是否同意(Y/N)：")
    if user_input == 'Y' or 'y' or 'Yes' or 'yes' or 'YES':
        print("用户批准")
        approval = Command(resume={"action": "approve", "reason": "Budget OK"})
    else:
        print("用户拒绝")
        approval = Command(resume={"action": "refuse", "reason": "Budget too expensive"})
    
    async for event in app.astream(approval, config=config):
        pass

    # --- Step 4: 验证 ---
    final_state = app.get_state(config).values
    print(f"\n最终状态: TargetModel={final_state.get('target_model')}")
    
if __name__ == "__main__":
    asyncio.run(run_test())
