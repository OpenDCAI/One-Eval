import asyncio
import os
from pathlib import Path
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder
from one_eval.nodes.query_understand_node import QueryUnderstandNode

async def main():
    # === 创建/寻找数据库 ===
    current_file_path = Path(__file__).resolve()
    
    # workflow.py 在 one_eval/graph/ 下，根目录就是往上找 3 层
    # parents[0]是graph, parents[1]是one_eval, parents[2]是项目根目录
    project_root = current_file_path.parents[1] 
    
    # 数据库的绝对路径
    db_dir = project_root / "checkpoints"
    db_path = db_dir / "eval.db"

    # 2. 准备配置 (存档槽位)
    thread_id = "test_ckpt_001"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"使用数据库: {db_path}")
    print(f"线程 ID: {thread_id}")

    # 3. 启动 Checkpointer
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        
        # --- 构建图 ---
        node = QueryUnderstandNode()
        builder = GraphBuilder(state_model=NodeState, entry_point=node.name)
        builder.add_node(node.name, node.run)
        
        # 关键：传入 checkpointer
        graph = builder.build(checkpointer=checkpointer)

        # --- 场景演示 ---
        
        # 检查是否已经有存档（是不是老用户？）
        snapshot = await graph.aget_state(config)
        if snapshot.values:
            print("\n发现历史存档！上次运行结果：")
            print(snapshot.values)
        else:
            print("\n新用户，开始第一次运行...")

        # 运行
        user_input = input("\n请输入指令 (输入 'exit' 退出): ")
        if user_input == 'exit': return

        inputs = NodeState(user_query=user_input)
        
        # ainvoke 会自动处理：读取旧状态 -> 合并新输入 -> 运行 -> 保存新状态
        result = await graph.ainvoke(inputs, config=config)
        
        print(f"\n运行完成，状态已保存。")
        print(f"当前结果: {result.get('target_model')}")

if __name__ == "__main__":
    # 运行两次这个脚本，你会发现第二次能“记得”之前的状态（取决于你的 Node 逻辑）
    asyncio.run(main())
