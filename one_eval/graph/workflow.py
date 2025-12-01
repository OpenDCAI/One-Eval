import asyncio
import os
from pathlib import Path
from langgraph.graph import START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder
from one_eval.toolkits.tool_manager import get_tool_manager
from one_eval.nodes.query_understand_node import QueryUnderstandNode
from one_eval.logger import get_logger

log = get_logger("OneEvalWorkflow")

def build_workflow(checkpointer=None, **kwargs):
    """
    OneEval Workflow。
    """
    tm = get_tool_manager()

    builder = GraphBuilder(state_model=NodeState, entry_point="QueryUnderstandNode")

    # === 注册节点 ===
    node1 = QueryUnderstandNode()          # 变成实例
    builder.add_node(
        name=node1.name,              # "query_understand"
        func=node1.run,                    # 传 run 函数！！！
    )
    builder.add_edge(START, node1.name)
    builder.add_edge(node1.name, END)

    # === 构建图 ===
    graph = builder.build(checkpointer=checkpointer)
    return graph


async def run_demo(user_query: str):
    log.info(f"[workflow] 输入: {user_query}")
    
    # === 创建/寻找数据库 ===
    current_file_path = Path(__file__).resolve()
    
    # workflow.py 在 one_eval/graph/ 下，根目录就是往上找 3 层
    # parents[0]是graph, parents[1]是one_eval, parents[2]是项目根目录
    project_root = current_file_path.parents[2] 
    
    # 数据库的绝对路径
    db_dir = project_root / "checkpoints"
    db_path = db_dir / "eval.db"

    # 自动创建目录 (如果不存在)
    db_dir.mkdir(parents=True, exist_ok=True)
    
    
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:

        # 不启用checkpointer
        # graph = build_workflow(checkpointer=None)
        
        # 启用checkpointer
        graph = build_workflow(checkpointer=checkpointer)
        
        # 初始 state
        initial_state = NodeState(user_query=user_query)

        # === 定义 Thread ID (必须) ===
        # Checkpointer 必须配合 thread_id 使用，否则它不知道把状态存给谁
        config = {"configurable": {"thread_id": "demo_run_001"}}

        # 运行 workflow 
        # 注意：使用 checkpointer 这里必须传入 config
        final_state = await graph.ainvoke(initial_state, config=config)

        log.info(f"[workflow] 最终状态: {final_state}")
    
        return final_state

if __name__ == "__main__":
    asyncio.run(
        run_demo("我想评估我的模型在文本 reasoning 任务上的表现")
    )
