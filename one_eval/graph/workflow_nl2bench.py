import asyncio
from pathlib import Path

from langgraph.graph import START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder

from one_eval.nodes.query_understand_node import QueryUnderstandNode
from one_eval.nodes.bench_search_node import BenchSearchNode

from one_eval.logger import get_logger

log = get_logger("OneEvalWorkflow")


def build_workflow(checkpointer=None, **kwargs):
    """
    OneEval Workflow:
    START → QueryUnderstandNode → BenchSearchNode → END
    """
    builder = GraphBuilder(
        state_model=NodeState,
        entry_point="QueryUnderstandNode",
    )

    # === Node 1: QueryUnderstand ===
    node1 = QueryUnderstandNode()
    builder.add_node(
        name=node1.name,   # "QueryUnderstandNode"
        func=node1.run,
    )

    # === Node 2: BenchSearch ===
    node2 = BenchSearchNode()
    builder.add_node(
        name=node2.name,   # "BenchSearchNode"
        func=node2.run,
    )

    # === 定义执行顺序 ===
    builder.add_edge(START, node1.name)
    builder.add_edge(node1.name, node2.name)
    builder.add_edge(node2.name, END)

    # === 构建图 ===
    graph = builder.build(checkpointer=checkpointer)
    return graph


async def run_demo(user_query: str):
    log.info(f"[workflow] 输入: {user_query}")

    # === Checkpointer root ===
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]

    db_dir = project_root / "checkpoints"
    db_path = db_dir / "eval.db"
    db_dir.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_workflow(checkpointer=checkpointer)

        initial_state = NodeState(user_query=user_query)

        config = {
            "configurable": {"thread_id": "demo_run_001"}
        }

        final_state = await graph.ainvoke(initial_state, config=config)

        log.info(f"[workflow] 最终状态: {final_state}")
        return final_state


if __name__ == "__main__":
    asyncio.run(
        run_demo("我想评估我的模型在文本reasoning领域上的表现")
    )
