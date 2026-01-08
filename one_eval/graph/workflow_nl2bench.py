import asyncio
from pathlib import Path
import json
import time
from dataclasses import asdict, is_dataclass

from langgraph.graph import START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder
from one_eval.nodes.query_understand_node import QueryUnderstandNode
from one_eval.nodes.bench_search_node import BenchSearchNode
from one_eval.nodes.interrupt_node import InterruptNode
from one_eval.utils import node_docs, validators
from one_eval.utils.checkpoint import get_checkpointer
from one_eval.logger import get_logger

log = get_logger("OneEvalWorkflow")


def build_workflow(checkpointer=None, **kwargs):
    """
    OneEval Workflow:
    START → QueryUnderstandNode → BenchSearchNode → HumanReviewNode(Interrupt) → END
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

    interrupt_node = InterruptNode(
        name="HumanReviewNode",
        validators=[
            validators.benches_manual_review,
        ],
        success_node=END,                 # 审核通过就直接结束
        failure_node=END,                 # 审核拒绝也可以直接结束，或者换成别的节点名
        rewind_nodes=[
            "QueryUnderstandNode",
            "BenchSearchNode",
        ],
        model_name="gpt-4o",
        node_docs=node_docs,
    )

    builder.add_node(
        name=interrupt_node.name,
        func=interrupt_node.run,
    )

    # === 定义执行顺序 ===
    builder.add_edge(START, node1.name)
    builder.add_edge(node1.name, node2.name)
    builder.add_edge(node2.name, interrupt_node.name)
    builder.add_edge(interrupt_node.name, END)

    # === 构建图 ===
    graph = builder.build(checkpointer=checkpointer)
    return graph


def _json_safe(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [ _json_safe(x) for x in obj ]
    if isinstance(obj, dict):
        return { k: _json_safe(v) for k, v in obj.items() }
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _save_state_json(data, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, ensure_ascii=False, indent=2)
    log.info(f"[workflow] 结果已保存: {path}")


async def run_demo(user_query: str, mode="debug"):
    log.info(f"[workflow] 输入: {user_query}")

    # === Checkpointer root ===
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]

    db_dir = project_root / "checkpoints"
    db_path = db_dir / "eval.db"
    db_dir.mkdir(parents=True, exist_ok=True)

    async with get_checkpointer(db_path, mode) as checkpointer: 
        graph = build_workflow(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "demo_run_006"}}

        # 1) 先看看这个 thread_id 有没有历史状态（有没有 ckpt）
        snap = None
        try:
            snap = await graph.aget_state(config)
        except Exception:
            snap = None

        has_ckpt = snap is not None and (
        (getattr(snap, "next", None) not in (None, ())) or
        (getattr(snap, "values", None) not in (None, {}))
        )

        print("[debug] has_ckpt =", has_ckpt)

        if not has_ckpt:
            # 2) 没有 ckpt：第一次运行，跑到 interrupt
            initial_state = NodeState(user_query=user_query)
            out = await graph.ainvoke(initial_state, config=config)
            print(out.get("__interrupt__"))
            if mode == "run":
                thread_id = config.get("configurable", {}).get("thread_id", "default")
                results_dir = project_root / "outputs"
                filename = f"nl2bench_{thread_id}_{int(time.time())}.json"
                _save_state_json(out, results_dir, filename)
            return out

        # 3) 有 ckpt：直接 resume（必须是上次停在 interrupt 才有意义）
        out = await graph.ainvoke(
            Command(resume="我其实想侧重文本在一些通用知识上覆盖面是否够广，以及看看这个模型能否做一些简单的尝试推理"),
            config=config
        )
        if mode == "run":
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            results_dir = project_root / "outputs"
            filename = f"nl2bench_{thread_id}_{int(time.time())}.json"
            _save_state_json(out, results_dir, filename)
        return out


if __name__ == "__main__":
    asyncio.run(
        run_demo("我想评估我的模型在文本reasoning领域上的表现", mode="run"),
    )
