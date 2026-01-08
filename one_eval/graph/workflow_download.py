import asyncio
from pathlib import Path
import json
import time
from dataclasses import asdict, is_dataclass

from langgraph.graph import START, END
from langgraph.types import Command

from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder
from one_eval.nodes.download_node import DownloadNode  # 你新写的
from one_eval.utils.checkpoint import get_checkpointer
from one_eval.utils.deal_json import _save_state_json, _restore_state_from_snap
from one_eval.logger import get_logger

log = get_logger("OneEvalWorkflow-DownloadOnly")


def build_download_workflow(checkpointer=None):
    """
    Download-only Workflow:
    START → DownloadNode → END
    """
    builder = GraphBuilder(
        state_model=NodeState,
        entry_point="DownloadNode",
    )

    node = DownloadNode()
    builder.add_node(name=node.name, func=node.run)

    builder.add_edge(START, node.name)
    builder.add_edge(node.name, END)

    return builder.build(checkpointer=checkpointer)


async def run_download_only(thread_id: str, mode: str = "debug"):
    # === ckpt path ===
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    db_path = project_root / "checkpoints" / "eval.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with get_checkpointer(db_path, mode) as checkpointer:
        graph = build_download_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # 1) 必须先确认 ckpt 存在，否则 DownloadNode 没 benches 可用
        snap = None
        try:
            snap = await graph.aget_state(config)
        except Exception:
            snap = None

        has_ckpt = snap is not None and (
            (getattr(snap, "next", None) not in (None, ())) or
            (getattr(snap, "values", None) not in (None, {}))
        )
        log.info(f"[download-only] thread_id={thread_id} has_ckpt={has_ckpt}")

        if not has_ckpt:
            # 也可以选择直接报错/返回，因为你说“从已有 ckpt 跑”
            log.error("[download-only] 未找到 ckpt：请先跑 nl2bench workflow 产出 benches 再执行 DownloadNode。")
            return None

        # 2) 从 ckpt 继续执行 DownloadNode
        snap = await graph.aget_state(config)
        values = getattr(snap, "values", {}) or {}

        state0 = _restore_state_from_snap(values)

        # 关键：传入 state0，强制从 START 跑 DownloadNode，并写入新 ckpt
        out = await graph.ainvoke(state0, config=config)

        if mode == "run":
            results_dir = project_root / "outputs"
            filename = f"download_{thread_id}_{int(time.time())}.json"
            _save_state_json(out, results_dir, filename)

        return out


if __name__ == "__main__":
    asyncio.run(run_download_only(thread_id="demo_run_006", mode="debug"))
