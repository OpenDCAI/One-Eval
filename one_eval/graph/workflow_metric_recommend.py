import asyncio
from pathlib import Path
import json
import time

from langgraph.graph import START, END

from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder

from one_eval.nodes.metric_recommend_node import MetricRecommendNode
from one_eval.nodes.score_calc_node import ScoreCalcNode
from one_eval.nodes.report_gen_node import ReportGenNode

from one_eval.utils.checkpoint import get_checkpointer
from one_eval.utils.deal_json import _save_state_json, _restore_state_from_snap
from one_eval.logger import get_logger

log = get_logger("OneEvalWorkflow-MetricRecommend")


def build_metric_recommend_workflow(checkpointer=None):
    """
    Metric Recommend Workflow:
    START -> MetricRecommendNode -> ScoreCalcNode -> ReportGenNode -> END
    (Pre-requisite: Eval Workflow must be completed)
    """
    builder = GraphBuilder(
        state_model=NodeState,
        entry_point="MetricRecommendNode",
    )

    node_metric = MetricRecommendNode()
    node_score = ScoreCalcNode()
    node_report = ReportGenNode()

    builder.add_node(name=node_metric.name, func=node_metric.run)
    builder.add_node(name=node_score.name, func=node_score.run)
    builder.add_node(name=node_report.name, func=node_report.run)

    builder.add_edge(START, node_metric.name)
    builder.add_edge(node_metric.name, node_score.name)
    builder.add_edge(node_score.name, node_report.name)
    builder.add_edge(node_report.name, END)

    return builder.build(checkpointer=checkpointer)


async def run_metric_recommend(thread_id: str, mode: str = "debug"):
    # === ckpt path ===
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    db_path = project_root / "checkpoints" / "eval.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with get_checkpointer(db_path, mode) as checkpointer:
        graph = build_metric_recommend_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # 1) 必须有 ckpt (从 Eval 阶段继承)
        snap = None
        try:
            snap = await graph.aget_state(config)
        except Exception:
            snap = None

        has_ckpt = snap is not None and (
            (getattr(snap, "next", None) not in (None, ())) or
            (getattr(snap, "values", None) not in (None, {}))
        )
        log.info(f"[metric-recommend] thread_id={thread_id} has_ckpt={has_ckpt}")

        if not has_ckpt:
            log.error("[metric-recommend] 未找到 ckpt：请先运行 workflow_eval.py。")
            return None

        # 2) 恢复 State
        snap = await graph.aget_state(config)
        values = getattr(snap, "values", {}) or {}
        state0 = _restore_state_from_snap(values)

        # 3) 执行 Workflow
        out = await graph.ainvoke(state0, config=config)

        if mode == "run":
            results_dir = project_root / "outputs"
            filename = f"metric_recommend_{thread_id}_{int(time.time())}.json"
            _save_state_json(out, results_dir, filename)

        return out

if __name__ == "__main__":
    asyncio.run(run_metric_recommend(thread_id="demo_run_hyh_24", mode="run"))
