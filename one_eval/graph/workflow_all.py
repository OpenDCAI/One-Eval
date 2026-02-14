import asyncio
from pathlib import Path
import json
import time
from dataclasses import asdict, is_dataclass

from langgraph.graph import START, END
from langgraph.types import Command

from one_eval.core.state import NodeState, ModelConfig
from one_eval.core.graph import GraphBuilder

# Import all nodes from previous workflows
from one_eval.nodes.query_understand_node import QueryUnderstandNode
from one_eval.nodes.bench_search_node import BenchSearchNode
from one_eval.nodes.interrupt_node import InterruptNode
from one_eval.nodes.dataset_structure_node import DatasetStructureNode
from one_eval.nodes.bench_config_recommend_node import BenchConfigRecommendNode
from one_eval.nodes.download_node import DownloadNode
from one_eval.nodes.dataset_keys_node import DatasetKeysNode
from one_eval.nodes.bench_task_infer_node import BenchTaskInferNode
from one_eval.nodes.dataflow_eval_node import DataFlowEvalNode
from one_eval.nodes.pre_eval_review_node import PreEvalReviewNode
from one_eval.nodes.metric_recommend_node import MetricRecommendNode
from one_eval.nodes.score_calc_node import ScoreCalcNode
from one_eval.nodes.report_gen_node import ReportGenNode

from one_eval.utils import node_docs, validators
from one_eval.utils.checkpoint import get_checkpointer
from one_eval.utils.deal_json import _save_state_json, _restore_state_from_snap
from one_eval.logger import get_logger

log = get_logger("OneEvalWorkflow-All")


def _route_after_eval(state: NodeState) -> str:
    benches = getattr(state, "benches", None) or []
    cursor = int(getattr(state, "eval_cursor", 0) or 0)
    if cursor < len(benches):
        return "DataFlowEvalNode"
    return "MetricRecommendNode"


def build_complete_workflow(checkpointer=None):
    """
    Complete OneEval Workflow:
    
    Phase 1: NL2Bench (Interactive)
    START → QueryUnderstandNode → BenchSearchNode → HumanReviewNode(Interrupt)
    
    Phase 2: Download & Prep
    → DatasetStructureNode → BenchConfigRecommendNode → DownloadNode
    
    Phase 3: Task Inference
    → DatasetKeysNode → BenchTaskInferNode
    
    Phase 4: Evaluation
    → DataFlowEvalNode

    Phase 5: Metric Recommend
    → MetricRecommendNode

    Phase 6: Score Calc
    → ScoreCalcNode

    Phase 7: Report Generation
    → ReportGenNode → END
    """
    builder = GraphBuilder(
        state_model=NodeState,
        entry_point="QueryUnderstandNode",
    )

    # === Phase 1: NL2Bench ===
    node_query = QueryUnderstandNode()
    node_search = BenchSearchNode()
    node_review = InterruptNode(
        name="HumanReviewNode",
        validators=[validators.benches_manual_review],
        success_node="DatasetStructureNode",  # Pass -> Phase 2
        failure_node=END,  # Reject -> End
        rewind_nodes=["QueryUnderstandNode", "BenchSearchNode"],
        model_name="gpt-4o",
        node_docs=node_docs,
    )

    builder.add_node(name=node_query.name, func=node_query.run)
    builder.add_node(name=node_search.name, func=node_search.run)
    builder.add_node(name=node_review.name, func=node_review.run)

    # === Phase 2: Download ===
    node_struct = DatasetStructureNode()
    node_config = BenchConfigRecommendNode()
    node_download = DownloadNode()

    builder.add_node(name=node_struct.name, func=node_struct.run)
    builder.add_node(name=node_config.name, func=node_config.run)
    builder.add_node(name=node_download.name, func=node_download.run)

    # === Phase 3: Task Infer ===
    node_keys = DatasetKeysNode()
    node_infer = BenchTaskInferNode()

    builder.add_node(name=node_keys.name, func=node_keys.run)
    builder.add_node(name=node_infer.name, func=node_infer.run)

    # === Phase 4: Eval ===
    node_pre_eval_review = PreEvalReviewNode()
    builder.add_node(name=node_pre_eval_review.name, func=node_pre_eval_review.run)

    node_eval = DataFlowEvalNode()
    builder.add_node(name=node_eval.name, func=node_eval.run)

    # === Phase 5: Metric Recommend ===
    node_metric = MetricRecommendNode()
    builder.add_node(name=node_metric.name, func=node_metric.run)

    # === Phase 6: Score Calc ===
    node_score = ScoreCalcNode()
    builder.add_node(name=node_score.name, func=node_score.run)

    node_report = ReportGenNode()
    builder.add_node(name=node_report.name, func=node_report.run)

    # === Edges ===
    # Phase 1
    builder.add_edge(START, node_query.name)
    builder.add_edge(node_query.name, node_search.name)
    builder.add_edge(node_search.name, node_review.name)
    
    # InterruptNode handles edge to success_node (DatasetStructureNode) or failure_node (END)
    # But we need to explicitly define the edge from success_node to next if InterruptNode returns state
    # Wait, InterruptNode returns Command(goto=success_node).
    # So we just need to ensure success_node is in the graph. It is.

    # Phase 2
    builder.add_edge(node_struct.name, node_config.name)
    builder.add_edge(node_config.name, node_download.name)

    # Phase 2 -> Phase 3
    builder.add_edge(node_download.name, node_keys.name)

    # Phase 3
    builder.add_edge(node_keys.name, node_infer.name)

    # Phase 3 -> Phase 4
    builder.add_edge(node_infer.name, node_pre_eval_review.name)

    # Phase 4 -> Phase 5 -> Phase 6 -> End
    builder.add_conditional_edge(node_eval.name, _route_after_eval)
    builder.add_edge(node_metric.name, node_score.name)
    builder.add_edge(node_score.name, node_report.name)
    builder.add_edge(node_report.name, END)

    return builder.build(checkpointer=checkpointer)


async def run_full_pipeline(user_query: str, thread_id: str = "demo_full_run", mode="run"):
    # === ckpt path ===
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    db_path = project_root / "checkpoints" / "eval.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with get_checkpointer(db_path, mode) as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # 1. Check existing state
        snap = None
        try:
            snap = await graph.aget_state(config)
        except Exception:
            snap = None

        has_ckpt = snap is not None and (
            (getattr(snap, "next", None) not in (None, ())) or
            (getattr(snap, "values", None) not in (None, {}))
        )
        
        log.info(f"[workflow-all] thread_id={thread_id} has_ckpt={has_ckpt}")

        if not has_ckpt:
            # Start fresh
            log.info("Starting new workflow...")
            initial_state = NodeState(
                user_query=user_query,
                # Inject default model config for testing if needed, 
                # normally this might come from user_query or external config
                target_model=ModelConfig(
                    model_name_or_path="/mnt/DataFlow/models/Qwen2.5-7B-Instruct",
                    tensor_parallel_size=1,
                    max_tokens=2048
                )
            )
            out = await graph.ainvoke(initial_state, config=config)
        else:
            # Resume (e.g. from Human Interrupt)
            # Check if we are at interrupt
            if snap and snap.next and ("HumanReviewNode" in snap.next or "PreEvalReviewNode" in snap.next):
                log.info("Resuming from Human Review...")
                # You can provide feedback here if automating, or CLI input
                # For demo purposes, we assume acceptance
                out = await graph.ainvoke(
                    Command(resume="approved"), 
                    config=config
                )
            else:
                log.info("Resuming existing workflow...")
                out = await graph.ainvoke(None, config=config)

        if mode == "run":
            results_dir = project_root / "outputs"
            filename = f"full_run_{thread_id}_{int(time.time())}.json"
            _save_state_json(out, results_dir, filename)
        
        return out

if __name__ == "__main__":
    # Example usage
    import sys
    
    query = "我想评估我的模型在MATH数据集上的表现"
    if len(sys.argv) > 1:
        query = sys.argv[1]
        
    asyncio.run(run_full_pipeline(
        user_query=query,
        thread_id="demo_run_hyh_21", 
        mode="run"
    ))
