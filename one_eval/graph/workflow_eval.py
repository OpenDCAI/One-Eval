import asyncio
from pathlib import Path
import json
import time

from langgraph.graph import START, END

from one_eval.core.state import NodeState, ModelConfig
from one_eval.core.graph import GraphBuilder

# 导入节点
from one_eval.nodes.dataflow_eval_node import DataFlowEvalNode

from one_eval.utils.checkpoint import get_checkpointer
from one_eval.utils.deal_json import _save_state_json, _restore_state_from_snap
from one_eval.logger import get_logger

log = get_logger("OneEvalWorkflow-Eval")


def build_eval_workflow(checkpointer=None):
    """
    Eval Workflow:
    START -> DataFlowEvalNode -> END
    (Pre-requisite: Task Infer Workflow must be completed)
    """
    builder = GraphBuilder(
        state_model=NodeState,
        entry_point="DataFlowEvalNode",
    )

    node = DataFlowEvalNode()

    builder.add_node(name=node.name, func=node.run)

    builder.add_edge(START, node.name)
    builder.add_edge(node.name, END)

    return builder.build(checkpointer=checkpointer)


async def run_eval(thread_id: str, mode: str = "debug"):
    # === ckpt path ===
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    db_path = project_root / "checkpoints" / "eval.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with get_checkpointer(db_path, mode) as checkpointer:
        graph = build_eval_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # 1) 必须有 ckpt (从 Task Infer 阶段继承)
        snap = None
        try:
            snap = await graph.aget_state(config)
        except Exception:
            snap = None

        has_ckpt = snap is not None and (
            (getattr(snap, "next", None) not in (None, ())) or
            (getattr(snap, "values", None) not in (None, {}))
        )
        log.info(f"[eval] thread_id={thread_id} has_ckpt={has_ckpt}")

        if not has_ckpt:
            log.error("[eval] 未找到 ckpt：请先运行 workflow_task_infer.py。")
            return None

        # 2) 恢复 State
        snap = await graph.aget_state(config)
        values = getattr(snap, "values", {}) or {}
        
        state0 = _restore_state_from_snap(values)
        
        # 3) 配置测试模型 (仅用于 Debug)
        # 在实际运行中，target_model 应该在更早的阶段（如 User Request 解析）被设置
        # 这里为了演示，我们手动注入一个配置
        if not state0.target_model:
            # 示例：使用 Qwen2.5-7B-Instruct
            # 请根据您的实际环境修改路径
            state0.target_model = "/mnt/DataFlow/models/Qwen2.5-7B-Instruct"
            state0.target_model = ModelConfig(
                model_name_or_path=state0.target_model,
                tensor_parallel_size=1,
                max_tokens=2048,
            )
            log.info(f"[eval] Injected debug model config: {state0.target_model}")

        # 4) 执行 Workflow
        out = await graph.ainvoke(state0, config=config)

        if mode == "run":
            results_dir = project_root / "outputs"
            filename = f"eval_{thread_id}_{int(time.time())}.json"
            _save_state_json(out, results_dir, filename)

        return out


if __name__ == "__main__":
    # 使用与之前相同的 thread_id
    asyncio.run(run_eval(thread_id="demo_run_006", mode="debug"))
