from __future__ import annotations

from typing import Any, Dict, List

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState, BenchInfo
from one_eval.logger import get_logger
from one_eval.metrics.runner import MetricRunner

log = get_logger("ScoreCalcAgent")


class ScoreCalcAgent(CustomAgent):
    """
    Step 3 Agent: Score计算
    """
    
    @property
    def role_name(self) -> str:
        return "ScoreCalcAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return ""

    @property
    def task_prompt_template_name(self) -> str:
        return ""

    async def run(self, state: NodeState) -> NodeState:
        benches: List[BenchInfo] = getattr(state, "benches", []) or []
        metric_plan: Dict[str, Any] = getattr(state, "metric_plan", {}) or {}

        if not benches:
            log.warning("state.benches 为空，跳过 score 计算")
            return state

        if not metric_plan:
            log.warning("state.metric_plan 为空，跳过 score 计算")
            return state

        if not getattr(state, "eval_results", None):
            state.eval_results = {}

        runner = MetricRunner()

        computed: List[str] = []
        failed: List[Dict[str, Any]] = []

        for bench in benches:
            bench_name = bench.bench_name
            
            # === 自动关联 DataFlowEvalNode 的结果 ===
            if bench.meta and bench.meta.get("eval_detail_path"):
                detail_path = bench.meta["eval_detail_path"]
                if "artifact_paths" not in bench.meta:
                    bench.meta["artifact_paths"] = {}
                # 将 detail_path (step2 result) 注册为 records_path
                # MetricRunner 会优先读取 records_path
                bench.meta["artifact_paths"]["records_path"] = detail_path
                log.info(f"[{bench_name}] Linked eval_detail_path to artifact_paths['records_path']: {detail_path}")

            plan = metric_plan.get(bench_name, []) or []
            if not plan:
                continue

            bench_result = runner.run_bench(bench, plan)
            state.eval_results[bench_name] = bench_result

            metrics = bench_result.get("metrics", {})
            num_samples = bench_result.get("num_samples", 0)
            summary = {
                "bench": bench_name,
                "num_samples": num_samples,
                "metrics": {mname: (mres.get("score")) for mname, mres in metrics.items()},
            }
            log.info(f"[{bench_name}] Summary: {summary}")

            if isinstance(bench_result, dict) and bench_result.get("error"):
                failed.append({"bench": bench_name, "error": bench_result.get("error")})
            else:
                computed.append(bench_name)

        if not getattr(state, "result", None):
            state.result = {}

        state.result[self.role_name] = {
            "computed": computed,
            "failed": failed,
        }

        return state