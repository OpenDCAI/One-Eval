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
    1. Data Loading (数据装载):
        - 解析 artifact_paths，定位本地预测文件(Pred)与真值文件(GT)。
        - 统一数据格式 (Standardize Format)，确保 MetricRunner 可读。

    2. Metric Execution (指标计算):
        - 依据 Step 3 生成的 Metric Plan。
        - 动态分发给 MetricRunner (支持 Math, Code, Text 等多模态计算)。

    3. Result Aggregation (结果聚合):
        - 汇总各 Benchmark 的多维指标得分。
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
            plan = metric_plan.get(bench_name, []) or []
            if not plan:
                continue

            bench_result = runner.run_bench(bench, plan)
            state.eval_results[bench_name] = bench_result

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