from typing import Optional, Dict, Any
from one_eval.core.state import NodeState

def benches_manual_review(state: NodeState) -> Dict[str, Any]:
    """
    无条件插一个人工审查：
    不管自动推荐结果如何，最后都让人看一眼当前的 bench 列表。
    """
    benches = getattr(state, "benches", []) or []
    bench_info = getattr(state, "bench_info", {}) or {}

    return {
        "type": "manual_bench_review",
        "reason": "自动推荐已经完成，请人工确认当前 benchmark 列表是否合理。",
        "payload": {
            "benches": benches,
            "bench_info": bench_info,
        },
    }

def no_bench_validator(state: NodeState) -> Optional[Dict[str, Any]]:
    """
    条件校验示例：如果一个 bench 都没挑出来，就中断。
    """
    benches = getattr(state, "benches", []) or []
    if benches:
        return None  # 通过，不拦截

    return {
        "type": "no_bench_found",
        "reason": "当前没有找到任何 benchmark，请人工确认是继续还是回退修改需求。",
        "payload": {},
    }

def metric_plan_review(state: NodeState) -> Dict[str, Any]:
    """
    Review the recommended metric plan before calculation.
    """
    metric_plan = getattr(state, "metric_plan", {}) or {}
    return {
        "type": "manual_metric_review",
        "reason": "Metric recommendation completed. Please review the metric plan.",
        "payload": {
            "metric_plan": metric_plan
        },
    }
