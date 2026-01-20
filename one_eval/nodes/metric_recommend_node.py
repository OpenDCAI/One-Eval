from __future__ import annotations
from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState
from one_eval.agents.metric_recommend_agent import MetricRecommendAgent
from one_eval.toolkits.tool_manager import get_tool_manager
from one_eval.logger import get_logger

log = get_logger("MetricRecommendNode")


class MetricRecommendNode(BaseNode):
    """
    Step 3 Node:
    根据 Benchmark 信息推荐评估指标 (Metric)。
    """
    
    def __init__(self):
        self.name = "MetricRecommendNode"

    async def run(self, state: NodeState) -> NodeState:
        log.info(f"[{self.name}] 节点开始执行")

        # 获取全局 ToolManager (如果后续 Metric 计算需要用到特定工具信息，可传入)
        tm = get_tool_manager()

        # 创建 Agent
        # 这里可以根据 state.task_domain 动态调整 model_name，或者使用默认
        agent = MetricRecommendAgent(
            tool_manager=tm,
            model_name="gpt-4o", # 推荐 Metric 需要较强的逻辑推理能力
            temperature=0.1      # 降低随机性，保证 Metric 的准确性
        )

        # 执行 Agent 逻辑 (包含双轨制处理)
        new_state = await agent.run(state)

        benches = getattr(new_state, "benches")
        log.info(f"[{self.name}] 正在检查 benches 数据...")
        for bench in benches:
            bench_name = bench.bench_name
            # log.info(f"[{self.name}] 检查 bench: {bench_name}")
            if bench_name in new_state.metric_plan:
                metrics = new_state.metric_plan[bench_name]
                # log.info(f"[{self.name}] 发现 {bench_name} 的 metrics: {metrics}")
                # 格式化输出: "accuracy (primary), bleu (secondary)"
                metric_summary = ", ".join([
                    f"{m.get('name')} ({m.get('priority', 'secondary')})" 
                    for m in metrics
                ])
                log.info(f"[{bench_name}] Recommended Metrics: [{metric_summary}]")
            else:
                log.warning(f"[{bench_name}] No metrics recommended!")

        log.info(f"[{self.name}] 执行结束，Metric Plan 更新完毕。")
        
        return new_state
