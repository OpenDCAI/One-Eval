import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# One-Eval 核心组件
from one_eval.core.state import NodeState, BenchInfo
from one_eval.core.graph import GraphBuilder
from one_eval.logger import get_logger

# 导入节点
from one_eval.nodes.query_understand_node import QueryUnderstandNode
from one_eval.nodes.bench_search_node import BenchSearchNode
from one_eval.nodes.metric_recommend_node import MetricRecommendNode  # 新增节点

# 加载环境变量 (确保 OE_API_KEY 等存在)
load_dotenv()
log = get_logger("WorkflowIntegrationTest")

def build_full_eval_workflow(checkpointer=None):
    """
    构建完整的评估流水线：
    START -> QueryUnderstand -> BenchSearch -> MetricRecommend -> END
    """
    builder = GraphBuilder(
        state_model=NodeState,
        entry_point="QueryUnderstandNode",
    )

    # === 1. Query Understand (理解用户意图) ===
    node_query = QueryUnderstandNode()
    builder.add_node(name=node_query.name, func=node_query.run)

    # === 2. Bench Search (搜索/匹配数据集) ===
    node_search = BenchSearchNode()
    builder.add_node(name=node_search.name, func=node_search.run)

    # === 3. Metric Recommend (核心测试点：推荐指标) ===
    node_metric = MetricRecommendNode()
    builder.add_node(name=node_metric.name, func=node_metric.run)

    # === 定义边 (Edges) ===
    # 线性执行流
    builder.add_edge(START, node_query.name)
    builder.add_edge(node_query.name, node_search.name)
    builder.add_edge(node_search.name, node_metric.name)
    builder.add_edge(node_metric.name, END)

    # === 构建图 ===
    graph = builder.build(checkpointer=checkpointer)
    return graph

async def run_integration_scenario():
    """
    运行集成测试场景
    """
    print("\n" + "="*60)
    print(" One-Eval 完整 Workflow 集成测试")
    print("包含节点: QueryUnderstand -> BenchSearch -> MetricRecommend")
    print("="*60)

    # 1. 设置 Checkpointer (模拟生产环境持久化)
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2] # 假设在 tests/ 或类似层级，根据实际情况调整
    db_path = project_root / "checkpoints" / "integration_test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. 构造测试 Query
    # 这个 Query 旨在触发：Registry命中(GSM8K, HumanEval) 和 LLM分析(Medical Safety)
    user_query = (
        "我想做一个综合评估：\n"
        "1. 测一下数学能力\n"
        "2. 测一下代码能力\n"
        )

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_full_eval_workflow(checkpointer=checkpointer)
        
        # 使用新的 thread_id 确保状态隔离
        config = {"configurable": {"thread_id": "integration_test_run_001"}}
        
        # 初始状态
        initial_state = NodeState(user_query=user_query)

        print(f"\n[Input] 用户指令:\n{user_query}\n")
        print("-" * 60)

        # 3. 执行 Workflow
        # 注意：这里会真实调用 LLM 和 Search 工具 (如果配置了的话)
        # 如果 BenchSearchNode 无法联网，它可能会返回空或模拟数据，
        # 为了测试 MetricRecommend，我们需要确保 State 里有 BenchInfo。
        
        try:
            final_state = await graph.ainvoke(initial_state, config=config)
        except Exception as e:
            log.error(f"Workflow 执行出错: {e}")
            import traceback
            traceback.print_exc()
            return

        # 4. 结果验证与可视化
        print("\n" + "="*60)
        print(" Workflow 执行完成！结果分析：")
        print("="*60)

        # A. 检查 BenchSearch 的产出
        benches = final_state.get("benches")
        print(f"\n [BenchSearch 产出] 共找到 {len(benches)} 个数据集:")
        for b in benches:
            print(f"  - {b.bench_name} (Domain: {b.meta.get('domain', 'N/A')})")

        # B. 检查 MetricRecommend 的产出 (重点)
        metric_plan = final_state.get("metric_plan")
        print(f"\n [MetricRecommend 产出] 评估指标方案:")
        
        if not metric_plan:
            print("   警告: Metric Plan 为空！")
        
        for bench_name, metrics in metric_plan.items():
            print(f"\n  Dataset: [{bench_name}]")
            for m in metrics:
                # 格式化输出
                prio = m.get('priority', 'secondary')
                icon = "🌟" if prio == 'primary' else "  "
                args = f" | args={m.get('args')}" if m.get('args') else ""
                desc = f" | {m.get('desc')[:30]}..." if m.get('desc') else ""
                print(f"    {icon} {m['name']:<20} [{prio}]{args}{desc}")


if __name__ == "__main__":
    # 确保 asyncio 环境
    asyncio.run(run_integration_scenario())
