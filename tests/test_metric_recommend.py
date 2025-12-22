import os
import asyncio
from dotenv import load_dotenv
from one_eval.core.state import NodeState, BenchInfo
from one_eval.nodes.metric_recommend_node import MetricRecommendNode
from one_eval.agents.metric_recommend_agent import MetricRecommendAgent
from one_eval.utils.metric_registry import metric_registry
from one_eval.logger import get_logger

"""
测试 MetricRecommendAgent 和 MetricRecommendNode (Updated)

测试场景：
1. 注册表匹配（已知数据集如 gsm8k, mmlu） -> 期望返回配置
2. 注册表未匹配（未知数据集） -> 期望返回 None
3. 用户指定 metrics（bench.meta["metrics"]） -> 期望优先使用
4. 格式验证和规范化 -> 期望正确处理 args/k 参数
5. 完整流程（Node Execution） -> 测试 Registry -> LLM -> Fallback 的流转
"""

log = get_logger("test_metric_recommend")
load_dotenv()


def test_metric_registry():
    """测试 metric_registry 的基础功能"""
    print("\n" + "="*60)
    print("测试 1: MetricRegistry 基础功能")
    print("="*60)
    
    # 测试用例
    test_cases = [
        "gsm8k",           # 精确匹配
        "mmlu",            # 精确匹配
        "openai/gsm8k",    # 模糊匹配 (包含)
        "hendrycks_math",  # 模糊匹配
        "unknown_dataset", # 未知数据集 -> 应该返回 None
        "mathematicas"     # 干扰项 -> 应该返回 None (避免误匹配 math)
    ]
    
    for dataset_name in test_cases:
        metrics = metric_registry.get_metrics(dataset_name)
        print(f"\n数据集: '{dataset_name}'")
        if metrics:
            print(f"   命中配置 (共 {len(metrics)} 个):")
            for m in metrics:
                print(f"    - {m.get('name')} ({m.get('priority')})")
        else:
            print(f"   未找到配置 (返回 None) - 符合预期")


async def test_registry_match_logic():
    """测试场景2: Agent 的注册表查询逻辑"""
    print("\n" + "="*60)
    print("测试 2: Agent 内部的注册表查询逻辑")
    print("="*60)
    
    # 创建 Agent (Mock)
    agent = MetricRecommendAgent(tool_manager=None)
    
    # 1. 测试已知数据集
    bench_known = "gsm8k"
    res_known = agent._check_registry(bench_known)
    print(f"\n查询已知 Benchmark '{bench_known}':")
    if res_known:
        print(f"   成功获取: {[m['name'] for m in res_known]}")
    else:
        print(f"   失败 (不应发生)")

    # 2. 测试未知数据集
    bench_unknown = "super_custom_dataset_v999"
    res_unknown = agent._check_registry(bench_unknown)
    print(f"\n查询未知 Benchmark '{bench_unknown}':")
    if res_unknown is None:
        print(f"   返回 None (正确，将转交给 LLM 处理)")
    else:
        print(f"   返回了数据: {res_unknown} (预期应为 None)")


async def test_user_specified_metrics():
    """测试场景3: 用户显式指定 metrics"""
    print("\n" + "="*60)
    print("测试 3: 用户显式指定 Metrics")
    print("="*60)
    
    state = NodeState(
        user_query="使用自定义指标评估",
        benches=[
            BenchInfo(
                bench_name="custom_bench",
                meta={
                    "metrics": [
                        {"name": "custom_metric_1", "priority": "primary", "desc": "自定义指标1"},
                        # 测试参数简写
                        {"name": "pass_at_k", "k": 10, "priority": "secondary"} 
                    ],
                    "domain": "custom"
                }
            )
        ]
    )
    
    agent = MetricRecommendAgent(tool_manager=None)
    
    # 执行推荐
    result_state = await agent.run(state)
    
    print(f"\nBenchmark: custom_bench")
    if "custom_bench" in result_state.metric_plan:
        metrics = result_state.metric_plan["custom_bench"]
        print(f"  使用用户指定的 Metrics ({len(metrics)} 个):")
        for m in metrics:
            args_str = f", args={m.get('args')}" if m.get('args') else ""
            print(f"    - {m['name']} (priority: {m['priority']}{args_str})")
    else:
        print("    未找到 metric_plan")


def test_metric_format_validation():
    """测试场景4: 指标格式验证和规范化"""
    print("\n" + "="*60)
    print("测试 4: 指标格式验证和规范化")
    print("="*60)
    
    agent = MetricRecommendAgent(tool_manager=None)
    
    # 测试各种格式的指标
    test_metrics = [
        # 1. 标准格式
        {"name": "accuracy", "priority": "primary", "desc": "准确率"},
        # 2. 缺少 priority（应该默认 secondary）
        {"name": "f1_score"},
        # 3. 扁平化参数 k -> args={'k': 1}
        {"name": "pass_at_k", "k": 1, "priority": "primary"},
        # 4. 扁平化参数 params -> args
        {"name": "bleu", "params": {"smooth": True}},
        # 5. 无效格式（缺少 name）-> 应被过滤
        {"priority": "primary"},
        # 6. 无效 priority -> 应修正为 secondary
        {"name": "test_metric", "priority": "super_high"},
    ]
    
    validated = agent._validate_metrics(test_metrics)
    
    print(f"\n输入指标数: {len(test_metrics)}")
    print(f"验证通过数: {len(validated)}")
    print(f"过滤掉数: {len(test_metrics) - len(validated)}")
    
    print("\n验证后的指标详情:")
    for m in validated:
        print(f"  - Name: {m['name']}")
        print(f"    Priority: {m['priority']}")
        if 'args' in m:
            print(f"    Args: {m['args']}")
        print("-" * 20)


async def test_metric_recommend_node_full():
    """测试场景5: 完整的 MetricRecommendNode 执行（包含 LLM 调用）"""
    print("\n" + "="*60)
    print("测试 5: MetricRecommendNode 完整执行")
    print("="*60)
    
    # 准备测试数据：混合已知和未知数据集
    state = NodeState(
        user_query="评估模型在多个任务上的表现",
        benches=[
            # 1. 已知数据集 -> 应该命中注册表
            BenchInfo(
                bench_name="gsm8k",
                meta={"domain": "math", "task_type": "numerical"}
            ),
            # 2. 未知数据集 -> 应该调用 LLM
            BenchInfo(
                bench_name="unknown_qa_dataset_v1",
                meta={
                    "domain": "qa",
                    "task_type": "question_answering",
                    "description": "关于自然科学的问答数据集",
                    "examples": [{"question": "天空为什么是蓝的？", "answer": "因为瑞利散射..."}]
                }
            ),
            # 3. 既未知又无描述 -> LLM 可能失败或推荐默认，最终触发兜底
            BenchInfo(
                bench_name="completely_random_name",
                meta={}
            )
        ]
    )
    
    node = MetricRecommendNode()
    
    print("\n执行 MetricRecommendNode...")
    
    # 检查 API Key，如果没有则提示
    has_api = os.getenv("OE_API_BASE") and os.getenv("OE_API_KEY")
    if not has_api:
        print("  警告: 未检测到 LLM API 配置。LLM 部分可能会失败或报错，但我们将测试兜底逻辑。")
    
    try:
        result_state = await node.run(state)
        
        print("\n" + "-"*60)
        print("最终推荐结果 (Metric Plan):")
        print("-"*60)
        
        for bench_name, metrics in result_state.metric_plan.items():
            print(f"\n Benchmark: {bench_name}")
            print(f"   推荐指标数: {len(metrics)}")
            for m in metrics:
                args_str = f", args: {m.get('args')}" if m.get('args') else ""
                print(f"   - {m['name']} ({m['priority']}){args_str}")
        
        # 打印统计信息
        if "MetricRecommendAgent" in result_state.result:
            stats = result_state.result["MetricRecommendAgent"]
            print("\n" + "-"*60)
            print("执行统计:")
            print(f"  Registry 命中: {stats.get('registry_hits', [])}")
            print(f"  LLM 分析列表: {stats.get('llm_analyzed', [])}")
        
    except Exception as e:
        print(f"\n 执行失败: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("MetricRecommend 功能测试套件 (Updated)")
    print("="*60)
    
    # 1. 基础 Registry 测试
    test_metric_registry()
    
    # 2. Agent 逻辑测试
    await test_registry_match_logic()
    
    # 3. 用户覆盖测试
    await test_user_specified_metrics()
    
    # 4. 格式验证测试
    test_metric_format_validation()

    # 5. 完整流程测试
    await test_metric_recommend_node_full()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
