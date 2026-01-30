import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from one_eval.core.state import NodeState, BenchInfo
from one_eval.nodes.metric_recommend_node import MetricRecommendNode
from one_eval.logger import get_logger
from one_eval.core.metric_registry import load_metric_implementations

# 加载环境变量
load_dotenv()
log = get_logger("test_real_world_flow")

async def run_real_world_simulation():
    """
    模拟实战场景：
    1. 接收一个真实的评测结果文件 (JSONL)。
    2. 模拟从文件名/路径中提取 Bench 信息。
    3. 运行 MetricRecommendNode 生成评测指标方案。
    4. 验证方案是否符合 GSM8K 的预期 (Numerical Match)。
    """
    print("\n" + "="*60)
    print(" 实战模拟: 基于真实评测结果文件的 Metric 推荐流程")
    print("="*60)

    # 1. 模拟输入文件路径
    # 用户提供的真实路径
    raw_file_path = r"d:\CODE\Agent-Eval\One-Eval\cache\eval_results\gsm8k_1768402034_steps\step_step1.jsonl"
    file_path = Path(raw_file_path)
    
    if not file_path.exists():
        print(f" 文件不存在: {file_path}")
        print("   (将使用 Mock 数据继续演示逻辑)")
        # Mock logic if file is missing locally during dev
        bench_name_extracted = "gsm8k"
        sample_data = {"question": "Mock Question", "answer": "Mock #### 42"}
    else:
        print(f"✅ 找到文件: {file_path.name}")
        
        # 2. 从路径提取信息 (模拟 BenchSearch 或 Pipeline 上游的逻辑)
        # 假设文件夹名或文件名包含 bench_name
        # 路径结构: ...\gsm8k_1768402034_steps\step_step1.jsonl
        # 提取 'gsm8k'
        parent_dir = file_path.parent.name # gsm8k_1768402034_steps
        bench_name_extracted = parent_dir.split('_')[0]
        print(f"ℹ️  从路径提取数据集名称: '{bench_name_extracted}'")
        
        # 读取一行样本看看
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                sample_data = json.loads(first_line)
                print(f"ℹ️  样本数据预览: Q='{sample_data.get('question')[:30]}...' A='{sample_data.get('answer')[:30]}...'")

    # 3. 构造 NodeState
    # 假设这是从上一步传递下来的状态
    bench_info = BenchInfo(
        bench_name=bench_name_extracted,
        meta={
            "source_file": str(file_path),
            # 在真实流程中，这里可能还有 task_type, domain 等信息
            # 但我们测试的是 Registry 的命中能力，所以先留空，看能不能靠名字命中
        }
    )
    
    state = NodeState(
        user_query="对这个运行结果进行评测",
        benches=[bench_info]
    )

    # 4. 初始化节点并运行
    # 确保 metric 实现已加载
    load_metric_implementations()
    
    node = MetricRecommendNode()
    print("\n🚀 正在运行 MetricRecommendNode...")
    
    try:
        result_state = await node.run(state)
        
        # 5. 验证结果
        print("\n" + "-"*60)
        print("推荐结果分析:")
        print("-" * 60)
        
        plan = result_state.metric_plan.get(bench_name_extracted)
        
        if not plan:
            print(f"❌ 失败: 未为 '{bench_name_extracted}' 生成任何 Metric 方案。")
            return

        print(f"数据集: {bench_name_extracted}")
        print(f"推荐指标数量: {len(plan)}")
        
        # 检查是否包含关键指标
        metric_names = [m['name'] for m in plan]
        print(f"指标列表: {metric_names}")
        
        expected_metric = "numerical_match"  # GSM8K 应该用这个
        if any(expected_metric in name for name in metric_names):
             print(f"✅ 成功: 包含了预期指标 '{expected_metric}'")
        else:
             print(f"⚠️  警告: 未找到预期指标 '{expected_metric}'。可能使用了别名或配置不同。")
             
        # 打印详细配置
        for m in plan:
            print(f"  - {m['name']} (Priority: {m.get('priority')})")
            if 'args' in m:
                print(f"    Args: {m['args']}")

    except Exception as e:
        print(f"❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_real_world_simulation())