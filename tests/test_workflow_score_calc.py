import asyncio
import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder
from one_eval.logger import get_logger
from one_eval.nodes.query_understand_node import QueryUnderstandNode
from one_eval.nodes.bench_search_node import BenchSearchNode
from one_eval.nodes.metric_recommend_node import MetricRecommendNode
from one_eval.nodes.score_calc_node import ScoreCalcNode

load_dotenv()
log = get_logger("WorkflowIntegrationTest")

# ==========================================
# 1. 路径配置
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent
TEST_DATA_ROOT = CURRENT_DIR / "mock_data_storage"

LOCAL_DATA_MAP = {
    "math-500": {
        "pred": TEST_DATA_ROOT / "math500_pred.jsonl",
        "gt":   TEST_DATA_ROOT / "math500_gt.jsonl"
    },
    "humaneval": {
        "pred": TEST_DATA_ROOT / "humaneval_pred.jsonl",
        "gt":   TEST_DATA_ROOT / "humaneval_gt.jsonl"
    }
}

# ==========================================
# 2. 数据生成 (已填入你提供的完整数据)
# ==========================================
def generate_and_verify_data():
    print(f"\n[Setup] 正在初始化数据目录: {TEST_DATA_ROOT}")
    
    if TEST_DATA_ROOT.exists():
        shutil.rmtree(TEST_DATA_ROOT)
    TEST_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # --- A. Math-500 Ground Truth (你提供的数据) ---
    math_gt = [
        {"sample_id": "q1", "target": "10"},
        {"sample_id": "q2", "target": "5"},
        {"sample_id": "q3_missing_pred", "target": "8"},     
        {"sample_id": "101", "target": "101"},
        {"sample_id": "102", "target": "888"},
        {"sample_id": "test_float", "target": "3.5"},
        {"sample_id": "test_text", "target": "42"},
        {"sample_id": "test_latex", "target": "\\frac{1}{2}"}, # 注意 Python 字符串转义
        {"sample_id": "test_wrong", "target": "100"},
        {"sample_id": "test_missing_2", "target": "9999"}
    ]

    # --- B. Math-500 Predict (你提供的数据) ---
    math_pred = [
        {"sample_id": "q1", "predict": "10"},                 
        {"sample_id": "q2", "predict": "Answer is 5."},       
        {"sample_id": "q_extra_pred", "predict": "999"},      
        {"sample_id": "101", "predict": "100"},               
        {"sample_id": "102", "predict": "888"},               
        {"sample_id": "test_float", "predict": "3.50"},       
        {"sample_id": "test_text", "predict": "The answer is clearly 42"}, 
        {"sample_id": "test_latex", "predict": "Result: \\frac{1}{2}"},    
        {"sample_id": "test_wrong", "predict": "0"}
    ]
    
    # --- C. Humaneval Ground Truth (你提供的数据) ---
    code_gt = [
        {
            "sample_id": "HumanEval/0", 
            "target": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                if abs(elem - elem2) < threshold:\n                    return True\n    return False"
        },
        {
            "sample_id": "HumanEval/1", 
            "target": "def separate_paren_groups(paren_string: str) -> List[str]:\n    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string.clear()\n    return result"
        },
        {
            "sample_id": "HumanEval/2", 
            "target": "def truncate_number(number: float) -> float:\n    return number % 1.0"
        },
        {
            "sample_id": "HumanEval/3", 
            "target": "def below_zero(operations: List[int]) -> bool:\n    balance = 0\n    for op in operations:\n        balance += op\n        if balance < 0:\n            return True\n    return False"
        },
        {
            "sample_id": "HumanEval/4", 
            "target": "def mean_absolute_deviation(numbers: List[float]) -> float:\n    mean = sum(numbers) / len(numbers)\n    return sum(abs(x - mean) for x in numbers) / len(numbers)"
        }
    ]

    # --- D. Humaneval Predict (你提供的数据) ---
    code_pred = [
        {
            "sample_id": "HumanEval/0", 
            "predict": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                if abs(elem - elem2) < threshold:\n                    return True\n    return False"
        },
        {
            "sample_id": "HumanEval/1", 
            "predict": "def separate_paren_groups(paren_string: str) -> List[str]:\n    return []  # I don't know how to solve this"
        },
        {
            "sample_id": "HumanEval/2", 
            "predict": "def truncate_number(number: float) -> float:\n    # This function truncates the decimal part\n    return number % 1.0"
        },
        {
            "sample_id": "HumanEval/3", 
            "predict": "def below_zero(operations: List[int]) -> bool:\n    return true; // java style syntax error"
        },
        {
            "sample_id": "HumanEval/999", 
            "predict": "print('This task does not exist in GT')"
        }
    ]

    # --- 写入函数 (严格 JSONL 格式) ---
    def write_safe_jsonl(path, data_list):
        with open(path, "w", encoding="utf-8") as f:
            for item in data_list:
                # json.dumps 会自动处理引号、换行符转义等
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  -> 已写入: {path.name} ({len(data_list)} 行)")

    write_safe_jsonl(LOCAL_DATA_MAP["math-500"]["gt"], math_gt)
    write_safe_jsonl(LOCAL_DATA_MAP["math-500"]["pred"], math_pred)
    write_safe_jsonl(LOCAL_DATA_MAP["humaneval"]["gt"], code_gt)
    write_safe_jsonl(LOCAL_DATA_MAP["humaneval"]["pred"], code_pred)

    print(" 数据生成与格式化完成。")
def inject_local_data(state: NodeState) -> NodeState:
    """
    这是一个辅助函数，用于在内存中修改 state，注入文件路径
    """
    benches = getattr(state, "benches", []) or []
    if not benches:
        return state

    print(f"\n[DataInjection] 正在为 {len(benches)} 个数据集注入本地数据路径...")

    for bench in benches:
        bench_name_lower = bench.bench_name.lower()
        
        # 查找匹配的配置
        matched_config = None
        for key, config in LOCAL_DATA_MAP.items():
            if key in bench_name_lower:
                matched_config = config
                break
        
        # 如果没匹配到，可以用 default (可选)
        if not matched_config and "default" in LOCAL_DATA_MAP:
            matched_config = LOCAL_DATA_MAP["default"]

        if matched_config:
            # 初始化 meta
            if not bench.meta:
                bench.meta = {}
            
            # 构造 artifact_paths
            artifact_paths = {}
            
            # 情况 A: 双文件模式 (pred + gt)
            if "pred" in matched_config and "gt" in matched_config:
                artifact_paths["predict_file"] = os.path.abspath(matched_config["pred"])
                artifact_paths["ground_truth_file"] = os.path.abspath(matched_config["gt"])
                print(f"  -> [{bench.bench_name}] 注入 Split 模式: {artifact_paths['predict_file']}")

            # 情况 B: 单文件模式 (records)
            elif "records" in matched_config:
                artifact_paths["records_path"] = os.path.abspath(matched_config["records"])
                print(f"  -> [{bench.bench_name}] 注入 Records 模式: {artifact_paths['records_path']}")
            
            # 写入 State
            bench.meta["artifact_paths"] = artifact_paths
        else:
            print(f"  -> [{bench.bench_name}] 未找到本地映射，将跳过计算...")

    return state
# =================================================================


def build_full_eval_workflow(checkpointer=None):
    """
    构建完整的评估流水线：
    START -> QueryUnderstand -> BenchSearch -> MetricRecommend -> ScoreCalc -> END
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

    # 包装一个带注入功能的搜索节点函数
    async def search_with_injection(state: NodeState) -> NodeState:
        # A. 先执行原始的搜索逻辑 (去 HuggingFace 找名字)
        state = await node_search.run(state)
        
        # B. 执行我们的注入逻辑 (修改 meta 路径)
        state = inject_local_data(state)
        
        return state
    
    builder.add_node(name=node_search.name, func=search_with_injection)

    # === 3. Metric Recommend (推荐指标) ===
    node_metric = MetricRecommendNode()
    builder.add_node(name=node_metric.name, func=node_metric.run)

    # === 4. Score Calc (计算分数) ===
    node_score = ScoreCalcNode()
    builder.add_node(name=node_score.name, func=node_score.run)

    # === 定义边 (Edges) ===
    # 线性执行流
    builder.add_edge(START, node_query.name)
    builder.add_edge(node_query.name, node_search.name)
    builder.add_edge(node_search.name, node_metric.name)
    builder.add_edge(node_metric.name, node_score.name)
    builder.add_edge(node_score.name, END)

    # === 构建图 ===
    graph = builder.build(checkpointer=checkpointer)
    return graph

async def run_integration_scenario():
    """
    运行集成测试场景
    """
    print("\n" + "="*60)
    print(" One-Eval 完整 Workflow 集成测试")
    print("包含节点: QueryUnderstand -> BenchSearch -> MetricRecommend -> ScoreCalc")
    print("="*60)

    generate_and_verify_data()

    # 1. 设置 Checkpointer (模拟生产环境持久化)
    current_file_path = Path(__file__).resolve()
    # 假设在 tests/ 或类似层级，根据实际情况调整 project_root
    project_root = current_file_path.parents[2] 
    db_path = project_root / "checkpoints" / "integration_test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. 构造测试 Query
    user_query = (
        "我想做一个综合评估：\n"
        "1. 测一下数学能力\n"
        "2. 测一下代码能力\n"
        )

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_full_eval_workflow(checkpointer=checkpointer)
        
        # 使用新的 thread_id 确保状态隔离，每次运行可视情况修改 id
        config = {"configurable": {"thread_id": "integration_test_run_001"}}
        
        # 初始状态
        initial_state = NodeState(user_query=user_query)

        print(f"\n[Input] 用户指令:\n{user_query}\n")
        print("-" * 60)

        # 3. 执行 Workflow
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
        print(f"\n [BenchSearch 产出] 共找到 {len(benches) if benches else 0} 个数据集:")
        if benches:
            for b in benches:
                print(f"  - {b.bench_name} (Domain: {b.meta.get('domain', 'N/A')})")

        # B. 检查 MetricRecommend 的产出
        metric_plan = final_state.get("metric_plan")
        print(f"\n [MetricRecommend 产出] 评估指标方案:")
        
        if not metric_plan:
            print("   警告: Metric Plan 为空！")
        else:
            for bench_name, metrics in metric_plan.items():
                print(f"\n  Dataset: [{bench_name}]")
                for m in metrics:
                    prio = m.get('priority', 'secondary')
                    tag = "PRIMARY" if prio == 'primary' else prio.upper()
                    args = f" | args={m.get('args')}" if m.get('args') else ""
                    desc = f" | {m.get('desc')[:30]}..." if m.get('desc') else ""
                    print(f"    - {m['name']:<20} [{tag}]{args}{desc}")

        # C. 检查 ScoreCalc 的产出
        eval_results = final_state.get("eval_results")
        print(f"\n [ScoreCalc 产出] 评估结果:")
        if not eval_results:
            print("   提示: Eval Results 为空 (可能是因为没有真实数据下载或运行，符合预期)")
        else:
            for bench_name, res in eval_results.items():
                print(f"\n  Dataset: [{bench_name}]")
                if isinstance(res, dict) and "metrics" in res:
                    for m_name, m_res in res["metrics"].items():
                        score = m_res.get("score", "N/A")
                        err = m_res.get("error", "")
                        print(f"    - {m_name:<20}: score={score} {f'(Error: {err})' if err else ''}")
                else:
                    print(f"    - Result: {res}")

if __name__ == "__main__":
    asyncio.run(run_integration_scenario())
