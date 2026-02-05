import os
import asyncio
from dotenv import load_dotenv
from one_eval.core.state import NodeState, BenchInfo
from one_eval.nodes.metric_recommend_node import MetricRecommendNode
from one_eval.agents.metric_recommend_agent import MetricRecommendAgent
# Use the dispatcher directly
from one_eval.metrics.dispatcher import metric_dispatcher as metric_registry
# Use the new core for registration
from one_eval.core.metric_registry import register_metric, load_metric_implementations
from one_eval.logger import get_logger

"""
Test Suite for Refactored Metric Architecture (Decentralized & 3-Layer Funnel)

Scenarios:
1. Registry Match (Known Datasets) -> Returns configured templates.
2. Decentralized Registration -> Dynamic @register_metric updates the registry.
3. Unknown Datasets -> Returns None (triggers LLM fallback).
4. User Specified Metrics -> Agent prioritizes bench.meta["metrics"].
5. Format Validation -> Agent normalizes metric formats.
6. Full Node Execution -> Verifies end-to-end flow.
"""

log = get_logger("test_refactored_arch")
load_dotenv()

def test_metric_registry_lookup():
    """Test Scenario 1: Registry Lookup for Known Datasets"""
    print("\n" + "="*60)
    print("Test 1: Registry Lookup (Known Datasets)")
    print("="*60)

    # Ensure metrics are loaded
    load_metric_implementations()
    # Force rebuild templates in case of prior tests
    if hasattr(metric_registry, "_build_templates"):
        metric_registry._build_templates()

    test_cases = [
        "gsm8k",           # Exact match in config
        "mmlu",            # Exact match
        "openai/gsm8k",    # Should match gsm8k if config supports keys
    ]

    for dataset_name in test_cases:
        metrics = metric_registry.get_metrics(dataset_name)
        print(f"\nDataset: '{dataset_name}'")
        if metrics:
            print(f"   HIT (Count: {len(metrics)}):")
            for m in metrics:
                print(f"    - {m.get('name')} ({m.get('priority')})")
        else:
            print(f"   MISS (None) - Check if '{dataset_name}' is in config.py")

def test_decentralized_registration():
    """Test Scenario 2: Decentralized Metric Registration"""
    print("\n" + "="*60)
    print("Test 2: Decentralized Registration (Dynamic Update)")
    print("="*60)

    # 1. Define a dynamic metric with a unique group
    unique_group = "test_dynamic_group_v1"
    
    @register_metric(
        name="dynamic_metric_v1",
        desc="Dynamic metric for testing",
        groups={unique_group: "primary"}
    )
    def compute_dynamic_v1(preds, refs, **kwargs):
        return {"score": 100}

    # 2. Refresh the registry (Simulate startup or manual refresh)
    # The new Dispatcher uses _build_templates
    if hasattr(metric_registry, "_build_templates"):
        metric_registry._build_templates()
        print("   Registry templates rebuilt.")
    
    # 3. Register a temporary dataset mapping for this test
    # The dispatcher should allow registering dataset->template mapping
    if hasattr(metric_registry, "register_dataset"):
        metric_registry.register_dataset("test_dataset_dynamic", unique_group)
    
    # 4. Verify lookup
    metrics = metric_registry.get_metrics("test_dataset_dynamic")
    if metrics and any(m["name"] == "dynamic_metric_v1" for m in metrics):
        print("   SUCCESS: Dynamic metric found via group lookup.")
    else:
        print(f"   FAILURE: Dynamic metric not found. Got: {metrics}")

async def test_agent_logic_unknown_dataset():
    """Test Scenario 3: Unknown Datasets (Agent Logic)"""
    print("\n" + "="*60)
    print("Test 3: Unknown Datasets (Agent Fallback)")
    print("="*60)

    agent = MetricRecommendAgent(tool_manager=None)
    
    # Unknown dataset with no meta hints
    bench = BenchInfo(bench_name="completely_unknown_ds_999", meta={})
    
    # Should return None from registry check
    res = agent._check_registry(bench)
    print(f"\nChecking '{bench.bench_name}':")
    if res is None:
        print("   SUCCESS: Returns None (Will trigger LLM).")
    else:
        print(f"   FAILURE: Returned {res} (Expected None).")

async def test_user_specified_metrics():
    """Test Scenario 4: User Specified Metrics Override"""
    print("\n" + "="*60)
    print("Test 4: User Specified Metrics Override")
    print("="*60)

    state = NodeState(
        user_query="Run custom eval",
        benches=[
            BenchInfo(
                bench_name="custom_bench",
                meta={
                    "metrics": [
                        {"name": "user_metric_A", "priority": "primary"},
                        {"name": "user_metric_B", "k": 5} # Short format
                    ]
                }
            )
        ]
    )
    
    agent = MetricRecommendAgent(tool_manager=None)
    result_state = await agent.run(state)
    
    plan = result_state.metric_plan.get("custom_bench")
    if plan:
        print("   SUCCESS: Plan generated from user metrics.")
        names = [m["name"] for m in plan]
        print(f"   Metrics: {names}")
        if "user_metric_A" in names and "user_metric_B" in names:
             print("   All user metrics present.")
        else:
             print("   MISSING user metrics.")
    else:
        print("   FAILURE: No plan generated.")

def test_format_validation():
    """Test Scenario 5: Metric Format Validation"""
    print("\n" + "="*60)
    print("Test 5: Format Validation")
    print("="*60)

    agent = MetricRecommendAgent(tool_manager=None)
    raw_metrics = [
        {"name": "valid_one", "priority": "primary"},
        {"name": "missing_prio"}, # Should default to secondary
        {"name": "flattened_args", "k": 10}, # Should move k to args
    ]
    
    validated = agent._validate_metrics(raw_metrics)
    print("Validated Metrics:")
    for m in validated:
        print(f"  - {m}")
        
    # Checks
    if validated[1]["priority"] == "secondary":
        print("   SUCCESS: Default priority applied.")
    if validated[2].get("args", {}).get("k") == 10:
        print("   SUCCESS: Flattened args moved to 'args' dict.")

async def test_full_node_execution():
    """Test Scenario 6: Full Node Execution"""
    print("\n" + "="*60)
    print("Test 6: Full Node Execution (End-to-End)")
    print("="*60)

    # Use a mix of known (registry) and unknown (llm)
    state = NodeState(
        user_query="Eval mixed tasks",
        benches=[
            BenchInfo(bench_name="gsm8k", meta={}), # Known
            BenchInfo(bench_name="unknown_qa", meta={"task_type": "qa"}) # Unknown
        ]
    )

    node = MetricRecommendNode()
    
    # Warn if no API key
    if not (os.getenv("OE_API_BASE") and os.getenv("OE_API_KEY")):
        print("   WARNING: No LLM API Key found. LLM part might fail or mock.")

    try:
        result = await node.run(state)
        print("\nMetric Plan:")
        for bench, metrics in result.metric_plan.items():
            print(f"   Bench: {bench} -> {len(metrics)} metrics")
            for m in metrics:
                 print(f"     - {m['name']}")
    except Exception as e:
        print(f"   Execution Failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    print("Starting Refactored Architecture Tests...")
    
    test_metric_registry_lookup()
    test_decentralized_registration()
    await test_agent_logic_unknown_dataset()
    await test_user_specified_metrics()
    test_format_validation()
    await test_full_node_execution()
    
    print("\nAll Tests Completed.")

if __name__ == "__main__":
    asyncio.run(main())