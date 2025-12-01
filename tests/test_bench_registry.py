import os
from one_eval.utils.bench_registry import BenchRegistry


def test_bench_registry():

    # ====== 路径 ======
    config_path = "one_eval/utils/bench_table/bench_config.json"
    assert os.path.exists(config_path), f"bench_config.json 不存在: {config_path}"

    # ====== 加载 registry ======
    registry = BenchRegistry(config_path)

    # ====== 测试 1：用户指定 benchmark ======
    specific = ["gsm8k", "MATH-500"]   # 混合大小写测试
    domain = []

    results = registry.search(
        specific_benches=specific,
        domain=domain
    )

    print("\n=== 测试 1：指定 benchmark ===")
    for r in results:
        print(r["bench_name"], "--", r["source"])

    # ====== 测试 2：domain 匹配 ======
    specific = []
    domain = ["math"]

    results = registry.search(
        specific_benches=specific,
        domain=domain
    )

    print("\n=== 测试 2：domain 匹配 ===")
    for r in results:
        print(r["bench_name"], "--", r["task_type"], "--", r["source"])

    # ====== 测试 3：指定 + 推荐 ======
    specific = ["gsm8k"]
    domain = ["math"]

    results = registry.search(
        specific_benches=specific,
        domain=domain
    )

    print("\n=== 测试 3：指定 + 自动推荐 ===")
    for r in results:
        print(r["bench_name"], "--", r["source"])


if __name__ == "__main__":
    test_bench_registry()
