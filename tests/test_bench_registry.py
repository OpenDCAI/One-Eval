import os
import json
from one_eval.utils.bench_registry import BenchRegistry


def test_remove_bench_persists_and_returns_removed_entry(tmp_path):
    config_path = tmp_path / "bench_gallery.json"
    local = {
        "bench_name": "local_demo",
        "meta": {"source": "user_upload"},
    }
    remote = {
        "bench_name": "remote_demo",
        "meta": {"source": "bench_item_list"},
    }
    config_path.write_text(json.dumps({"benches": [local, remote]}), encoding="utf-8")

    registry = BenchRegistry(str(config_path))
    removed = registry.remove_bench("LOCAL_DEMO", str(config_path))

    assert removed is not None
    assert removed["bench_name"] == "local_demo"
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert [item["bench_name"] for item in persisted["benches"]] == ["remote_demo"]
    assert registry.get_bench_by_name("local_demo") is None


def test_bench_registry_deduplicates_names_and_prefers_last_entry(tmp_path):
    config_path = tmp_path / "bench_gallery.json"
    config_path.write_text(
        json.dumps(
            {
                "benches": [
                    {"bench_name": "bfcl", "meta": {"source": "old"}},
                    {"bench_name": "BFCL", "meta": {"source": "new"}},
                ]
            }
        ),
        encoding="utf-8",
    )

    registry = BenchRegistry(str(config_path))

    assert len(registry.get_all_benches()) == 1
    assert registry.get_bench_by_name("bfcl")["meta"]["source"] == "new"


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
