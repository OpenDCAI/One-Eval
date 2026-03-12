"""
Benchmark 可用性批量验证脚本

流程：跳过 Phase 1 (NL2Bench)，直接从 bench_gallery.json 批量构造 BenchInfo，
依次跑 Phase 2~3：
  DatasetStructureNode → BenchConfigRecommendNode → DownloadNode（限制条数）
  → DatasetKeysNode → BenchTaskInferNode

最终输出每个 benchmark 的通过/失败报告。
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import asyncio
import json
import time
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from one_eval.core.state import NodeState, BenchInfo
from one_eval.nodes.dataset_structure_node import DatasetStructureNode
from one_eval.nodes.bench_config_recommend_node import BenchConfigRecommendNode
from one_eval.nodes.download_node import DownloadNode
from one_eval.nodes.dataset_keys_node import DatasetKeysNode
from one_eval.nodes.bench_task_infer_node import BenchTaskInferNode
from one_eval.logger import get_logger

log = get_logger("BenchVerify")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH_GALLERY_PATH = PROJECT_ROOT / "one_eval/utils/bench_table/bench_gallery.json"

# 每个 benchmark 最多下载的数据条数（减少等待时间）
MAX_ROWS_PER_BENCH = 20


def load_all_benches() -> list[BenchInfo]:
    """从 bench_gallery.json 加载所有 benchmark，构造 BenchInfo 列表"""
    with open(BENCH_GALLERY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    benches = []
    for b in data["benches"]:
        info = BenchInfo(
            bench_name=b["bench_name"],
            bench_table_exist=b.get("bench_table_exist", False),
            bench_source_url=b.get("bench_source_url"),
            bench_dataflow_eval_type=b.get("bench_dataflow_eval_type"),
            bench_keys=b.get("bench_keys") or [],
            meta=b.get("meta") or {},
        )
        benches.append(info)
    return benches


def _patch_download_limited(max_rows: int):
    """
    返回一个 patch 上下文，让 HFDownloadTool.download_and_convert
    只写入前 max_rows 条数据，避免全量下载。
    """
    from one_eval.toolkits import hf_download_tool as _mod

    original_fn = _mod.HFDownloadTool.download_and_convert

    def limited_download(self, repo_id, config_name, split, output_path, revision=None):
        from datasets import load_dataset
        import json as _json
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"[LimitedDownload] {repo_id} config={config_name} split={split} max_rows={max_rows}")
        try:
            kwargs = {
                "split": split,
                "revision": revision,
                "token": self.hf_token,
                "streaming": True,  # 流式加载，不下载整个数据集
            }
            try:
                ds = load_dataset(repo_id, config_name, trust_remote_code=True, **kwargs)
            except Exception:
                ds = load_dataset(repo_id, config_name, **kwargs)
        except Exception as e:
            return {"ok": False, "error": f"load_dataset 失败: {e}"}

        try:
            count = 0
            with out_path.open("w", encoding="utf-8") as f:
                for item in ds:
                    if count >= max_rows:
                        break
                    clean_item = {}
                    for k, v in item.items():
                        try:
                            _json.dumps({k: v})
                            clean_item[k] = v
                        except (TypeError, OverflowError):
                            clean_item[k] = str(v)
                    f.write(_json.dumps(clean_item, ensure_ascii=False) + "\n")
                    count += 1
            log.info(f"[LimitedDownload] 写入 {count} 条 -> {out_path}")
            cols = ds.column_names if hasattr(ds, "column_names") and ds.column_names else []
            return {
                "ok": True,
                "output_path": str(out_path),
                "num_rows": count,
                "columns": list(cols),
            }
        except Exception as e:
            return {"ok": False, "error": f"写入失败: {e}"}

    return patch.object(_mod.HFDownloadTool, "download_and_convert", limited_download)


async def verify_benches(
    bench_names: Optional[list[str]] = None,
    max_rows: int = MAX_ROWS_PER_BENCH,
    skip_structure: bool = False,
    skip_config_recommend: bool = False,
    output_file: Optional[str] = None,
):
    """
    对 gallery 中的 benchmark 执行 Phase 2~3 验证。

    Args:
        bench_names: 指定要验证的 benchmark 名称列表，None 表示全部
        max_rows: 每个 benchmark 最多下载的数据条数
        skip_structure: 若 meta.structure 已存在则跳过（DatasetStructureNode 本身会跳过，这里只是说明）
        skip_config_recommend: 若 meta.download_config 已存在则跳过 BenchConfigRecommendNode
        output_file: 结果保存路径，None 则自动命名
    """
    all_benches = load_all_benches()

    if bench_names:
        all_benches = [b for b in all_benches if b.bench_name in bench_names]
        log.info(f"指定验证 {len(all_benches)} 个 benchmark: {bench_names}")
    else:
        log.info(f"验证全部 {len(all_benches)} 个 benchmark")

    # 若 download_config 已存在，跳过 BenchConfigRecommendNode（可选）
    if skip_config_recommend:
        to_verify = all_benches
    else:
        to_verify = all_benches

    state = NodeState(
        user_query="[bench_verify] batch validation",
        benches=to_verify,
    )

    # === Phase 2: DatasetStructureNode ===
    log.info("=" * 60)
    log.info("Phase 2-1: DatasetStructureNode")
    log.info("=" * 60)
    node_struct = DatasetStructureNode()
    state = await node_struct.run(state)

    # === Phase 2: BenchConfigRecommendNode ===
    log.info("=" * 60)
    log.info("Phase 2-2: BenchConfigRecommendNode")
    log.info("=" * 60)
    node_config = BenchConfigRecommendNode()
    state = await node_config.run(state)

    # === Phase 2: DownloadNode（限制条数）===
    log.info("=" * 60)
    log.info(f"Phase 2-3: DownloadNode (max_rows={max_rows})")
    log.info("=" * 60)
    node_download = DownloadNode()
    with _patch_download_limited(max_rows):
        state = await node_download.run(state)

    # === Phase 3: DatasetKeysNode ===
    log.info("=" * 60)
    log.info("Phase 3-1: DatasetKeysNode")
    log.info("=" * 60)
    node_keys = DatasetKeysNode()
    state = await node_keys.run(state)

    # === Phase 3: BenchTaskInferNode ===
    log.info("=" * 60)
    log.info("Phase 3-2: BenchTaskInferNode")
    log.info("=" * 60)
    node_infer = BenchTaskInferNode()
    state = await node_infer.run(state)

    # === 汇总结果 ===
    results = _summarize(state.benches)
    _print_report(results)

    # 保存结果
    if output_file is None:
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f"bench_verify_{int(time.time())}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"结果已保存: {output_file}")

    return results


def _summarize(benches: list[BenchInfo]) -> dict:
    passed = []
    failed = []

    for b in benches:
        entry = {
            "bench_name": b.bench_name,
            "download_status": b.download_status,
            "dataset_cache": b.dataset_cache,
            "bench_keys": b.bench_keys,
            "bench_dataflow_eval_type": b.bench_dataflow_eval_type,
            "key_mapping": (b.meta or {}).get("key_mapping"),
            "download_config": (b.meta or {}).get("download_config"),
            "download_error": (b.meta or {}).get("download_error"),
            "structure_error": (b.meta or {}).get("structure_error"),
        }

        # 判断是否完全通过（能运行评测的前一步）
        ok = (
            b.download_status == "success"
            and b.dataset_cache
            and b.bench_keys
            and b.bench_dataflow_eval_type
            and (b.meta or {}).get("key_mapping")
        )

        if ok:
            passed.append(entry)
        else:
            # 记录失败原因
            reasons = []
            if b.download_status != "success":
                reasons.append(f"download={b.download_status}")
            if not b.dataset_cache:
                reasons.append("no dataset_cache")
            if not b.bench_keys:
                reasons.append("no bench_keys")
            if not b.bench_dataflow_eval_type:
                reasons.append("no eval_type")
            if not (b.meta or {}).get("key_mapping"):
                reasons.append("no key_mapping")
            entry["fail_reasons"] = reasons
            failed.append(entry)

    return {
        "total": len(benches),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": f"{len(passed)/len(benches)*100:.1f}%" if benches else "0%",
        "passed_benches": passed,
        "failed_benches": failed,
    }


def _print_report(results: dict):
    log.info("=" * 60)
    log.info("验证报告")
    log.info("=" * 60)
    log.info(f"总计: {results['total']} | 通过: {results['passed']} | 失败: {results['failed']} | 通过率: {results['pass_rate']}")

    if results["passed_benches"]:
        log.info(f"\n✅ 通过的 benchmark ({results['passed']}):")
        for b in results["passed_benches"]:
            log.info(f"  {b['bench_name']} | eval_type={b['bench_dataflow_eval_type']}")

    if results["failed_benches"]:
        log.info(f"\n❌ 失败的 benchmark ({results['failed']}):")
        for b in results["failed_benches"]:
            log.info(f"  {b['bench_name']} | 原因: {b.get('fail_reasons')} | download_error={b.get('download_error')}")


if __name__ == "__main__":
    import sys

    # 用法：
    #   python workflow_bench_verify.py                  # 验证全部
    #   python workflow_bench_verify.py gsm8k mmlu       # 验证指定的
    #   python workflow_bench_verify.py --max-rows 5     # 只下5条

    bench_names = None
    max_rows = MAX_ROWS_PER_BENCH

    args = sys.argv[1:]
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == "--max-rows" and i + 1 < len(args):
            max_rows = int(args[i + 1])
            i += 2
        else:
            filtered_args.append(args[i])
            i += 1

    if filtered_args:
        bench_names = filtered_args

    asyncio.run(verify_benches(
        bench_names=bench_names,
        max_rows=max_rows,
    ))
