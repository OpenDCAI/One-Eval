#!/usr/bin/env python3
"""
prepare_bench.py — benchmark 数据准备 + 结构预览（接入新 bench 前的探查工具）。

用途：
  当 agent 要把一个「不在 gallery 里的新 HF 数据集」接进评测时，先用本脚本：
    1. 从 HuggingFace 下载并转成 jsonl（HFDownloadTool）
    2. 预览前几条样本的 **嵌套 key 结构**（点路径形式，如 a.b.c）
    3. 据此让 agent 判断该数据集属于 6 种 eval_type 中的哪一种、
       以及 key_mapping 该怎么填（多层嵌套时需先写转换代码拍平到目标格式）

为什么单独成一个工具：
  评测内核（run_eval.py）只接受「已拍平、key 对齐」的数据。新数据集字段往往多层嵌套，
  必须先探查结构、规划 key_mapping，必要时写预处理代码，才能进入正式评测。

用法：
  # 下载并预览结构（默认预览 3 条、展开嵌套）
  python prepare_bench.py --repo-id HuggingFaceH4/MATH-500 --split test

  # 指定子配置 + 输出路径
  python prepare_bench.py --repo-id cais/mmlu --config abstract_algebra --split test

  # 只预览已下载的本地 jsonl（不重新下载）
  python prepare_bench.py --preview-only path/to/data.jsonl

退出码：0 = 成功；非 0 = 下载或预览失败（stderr 给出可读原因与配置指引）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402

PREVIEW_SAMPLES = 3


def _flatten_keys(obj, prefix: str = "") -> dict:
    """递归展开嵌套结构，返回 {点路径: 值类型/样例}。

    list 取第 0 个元素探查其结构，路径标记为 key[]。
    """
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            out.update(_flatten_keys(v, path))
    elif isinstance(obj, list):
        path = f"{prefix}[]"
        if obj:
            out.update(_flatten_keys(obj[0], path))
        else:
            out[path] = "(empty list)"
    else:
        out[prefix] = _describe_value(obj)
    return out


def _describe_value(v) -> str:
    """对标量给出「类型 + 截断样例」描述。"""
    t = type(v).__name__
    s = str(v).replace("\n", " ")
    if len(s) > 60:
        s = s[:60] + "..."
    return f"<{t}> {s}"


def _preview_jsonl(path: str, n: int = PREVIEW_SAMPLES) -> dict:
    """读前 n 条，汇总所有出现过的嵌套 key 点路径。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break

    if not rows:
        return {"ok": False, "reason": "数据为空，无法预览结构"}

    # 汇总所有样本的 key 点路径（取并集，值描述用第一条出现的）
    merged = {}
    for r in rows:
        for k, desc in _flatten_keys(r).items():
            merged.setdefault(k, desc)

    return {
        "ok": True,
        "num_previewed": len(rows),
        "key_paths": merged,
        "first_sample": rows[0],
    }


def _download(repo_id: str, config_name, split: str, out_path: Path) -> dict:
    """调 HFDownloadTool 下载转换；失败时给出可读指引。"""
    try:
        from one_eval.toolkits.hf_download_tool import HFDownloadTool
    except Exception as e:
        return {"ok": False, "reason": f"无法导入 HFDownloadTool（评测引擎未装好）: {e}"}

    tool = HFDownloadTool(cache_dir=str(out_path.parent))
    try:
        res = tool.download_and_convert(
            repo_id=repo_id, config_name=config_name, split=split, output_path=out_path,
        )
    except Exception as e:
        return _download_error_hint(repo_id, f"{type(e).__name__}: {e}")

    if not res.get("ok"):
        return _download_error_hint(repo_id, res.get("error", "未知错误"))
    return {"ok": True, "output_path": str(out_path),
            "num_rows": res.get("num_rows"), "columns": res.get("columns")}


def _download_error_hint(repo_id: str, err: str) -> dict:
    """把底层下载错误翻译成「带配置指引」的可读原因。"""
    low = err.lower()
    hint = ""
    if "401" in err or "403" in err or "gated" in low or "authentication" in low:
        hint = ("该数据集可能是 gated/private，或需要 HF token。请配置环境变量 "
                "HF_TOKEN=<你的 token> 后重试（在 https://huggingface.co/settings/tokens 申请）。")
    elif ("timeout" in low or "connection" in low or "max retries" in low
          or "name resolution" in low or "ssl" in low):
        hint = ("网络无法访问 huggingface.co。国内可设置镜像："
                "export HF_ENDPOINT=https://hf-mirror.com 后重试。")
    elif "404" in err or "not found" in low or "doesn't exist" in low:
        hint = (f"仓库或配置不存在：核对 repo_id={repo_id!r} 是否正确，"
                "以及该数据集是否需要指定 --config / --split。")
    elif "config" in low and "split" in low:
        hint = "需要指定子配置：用 --config <name> 选择该数据集的子集。"
    return {"ok": False, "reason": err, "hint": hint}


def _print_preview(prev: dict) -> None:
    print(f"\n  预览 {prev['num_previewed']} 条样本的字段结构（点路径）：")
    for path, desc in prev["key_paths"].items():
        print(f"    {path:<40} {desc}")
    print("\n  → 据此判断 eval_type 与 key_mapping。多层嵌套（含 [] 或 a.b.c）"
          "需先写预处理代码拍平到顶层 key 再评测。")


def main(argv=None):
    p = argparse.ArgumentParser(description="benchmark 数据准备 + 嵌套结构预览")
    p.add_argument("--repo-id", help="HF 数据集仓库，如 HuggingFaceH4/MATH-500")
    p.add_argument("--config", dest="config", default=None, help="子配置名（部分数据集必填）")
    p.add_argument("--split", default="test", help="划分（默认 test）")
    p.add_argument("--output-path", help="输出 jsonl 路径（默认写入 skill cache 目录）")
    p.add_argument("--preview-only", help="只预览已存在的本地 jsonl，不下载")
    p.add_argument("--n", type=int, default=PREVIEW_SAMPLES, help="预览条数（默认 3）")
    args = p.parse_args(argv or sys.argv[1:])

    # 模式一：只预览本地文件
    if args.preview_only:
        path = args.preview_only
        if not Path(path).exists():
            print(f"✗ 文件不存在: {path}", file=sys.stderr)
            return 2
        prev = _preview_jsonl(path, args.n)
        if not prev["ok"]:
            print(f"✗ 预览失败: {prev['reason']}", file=sys.stderr)
            return 1
        print(f"✓ 本地数据: {path}")
        _print_preview(prev)
        return 0

    # 模式二：下载 + 预览
    if not args.repo_id:
        print("错误：需要 --repo-id 或 --preview-only", file=sys.stderr)
        return 2

    cache_dir = common.DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        safe = f"{args.repo_id.replace('/', '__')}__{args.config}__{args.split}.jsonl"
        out_path = cache_dir / safe

    print(f"下载 {args.repo_id} (config={args.config}, split={args.split}) ...", flush=True)
    dl = _download(args.repo_id, args.config, args.split, out_path)
    if not dl["ok"]:
        print(f"✗ 下载失败：{dl['reason']}", file=sys.stderr)
        if dl.get("hint"):
            print(f"  指引：{dl['hint']}", file=sys.stderr)
        return 1

    print(f"✓ 已下载 {dl.get('num_rows')} 条 → {dl['output_path']}")
    print(f"  顶层列：{dl.get('columns')}")

    prev = _preview_jsonl(dl["output_path"], args.n)
    if prev["ok"]:
        _print_preview(prev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
