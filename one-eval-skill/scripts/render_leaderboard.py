#!/usr/bin/env python3
"""
render_leaderboard.py — 把用户自评模型的分数穿插进公开模型分数里排名，输出 markdown。

为什么需要：
  评测分数单看是「0.94」，没有参照系。把它放到同一 bench 的公开模型分数里排个名，
  用户一眼就知道「自己测的模型大概在什么水位」。这是面向「模型性能」的报告该有的样子，
  而不是孤零零一个数。

怎么用（确定性脚本，不做 LLM 编排）：
  python scripts/render_leaderboard.py --results eval_outputs/eval_results.json
  # 可选 --scores 指定自定义分数表；默认读 references/leaderboard_scores.json
  # 输出 markdown 到 stdout（或 --out 写文件），由 agent 嵌进报告的 leaderboard 小节。

可信度原则：
  公开分数全部带来源标注，脚本原样展示 source/setting/as_of，绝不洗掉出处。
  setting 与用户评测可能不同（shot 数 /CoT/子集），脚本会在表头标注「设置不同仅供参考」，
  排名只是粗略定位，不是严格 SOTA 比较。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402

DEFAULT_SCORES = common.SKILL_DIR / "references" / "leaderboard_scores.json"


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_result_index(metrics: dict) -> dict:
    return {
        r.get("bench_name"): r
        for r in (metrics or {}).get("metric_results", []) or []
        if r.get("bench_name")
    }


def _primary_score_from_metric_row(row: dict):
    primary = row.get("primary_metric_result")
    if isinstance(primary, dict) and primary.get("score") is not None:
        return primary.get("score")
    for val in (row.get("metrics") or {}).values():
        if isinstance(val, dict) and val.get("priority") == "primary" and val.get("score") is not None:
            return val.get("score")
    return None


def _user_scores(results: dict, metrics: dict | None = None) -> dict:
    """取每个 bench 的主分数；缺 primary 时才 fallback 到 DataFlow diagnostic。"""
    model = results.get("model") or "（待评测模型）"
    out = {}
    metric_rows = _metric_result_index(metrics or {})
    for r in results.get("results", []) or []:
        if not r.get("ok"):
            continue
        score = None
        if isinstance(r.get("primary_metric_result"), dict):
            score = r["primary_metric_result"].get("score")
        if score is None:
            score = _primary_score_from_metric_row(metric_rows.get(r.get("bench_name"), {}))
        if score is None:
            score = (r.get("dataflow_score") or {}).get("score")
        if score is None:
            continue
        out[r.get("bench_name")] = score
    return model, out


def _rank_table(bench: str, user_model: str, user_score: float, public: list) -> str:
    """生成单个 bench 的排名 markdown 表，用户模型用 ★ 标出。"""
    rows = []
    for e in public:
        rows.append({
            "model": e.get("model", "?"),
            "score": e.get("score"),
            "setting": e.get("setting", "—"),
            "source": e.get("source", "—"),
            "source_url": e.get("source_url", ""),
            "as_of": e.get("as_of", "—"),
            "is_user": False,
        })
    rows.append({
        "model": f"{user_model}（本次评测）",
        "score": user_score,
        "setting": "见报告「评测设置」",
        "source": "本次评测",
        "source_url": "",
        "as_of": "—",
        "is_user": True,
    })
    # 分数降序；None 垫底
    rows.sort(key=lambda x: (x["score"] is None, -(x["score"] or 0)))

    lines = [
        f"#### {bench}",
        "",
        "> 公开分数来源/设置见末列；各家 shot 数、是否 CoT、子集可能不同，**排名仅供粗略定位，非严格对标**。",
        "",
        "| 排名 | 模型 | 主分数 | 设置 | 来源（as_of） |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        mark = "★ " if r["is_user"] else ""
        s = f"{r['score']:.3f}" if isinstance(r["score"], (int, float)) else "—"
        src = r["source"]
        if r["source_url"]:
            src = f"[{src}]({r['source_url']})"
        src = f"{src}（{r['as_of']}）" if r["as_of"] != "—" else src
        lines.append(f"| {i} | {mark}{r['model']} | {s} | {r['setting']} | {src} |")
    lines.append("")
    return "\n".join(lines)


def build_markdown(results: dict, scores: dict, metrics: dict | None = None) -> str:
    user_model, user_bench_scores = _user_scores(results, metrics)
    table = scores.get("benchmarks", {}) or {}

    parts = ["## Leaderboard：本模型在公开分数中的位置", ""]
    matched = 0
    no_ref = []
    for bench, uscore in user_bench_scores.items():
        public = table.get(bench)
        if not public:
            no_ref.append(bench)
            continue
        parts.append(_rank_table(bench, user_model, uscore, public))
        matched += 1

    if matched == 0:
        parts.append(
            "_本次评测的 bench 在分数表里都没有公开参照（可在 "
            "`references/leaderboard_scores.json` 按格式补充带来源的分数后重跑）。_\n"
        )
    if no_ref:
        parts.append(
            f"> 无公开参照、未参与排名的 bench：{', '.join(no_ref)}。"
            f"如需对标，请在 `references/leaderboard_scores.json` 补充其公开分数（务必带来源）。\n"
        )
    return "\n".join(parts)


def main(argv=None):
    p = argparse.ArgumentParser(description="渲染 leaderboard 排名 markdown")
    p.add_argument("--results", required=True, help="eval_results.json 路径")
    p.add_argument("--metrics", help="metric_results.json 路径（推荐，包含 primary metric）")
    p.add_argument("--scores", default=str(DEFAULT_SCORES), help="公开分数表 json 路径")
    p.add_argument("--out", help="输出 markdown 文件路径（默认打印到 stdout）")
    args = p.parse_args(argv or sys.argv[1:])

    if not Path(args.results).exists():
        print(f"✗ 结果文件不存在: {args.results}", file=sys.stderr)
        return 2
    results = _load(args.results)
    metrics = _load(args.metrics) if args.metrics and Path(args.metrics).exists() else {}
    scores = _load(args.scores) if Path(args.scores).exists() else {"benchmarks": {}}

    md = build_markdown(results, scores, metrics)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"✓ leaderboard 已写入: {out.absolute()}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
