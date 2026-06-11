#!/usr/bin/env python3
"""
render_report.py — 把评测结果渲染成「自包含单文件 HTML 报告」。

为什么要它：
  markdown 报告 + PNG 图只能静态呈现；HTML 报告自由度高得多——leaderboard 可交互
  （点列头排序、hover 看来源）、本模型高亮、配色统一，一个文件双击即开，断网也能看。

设计原则：
  - 单文件：CSS/JS 全部内联，**零 CDN、零第三方库**，可入库、可离线、可长期复用。
  - 图表用纯 CSS/SVG 条形（不引 echarts 等），保证离线打开。
  - 数据在生成时序列化进页面，满足「每次套数据进去」。
  - 右上角 One-Eval(OpenDCAI) 仓库 logo，点击跳转上游官方仓库。
  - 不复盘评测流程，只面向「被评测模型性能」；评测设置如实附在末尾。

用法：
  python scripts/render_report.py --results eval_outputs/eval_results.json \
      --metrics eval_outputs/metric_results.json \
      --out eval_outputs/report.html
  # --metrics 可选；--scores 默认读 references/leaderboard_scores.json
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402

DEFAULT_SCORES = common.SKILL_DIR / "references" / "leaderboard_scores.json"
REPO_URL = "https://github.com/OpenDCAI/One-Eval"


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_overview(results: dict) -> dict:
    """从 eval_results.json 提炼总览：模型名 + 每 bench 主分/有效样本/模式。"""
    model = results.get("model") or "（待评测模型）"
    benches = []
    for r in results.get("results", []) or []:
        ds = r.get("dataflow_score") or {}
        benches.append({
            "bench_name": r.get("bench_name"),
            "eval_type": r.get("bench_dataflow_eval_type"),
            "score": ds.get("score"),
            "accuracy": ds.get("accuracy"),
            "total": ds.get("total_samples"),
            "valid": ds.get("valid_samples"),
            "metric": ds.get("metric"),
            "mode": r.get("mode"),
            "ok": r.get("ok"),
            "detail_path": r.get("detail_path"),
            "elapsed_sec": r.get("elapsed_sec"),
            "external": r.get("mode") == "external_repo_pending",
        })
    return {"model": model, "benches": benches}


def extract_metrics(metrics: dict) -> dict:
    """从 metric_results.json 提炼 bench × metric 分数矩阵（只取 score，不带逐样本 details）。"""
    if not metrics:
        return {"metric_names": [], "rows": []}
    names: list[str] = []
    rows = []
    for mr in metrics.get("metric_results", []) or []:
        cell = {}
        for mname, mval in (mr.get("metrics") or {}).items():
            if mname not in names:
                names.append(mname)
            score = mval.get("score") if isinstance(mval, dict) else mval
            cell[mname] = score
        rows.append({"bench_name": mr.get("bench_name"),
                     "num_samples": mr.get("num_samples"), "cells": cell})
    return {"metric_names": names, "rows": rows}


def build_leaderboard_data(overview: dict, scores: dict) -> dict:
    """把本模型主分穿插进公开分，生成每 bench 的排名行（供前端渲染条形图 + 排序）。

    输出结构：
      { "boards": [ {bench, user_score, rows:[{model,score,setting,source,
                     source_url,as_of,is_user}], max_score} ], "no_ref": [...] }
    """
    model = overview["model"]
    table = (scores or {}).get("benchmarks", {}) or {}
    user_scores = {b["bench_name"]: b["score"] for b in overview["benches"]
                   if b.get("ok") and b.get("score") is not None}

    boards = []
    no_ref = []
    for bench, uscore in user_scores.items():
        public = table.get(bench)
        if not public:
            no_ref.append(bench)
            continue
        rows = []
        for e in public:
            rows.append({
                "model": e.get("model", "?"), "score": e.get("score"),
                "setting": e.get("setting", "—"), "source": e.get("source", "—"),
                "source_url": e.get("source_url", ""), "as_of": e.get("as_of", "—"),
                "note": e.get("note", ""), "is_user": False,
            })
        rows.append({
            "model": f"{model}", "score": uscore,
            "setting": "本次评测（见末尾「评测设置」）", "source": "本次评测",
            "source_url": "", "as_of": "—", "note": "", "is_user": True,
        })
        rows.sort(key=lambda x: (x["score"] is None, -(x["score"] or 0)))
        valid = [r["score"] for r in rows if isinstance(r["score"], (int, float))]
        boards.append({"bench": bench, "user_score": uscore,
                       "rows": rows, "max_score": max(valid) if valid else 1.0})
    return {"boards": boards, "no_ref": no_ref}


def _esc(v) -> str:
    return html.escape(str(v), quote=True)


CSS = """
:root{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2330; --border:#2d3748;
  --txt:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --star:#f0b429;
  --bar:#3b6fb0; --bar-user:#f0b429; --good:#3fb950; --bad:#f85149;
  --shadow:0 1px 3px rgba(0,0,0,.4);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
  "Hiragino Sans GB","Microsoft YaHei",sans-serif;line-height:1.6;font-size:15px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 80px}
header.top{position:sticky;top:0;z-index:10;background:rgba(13,17,23,.92);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 24px;margin-bottom:8px}
header.top .title{font-size:17px;font-weight:600;letter-spacing:.3px}
header.top .title .sub{color:var(--muted);font-weight:400;font-size:13px;margin-left:10px}
.repo-logo{display:flex;align-items:center;gap:8px;padding:6px 12px;
  border:1px solid var(--border);border-radius:8px;background:var(--panel);
  color:var(--txt);font-size:13px;font-weight:500;transition:.15s}
.repo-logo:hover{border-color:var(--accent);text-decoration:none;background:var(--panel2)}
.repo-logo svg{width:20px;height:20px;fill:currentColor}
h1{font-size:26px;margin:28px 0 6px}
h2{font-size:20px;margin:40px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--border)}
h3{font-size:16px;margin:24px 0 10px;color:var(--txt)}
.meta{color:var(--muted);font-size:13px;margin-bottom:8px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;margin:18px 0}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;
  padding:16px 18px;box-shadow:var(--shadow)}
.card .bn{font-size:13px;color:var(--muted);margin-bottom:6px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .sc{font-size:30px;font-weight:700;color:var(--accent)}
.card .ex{font-size:12px;color:var(--muted);margin-top:6px}
.card .tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:6px;
  background:var(--panel2);color:var(--muted);margin-top:8px}
.callout{background:var(--panel);border-left:3px solid var(--accent);
  border-radius:6px;padding:12px 16px;margin:16px 0;color:var(--txt)}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:14px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;
  letter-spacing:.4px;cursor:pointer;user-select:none}
th.nosort{cursor:default}
th:hover:not(.nosort){color:var(--accent)}
tr:hover td{background:var(--panel)}
.lb-board{background:var(--panel);border:1px solid var(--border);
  border-radius:10px;padding:18px 20px;margin:18px 0;box-shadow:var(--shadow)}
.lb-board h3{margin-top:0;display:flex;align-items:center;gap:10px}
.lb-hint{font-size:12px;color:var(--muted);margin:2px 0 14px}
.bar-row{display:grid;grid-template-columns:210px 1fr 64px;align-items:center;
  gap:10px;padding:5px 0;cursor:default}
.bar-row .nm{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-row.user .nm{color:var(--star);font-weight:700}
.bar-track{background:var(--panel2);border-radius:5px;height:20px;overflow:hidden;position:relative}
.bar-fill{height:100%;border-radius:5px;background:linear-gradient(90deg,#2c5282,#3b6fb0);
  transition:width .5s ease}
.bar-row.user .bar-fill{background:linear-gradient(90deg,#d69e2e,#f0b429)}
.bar-val{font-size:13px;font-variant-numeric:tabular-nums;text-align:right;color:var(--muted)}
.bar-row.user .bar-val{color:var(--star);font-weight:700}
.rank{display:inline-block;width:22px;color:var(--muted);font-size:12px;text-align:right;margin-right:6px}
.src{font-size:11px;color:var(--muted);margin-left:4px}
.heat td{text-align:center;font-variant-numeric:tabular-nums}
.heat td.bn{text-align:left;color:var(--muted)}
.detail{background:var(--panel);border:1px solid var(--border);border-radius:10px;
  padding:16px 20px;margin:14px 0}
.detail .row{display:flex;flex-wrap:wrap;gap:18px;font-size:13px;color:var(--muted);margin:6px 0}
.detail .row b{color:var(--txt);font-weight:600}
.path{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
  color:var(--muted);word-break:break-all}
.kv{font-size:13px}.kv b{display:inline-block;min-width:130px;color:var(--muted);font-weight:500}
footer{margin-top:50px;padding-top:18px;border-top:1px solid var(--border);
  color:var(--muted);font-size:12px;text-align:center}
.pill{display:inline-block;font-size:11px;padding:1px 8px;border-radius:10px}
.pill.full{background:rgba(63,185,80,.15);color:var(--good)}
.pill.smoke{background:rgba(240,180,41,.15);color:var(--star)}
.pill.ext{background:rgba(88,166,255,.15);color:var(--accent)}
"""


GH_SVG = ('<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 '
          '8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37'
          '-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08'
          '.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64'
          '-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 '
          '1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 '
          '2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 '
          '1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z">'
          '</path></svg>')


def fmt_score(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def build_header(model: str) -> str:
    return (f'<header class="top"><div class="title">One-Eval 评测报告'
            f'<span class="sub">{_esc(model)}</span></div>'
            f'<a class="repo-logo" href="{REPO_URL}" target="_blank" rel="noopener" '
            f'title="前往 One-Eval 官方仓库">{GH_SVG}<span>One-Eval</span></a></header>')


def build_overview(overview: dict, lb: dict) -> str:
    benches = overview["benches"]
    n = len(benches)
    total = sum((b.get("total") or 0) for b in benches)
    cards = []
    for b in benches:
        if b["external"]:
            sc = '<div class="sc" style="font-size:18px;color:var(--accent)">外部待执行</div>'
            ex = "external_repo（需沙箱回填）"
        else:
            sc = f'<div class="sc">{fmt_score(b["score"])}</div>'
            ex = f'有效 {b.get("valid","?")}/{b.get("total","?")} ｜ {_esc(b.get("metric") or "")}'
        mode = b.get("mode") or ""
        pill = "ext" if b["external"] else ("full" if mode == "full" else "smoke")
        cards.append(
            f'<div class="card"><div class="bn" title="{_esc(b["bench_name"])}">'
            f'{_esc(b["bench_name"])}</div>{sc}<div class="ex">{ex}</div>'
            f'<span class="pill {pill}">{_esc(mode or "—")}</span> '
            f'<span class="tag">{_esc(b.get("eval_type") or "")}</span></div>')
    ranked = len(lb["boards"])
    return (
        '<h1>评测总览</h1>'
        f'<div class="meta">生成时间 {datetime.now():%Y-%m-%d %H:%M} ｜ '
        f'数据来源 eval_results.json（+ metric_results.json）</div>'
        f'<div class="callout"><b>{_esc(overview["model"])}</b> 共评测 <b>{n}</b> 个 '
        f'benchmark、合计约 <b>{total}</b> 条样本；其中 <b>{ranked}</b> 个有公开分参照、'
        f'已穿插进 leaderboard 排名（下节）。各 bench 主分见下方卡片。</div>'
        f'<div class="cards">{"".join(cards)}</div>')


def build_leaderboard(lb: dict) -> str:
    if not lb["boards"] and not lb["no_ref"]:
        return ""
    parts = ['<h2>Leaderboard：本模型在公开分中的位置</h2>',
             '<div class="lb-hint">★ 为本次评测模型。公开分各家 shot 数 / 是否 CoT / 子集'
             '可能不同，<b>排名仅供粗略定位，非严格对标</b>；hover 条形看来源与设置。</div>']
    for bd in lb["boards"]:
        bars = []
        for i, r in enumerate(bd["rows"], 1):
            pct = (r["score"] / bd["max_score"] * 100) if isinstance(r["score"], (int, float)) and bd["max_score"] else 0
            star = "★ " if r["is_user"] else ""
            src = r["source"]
            if r["source_url"]:
                src = f'<a href="{_esc(r["source_url"])}" target="_blank" rel="noopener">{_esc(src)}</a>'
            asof = f'（{_esc(r["as_of"])}）' if r["as_of"] != "—" else ""
            tip = _esc(f'{r["model"]} ｜ {fmt_score(r["score"])} ｜ {r["setting"]} ｜ {r["source"]}{r["as_of"] if r["as_of"]!="—" else ""}')
            cls = "bar-row user" if r["is_user"] else "bar-row"
            bars.append(
                f'<div class="{cls}" title="{tip}"><div class="nm">'
                f'<span class="rank">{i}</span>{star}{_esc(r["model"])}'
                f'<span class="src">· {src}{asof}</span></div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
                f'<div class="bar-val">{fmt_score(r["score"])}</div></div>')
        parts.append(
            f'<div class="lb-board"><h3>{_esc(bd["bench"])} '
            f'<span class="src">（{len(bd["rows"])-1} 个公开模型 + 本模型）</span></h3>'
            f'{"".join(bars)}</div>')
    if lb["no_ref"]:
        names = "、".join(_esc(b) for b in lb["no_ref"])
        parts.append(
            f'<div class="callout">无公开参照、未参与排名的 bench：<b>{names}</b>。'
            f'如需对标，请在 <span class="path">references/leaderboard_scores.json</span> '
            f'按格式补充其公开分数（务必带来源）后重跑。</div>')
    return "".join(parts)


def build_metric_section(mx: dict) -> str:
    if not mx["metric_names"] or not mx["rows"]:
        return ""
    heads = "".join(f'<th>{_esc(m)}</th>' for m in mx["metric_names"])
    rows = []
    for r in mx["rows"]:
        tds = [f'<td class="bn">{_esc(r["bench_name"])}</td>']
        for m in mx["metric_names"]:
            v = r["cells"].get(m)
            if isinstance(v, (int, float)):
                # 0→红 1→绿 线性着色
                g = int(60 + v * 120); rr = int(180 - v * 120)
                bg = f'background:rgba({rr},{g},80,.22)'
                tds.append(f'<td style="{bg}">{v:.3f}</td>')
            else:
                tds.append('<td>—</td>')
        rows.append(f'<tr>{"".join(tds)}</tr>')
    return (
        '<h2>多维度 Metric</h2>'
        '<div class="lb-hint">dataflow 主分之外的补充维度。颜色越绿越高；'
        '低分先看正确性是「能力不足」还是「格式/抽取失败」。</div>'
        f'<table class="heat"><thead><tr><th class="nosort">Benchmark</th>{heads}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>')


def build_details(overview: dict) -> str:
    parts = ['<h2>逐 Benchmark 详情</h2>']
    for b in overview["benches"]:
        if b["external"]:
            body = ('<div class="row">该 bench 为 <b>external_repo</b> 类型，需在外部仓库/'
                    '沙箱执行后回填分数（详见 references/external_bench.md）。</div>')
        else:
            el = f'{b["elapsed_sec"]:.1f}s' if isinstance(b.get("elapsed_sec"), (int, float)) else "—"
            body = (
                f'<div class="row"><span><b>主分数</b> {fmt_score(b["score"])}</span>'
                f'<span><b>有效样本</b> {b.get("valid","?")}/{b.get("total","?")}</span>'
                f'<span><b>主指标</b> {_esc(b.get("metric") or "—")}</span>'
                f'<span><b>模式</b> {_esc(b.get("mode") or "—")}</span>'
                f'<span><b>耗时</b> {el}</span></div>')
            if b.get("detail_path"):
                body += f'<div class="row"><b>逐样本明细</b> <span class="path">{_esc(b["detail_path"])}</span></div>'
        parts.append(
            f'<div class="detail"><h3>{_esc(b["bench_name"])} '
            f'<span class="tag">{_esc(b.get("eval_type") or "")}</span></h3>{body}</div>')
    return "".join(parts)


def build_settings(overview: dict, args, paths: dict) -> str:
    rows = [
        ('采样规模', '各 bench 见上方「模式」（full=全量 / smoke=3 条抽样）'),
        ('使用 metric', '主分（各 eval_type 默认）' + (' + 见「多维度 Metric」节' if paths.get('metrics') else '')),
        ('eval_results', f'<span class="path">{_esc(paths["results"])}</span>'),
    ]
    if paths.get("metrics"):
        rows.append(('metric_results', f'<span class="path">{_esc(paths["metrics"])}</span>'))
    rows.append(('leaderboard 分数表', f'<span class="path">{_esc(paths["scores"])}</span>'))
    rows.append(('本报告', f'<span class="path">{_esc(paths["out"])}</span>'))
    body = "".join(f'<div class="kv"><b>{k}</b>{v}</div>' for k, v in rows)
    return ('<h2>附：评测设置（供判断分数可比性）</h2>'
            '<div class="callout" style="border-left-color:var(--muted)">'
            '模型生成参数（temperature / top_p / max_tokens / seed）记录于本次 '
            'evalspec.yaml；如需写入报告，请在生成时补充。</div>'
            f'<div class="detail">{body}</div>')


JS = """
// 表格列头点击排序（数值/文本自适应），用于 metric 热力表等普通表格。
document.querySelectorAll('table').forEach(function(tbl){
  var heads = tbl.querySelectorAll('th');
  heads.forEach(function(th, idx){
    if (th.classList.contains('nosort')) return;
    var dir = 1;
    th.addEventListener('click', function(){
      var tb = tbl.querySelector('tbody'); if(!tb) return;
      var rows = Array.prototype.slice.call(tb.querySelectorAll('tr'));
      rows.sort(function(a,b){
        var x=a.children[idx].textContent.trim(), y=b.children[idx].textContent.trim();
        var nx=parseFloat(x), ny=parseFloat(y);
        if(!isNaN(nx)&&!isNaN(ny)) return (nx-ny)*dir;
        return x.localeCompare(y,'zh')*dir;
      });
      dir*=-1; rows.forEach(function(r){tb.appendChild(r);});
    });
  });
});
"""


def render_html(results: dict, metrics: dict, scores: dict, args, paths: dict) -> str:
    overview = extract_overview(results)
    mx = extract_metrics(metrics)
    lb = build_leaderboard_data(overview, scores)

    body = (build_header(overview["model"]) + '<div class="wrap">'
            + build_overview(overview, lb)
            + build_leaderboard(lb)
            + build_metric_section(mx)
            + build_details(overview)
            + build_settings(overview, args, paths)
            + '<footer>由 One-Eval skill 自动生成 · '
            f'<a href="{REPO_URL}" target="_blank" rel="noopener">github.com/OpenDCAI/One-Eval</a>'
            ' · 排名仅供粗略定位，公开分以各自来源为准</footer></div>')
    return (f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>One-Eval 评测报告 · {_esc(overview["model"])}</title>'
            f'<style>{CSS}</style></head><body>{body}'
            f'<script>{JS}</script></body></html>')


def main(argv=None):
    p = argparse.ArgumentParser(description="生成自包含 HTML 评测报告")
    p.add_argument("--results", required=True, help="eval_results.json 路径")
    p.add_argument("--metrics", help="metric_results.json 路径（可选）")
    p.add_argument("--scores", default=str(DEFAULT_SCORES), help="公开分数表 json 路径")
    p.add_argument("--out", help="输出 html 路径（默认 eval_outputs/report.html）")
    p.add_argument("--no-open", action="store_true", help="生成后不自动在浏览器打开")
    args = p.parse_args(argv or sys.argv[1:])

    if not Path(args.results).exists():
        print(f"✗ 结果文件不存在: {args.results}", file=sys.stderr)
        return 2
    results = _load(args.results)
    metrics = _load(args.metrics) if args.metrics and Path(args.metrics).exists() else {}
    scores = _load(args.scores) if Path(args.scores).exists() else {"benchmarks": {}}

    out = Path(args.out) if args.out else Path(args.results).parent / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": str(Path(args.results).absolute()),
        "metrics": str(Path(args.metrics).absolute()) if args.metrics and Path(args.metrics).exists() else "",
        "scores": str(Path(args.scores).absolute()),
        "out": str(out.absolute()),
    }
    html_text = render_html(results, metrics, scores, args, paths)
    out.write_text(html_text, encoding="utf-8")
    print(f"✓ HTML 报告已生成: {out.absolute()}")
    if args.no_open:
        print(f"  在浏览器打开即可查看（单文件、零依赖、可离线）。")
    else:
        try:
            webbrowser.open(out.absolute().as_uri())
            print(f"  已尝试在默认浏览器打开（单文件、零依赖、可离线）。")
        except Exception:
            print(f"  自动打开失败，请手动在浏览器打开上面的路径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
