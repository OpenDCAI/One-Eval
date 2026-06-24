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


def _json_for_script(data) -> str:
    text = json.dumps(data or {}, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _metric_result_index(metrics: dict) -> dict:
    return {
        r.get("bench_name"): r
        for r in (metrics or {}).get("metric_results", []) or []
        if r.get("bench_name")
    }


def _primary_from_metric_row(row: dict) -> dict:
    primary = row.get("primary_metric_result")
    if isinstance(primary, dict) and primary.get("score") is not None:
        return dict(primary)
    for name, val in (row.get("metrics") or {}).items():
        if isinstance(val, dict) and val.get("priority") == "primary" and val.get("score") is not None:
            return {
                "metric": name,
                "score": val.get("score"),
                "score_source": "metric_stage",
                "denominator": val.get("denominator"),
            }
    return {}


def extract_overview(results: dict, metrics: dict | None = None) -> dict:
    """从结果提炼总览：模型名 + 每 bench 主分/诊断分/样本/模式。"""
    model = results.get("model") or "（待评测模型）"
    benches = []
    metric_rows = _metric_result_index(metrics or {})
    for r in results.get("results", []) or []:
        ds = r.get("dataflow_score") or {}
        metric_row = metric_rows.get(r.get("bench_name"), {})
        primary = r.get("primary_metric_result") if isinstance(r.get("primary_metric_result"), dict) else {}
        if not primary:
            primary = _primary_from_metric_row(metric_row)
        warning = ""
        if not primary:
            primary = {
                "metric": ds.get("metric"),
                "score": ds.get("score"),
                "score_source": "dataflow_diagnostic_fallback",
                "denominator": "valid_only",
            }
            if ds.get("score") is not None:
                warning = "primary metric was not computed; showing DataFlow diagnostic score"
        benches.append({
            "bench_name": r.get("bench_name"),
            "eval_type": r.get("bench_dataflow_eval_type"),
            "score": primary.get("score"),
            "accuracy": primary.get("score"),
            "total": primary.get("total_samples") or ds.get("total_samples"),
            "valid": primary.get("valid_predictions") or ds.get("valid_samples"),
            "metric": primary.get("metric"),
            "denominator": primary.get("denominator"),
            "score_source": primary.get("score_source"),
            "parser": primary.get("parser"),
            "parse_failed": primary.get("parse_failed"),
            "empty_output": primary.get("empty_output"),
            "scored_samples": primary.get("scored_samples"),
            "official_metric": primary.get("official_metric"),
            "official_compatibility": primary.get("official_compatibility"),
            "warning": warning,
            "dataflow_score": ds.get("score"),
            "dataflow_metric": ds.get("metric"),
            "dataflow_valid": ds.get("valid_samples"),
            "dataflow_total": ds.get("total_samples"),
            "mode": r.get("mode"),
            "ok": r.get("ok"),
            "detail_path": r.get("detail_path"),
            "primary_detail_path": metric_row.get("primary_detail_path") or r.get("primary_detail_path"),
            "elapsed_sec": r.get("elapsed_sec"),
            "external": r.get("mode") == "external_repo_pending",
            "prompt": r.get("prompt"),
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
.report-app{display:grid;grid-template-columns:280px minmax(0,1fr);min-height:100vh}
.report-sidebar{position:sticky;top:0;height:100vh;overflow:auto;background:#0b1017;
  border-right:1px solid var(--border);padding-bottom:14px}
.side-brand{padding:18px 16px 14px;border-bottom:1px solid var(--border)}
.side-brand .name{font-size:18px;font-weight:700;margin-bottom:4px}
.side-brand .sub{color:var(--muted);font-size:12px;word-break:break-all;line-height:1.45}
.side-repo{margin-top:12px;display:inline-flex;align-items:center;gap:7px;color:var(--txt);
  border:1px solid var(--border);border-radius:8px;padding:6px 9px;background:var(--panel);font-size:12px}
.side-repo:hover{border-color:var(--accent);background:var(--panel2);text-decoration:none}
.side-repo svg{width:17px;height:17px;fill:currentColor}
.nav-section{padding:10px 10px 0}
.nav-label{padding:8px 7px 4px;color:var(--muted);font-size:11px;text-transform:uppercase;
  letter-spacing:.5px}
.report-nav-btn{width:100%;border:1px solid transparent;background:transparent;color:var(--txt);
  border-radius:8px;padding:9px 10px;margin:2px 0;text-align:left;cursor:pointer}
.report-nav-btn:hover{background:var(--panel)}
.report-nav-btn.active{background:var(--panel2);border-color:var(--accent)}
.nav-name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nav-meta{display:flex;gap:8px;align-items:center;color:var(--muted);font-size:12px;margin-top:3px}
.nav-dot{width:7px;height:7px;border-radius:50%;display:inline-block;background:var(--muted)}
.nav-dot.ok{background:var(--good)}.nav-dot.bad{background:var(--bad)}
.report-main{min-width:0}
.report-page{display:none;min-height:calc(100vh - 61px)}
.report-page.active{display:block}
.wrap{max-width:1120px;margin:0 auto;padding:0 28px 80px}
.wide-wrap{max-width:none;padding:0 24px 56px}
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
.sample-dashboard{width:100%;background:var(--panel);border:1px solid var(--border);
  border-radius:10px;box-shadow:var(--shadow);overflow:hidden}
.sample-dashboard .sample-top{padding:14px 16px;border-bottom:1px solid var(--border);background:#111824}
.sample-dashboard .sample-title{font-size:16px;font-weight:600;margin-bottom:5px}
.sample-dashboard .sample-meta{display:flex;flex-wrap:wrap;gap:12px;color:var(--muted);font-size:12px}
.sample-dashboard .sample-toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:12px}
.sample-dashboard input,.sample-dashboard select,.sample-dashboard button{font:inherit;background:var(--panel);
  border:1px solid var(--border);color:var(--txt);border-radius:8px;padding:7px 10px}
.sample-dashboard input{width:min(420px,100%)}
.sample-dashboard label{display:flex;gap:6px;align-items:center;color:var(--muted);font-size:13px}
.sample-body{min-height:520px}
.sample-benches{border-right:1px solid var(--border);background:#0f1620;max-height:680px;overflow:auto;padding:8px}
.sample-bench-btn{width:100%;text-align:left;border:1px solid transparent;background:transparent;color:var(--txt);
  border-radius:8px;padding:9px;margin:2px 0;cursor:pointer}
.sample-bench-btn:hover{background:var(--panel2)}
.sample-bench-btn.active{background:#1f2a3a;border-color:var(--accent)}
.sample-bench-name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sample-bench-sub{display:flex;gap:8px;color:var(--muted);font-size:12px;margin-top:4px}
.sample-dot{width:7px;height:7px;border-radius:50%;display:inline-block;background:var(--muted);margin-top:5px}
.sample-dot.ok{background:var(--good)}.sample-dot.bad{background:var(--bad)}
.sample-main{min-width:0;padding:14px}
.sample-notice{border-left:3px solid var(--star);background:#151d29;padding:9px 11px;border-radius:6px;margin-bottom:10px}
.sample-pager{display:flex;justify-content:space-between;align-items:center;gap:10px;color:var(--muted);font-size:13px;margin-bottom:10px}
.sample-page-buttons{display:flex;gap:8px}
.case-layout{display:grid;grid-template-columns:minmax(300px,420px) minmax(0,1fr);gap:14px;align-items:start}
.case-list-pane,.case-detail-pane{min-width:0}
.case-list{display:flex;flex-direction:column;gap:10px;max-height:calc(100vh - 270px);overflow:auto;padding-right:3px}
.case-card{border:1px solid var(--border);background:#101722;border-radius:8px;padding:12px;cursor:pointer}
.case-card:hover{background:#142033}
.case-card.active{border-color:var(--accent);box-shadow:inset 3px 0 0 var(--accent);background:#111c2b}
.case-card-top{display:flex;justify-content:space-between;gap:8px;align-items:flex-start;margin-bottom:7px}
.case-index{font-weight:700;color:var(--txt)}
.case-badges{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px}
.case-badge{font-size:12px;border-radius:999px;padding:2px 8px;background:var(--panel2);color:var(--muted);white-space:nowrap}
.case-badge.good{background:rgba(63,185,80,.16);color:var(--good)}
.case-badge.bad{background:rgba(248,81,73,.16);color:var(--bad)}
.case-question{font-size:14px;font-weight:600;line-height:1.45;margin-bottom:7px;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.case-meta{display:flex;flex-wrap:wrap;gap:8px;color:var(--muted);font-size:12px}
.case-detail-pane{border:1px solid var(--border);border-radius:10px;background:#101722;overflow:hidden}
.case-detail-head{padding:16px 18px;border-bottom:1px solid var(--border);background:#111824}
.case-detail-title{font-size:18px;font-weight:700;line-height:1.45;margin-bottom:9px}
.case-detail-meta{display:flex;flex-wrap:wrap;gap:10px;color:var(--muted);font-size:12px}
.case-fields{display:grid;grid-template-columns:1fr;gap:10px;padding:14px;max-height:calc(100vh - 355px);overflow:auto}
.case-field{border:1px solid var(--border);border-radius:8px;overflow:hidden;background:#0f1620}
.case-field-key{display:flex;justify-content:space-between;gap:10px;padding:8px 11px;background:#151c27;
  border-bottom:1px solid var(--border);font-weight:700}
.case-field-type{color:var(--muted);font-size:12px;font-weight:600}
.case-field-value{padding:10px 11px;white-space:pre-wrap;word-break:break-word;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;line-height:1.5;max-height:320px;overflow:auto}
.sample-table-wrap{border:1px solid var(--border);border-radius:8px;overflow:auto;max-height:650px;background:#111824}
.sample-table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;margin:0}
.sample-table th,.sample-table td{border-bottom:1px solid var(--border);border-right:1px solid var(--border);vertical-align:top}
.sample-table th{position:sticky;top:0;z-index:2;background:#111824;color:var(--muted);font-size:12px;
  text-align:left;padding:8px 9px;white-space:nowrap;cursor:pointer;text-transform:none;letter-spacing:0}
.sample-table td{background:#101722;padding:0;max-width:520px;min-width:110px}
.sample-table tr:hover td{background:#142033}
.sample-table th:first-child,.sample-table td:first-child{position:sticky;left:0;z-index:3}
.sample-table td:first-child{background:#101722;min-width:72px;max-width:90px}
.sample-table th:first-child{z-index:5;background:#111824}
.sample-cell{max-height:190px;overflow:auto;padding:8px 9px;white-space:pre-wrap;word-break:break-word;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;line-height:1.45}
.sample-empty{color:#52606d}.sample-true{color:var(--good);font-weight:700}.sample-false{color:var(--bad);font-weight:700}
footer{margin-top:50px;padding-top:18px;border-top:1px solid var(--border);
  color:var(--muted);font-size:12px;text-align:center}
.pill{display:inline-block;font-size:11px;padding:1px 8px;border-radius:10px}
.pill.full{background:rgba(63,185,80,.15);color:var(--good)}
.pill.smoke{background:rgba(240,180,41,.15);color:var(--star)}
.pill.ext{background:rgba(88,166,255,.15);color:var(--accent)}
@media (max-width:900px){
  .report-app{grid-template-columns:1fr}
  .report-sidebar{position:static;height:auto;border-right:0;border-bottom:1px solid var(--border)}
  .nav-section{display:flex;gap:6px;overflow:auto;padding:8px}
  .nav-label{display:none}
  .report-nav-btn{min-width:170px}
  .wrap,.wide-wrap{padding-left:16px;padding-right:16px}
  header.top{position:static}
  .case-layout{grid-template-columns:1fr}
  .case-list,.case-fields{max-height:none}
}
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


def build_sidebar(overview: dict, sample_data: dict | None = None) -> str:
    sample_benches = {
        b.get("bench_name"): b
        for b in (sample_data or {}).get("benches", []) or []
        if b.get("bench_name")
    }
    bench_buttons = []
    for b in overview.get("benches", []) or []:
        name = b.get("bench_name") or "unknown"
        sample = sample_benches.get(name, {})
        rows = sample.get("loaded_rows")
        rows_text = f'{rows} rows' if isinstance(rows, int) else "detail"
        ok = b.get("ok")
        dot_cls = "ok" if ok else "bad"
        bench_buttons.append(
            '<button class="report-nav-btn" type="button" '
            f'data-report-bench-name="{_esc(name)}">'
            f'<div class="nav-name" title="{_esc(name)}">{_esc(name)}</div>'
            f'<div class="nav-meta"><span class="nav-dot {dot_cls}"></span>'
            f'<span>{_esc(rows_text)}</span><span>{_esc(b.get("metric") or "—")}</span></div>'
            '</button>'
        )
    return (
        '<aside class="report-sidebar">'
        '<div class="side-brand">'
        '<div class="name">One-Eval</div>'
        f'<div class="sub">{_esc(overview.get("model") or "—")}</div>'
        '</div>'
        '<div class="nav-section">'
        '<div class="nav-label">Overview</div>'
        '<button class="report-nav-btn active" type="button" data-report-page="overview">'
        '<div class="nav-name">测评总览</div>'
        f'<div class="nav-meta"><span>{len(overview.get("benches", []) or [])} benches</span></div>'
        '</button>'
        '<div class="nav-label">Bench Details</div>'
        f'{"".join(bench_buttons)}'
        '</div>'
        '</aside>'
    )


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
            source = b.get("score_source") or "—"
            ex = f'有效 {b.get("valid","?")}/{b.get("total","?")} ｜ {_esc(b.get("metric") or "")} ｜ {_esc(source)}'
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
        f'已穿插进 leaderboard 排名（下节）。各 bench 主分优先来自 metric stage；'
        f'DataFlow 分数仅作为 diagnostic。</div>'
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
        '<div class="lb-hint">metric stage 的补充维度。颜色越绿越高；'
        '低分先看正确性是「能力不足」还是「格式/抽取失败」。</div>'
        f'<table class="heat"><thead><tr><th class="nosort">Benchmark</th>{heads}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>')


def build_details(overview: dict, paths: dict | None = None) -> str:
    parts = ['<h2>逐 Benchmark 详情</h2>']
    for b in overview["benches"]:
        if b["external"]:
            body = ('<div class="row">该 bench 为 <b>external_repo</b> 类型，需在外部仓库/'
                    '沙箱执行后回填分数（详见 references/external_bench.md）。</div>')
        else:
            el = f'{b["elapsed_sec"]:.1f}s' if isinstance(b.get("elapsed_sec"), (int, float)) else "—"
            body = (
                f'<div class="row"><span><b>主分数</b> {fmt_score(b["score"])}</span>'
                f'<span><b>有效预测</b> {b.get("valid","?")}/{b.get("total","?")}</span>'
                f'<span><b>主指标</b> {_esc(b.get("metric") or "—")}</span>'
                f'<span><b>来源</b> {_esc(b.get("score_source") or "—")}</span>'
                f'<span><b>分母</b> {_esc(b.get("denominator") or "—")}</span>'
                f'<span><b>模式</b> {_esc(b.get("mode") or "—")}</span>'
                f'<span><b>耗时</b> {el}</span></div>')
            if b.get("parse_failed") is not None or b.get("empty_output") is not None:
                body += (
                    f'<div class="row"><span><b>解析失败</b> {b.get("parse_failed","—")}</span>'
                    f'<span><b>空输出</b> {b.get("empty_output","—")}</span>'
                    f'<span><b>计分样本</b> {b.get("scored_samples","—")}</span></div>'
                )
            if b.get("dataflow_score") is not None:
                body += (
                    f'<div class="row"><span><b>DataFlow diagnostic</b> {fmt_score(b.get("dataflow_score"))}</span>'
                    f'<span><b>DataFlow metric</b> {_esc(b.get("dataflow_metric") or "—")}</span>'
                    f'<span><b>DataFlow valid</b> {b.get("dataflow_valid","?")}/{b.get("dataflow_total","?")}</span></div>'
                )
            if b.get("warning"):
                body += f'<div class="callout" style="border-left-color:var(--star)">{_esc(b["warning"])}</div>'
            prompt = b.get("prompt") or {}
            if prompt:
                body += (
                    f'<div class="row"><span><b>Prompt source</b> {_esc(prompt.get("prompt_source") or prompt.get("source") or "—")}</span>'
                    f'<span><b>Prompt id</b> {_esc(prompt.get("prompt_template_id") or prompt.get("template_id") or "—")}</span></div>'
                )
            if b.get("detail_path"):
                body += (
                    f'<div class="row"><b>逐样本看板</b> '
                    f'<a href="#bench" data-sample-bench="{_esc(b.get("bench_name") or "")}" '
                    f'data-report-bench-name="{_esc(b.get("bench_name") or "")}">'
                    f'打开该 bench 明细页</a></div>'
                )
                if b.get("primary_detail_path"):
                    body += f'<div class="row"><b>Primary step3</b> <span class="path">{_esc(b["primary_detail_path"])}</span></div>'
                body += f'<div class="row"><b>DataFlow step2</b> <span class="path">{_esc(b["detail_path"])}</span></div>'
        parts.append(
            f'<div class="detail"><h3>{_esc(b["bench_name"])} '
            f'<span class="tag">{_esc(b.get("eval_type") or "")}</span></h3>{body}</div>')
    return "".join(parts)


def build_settings(results: dict, overview: dict, args, paths: dict) -> str:
    mc = results.get("model_config") or {}
    rt = results.get("runtime") or {}
    run_id = results.get("run_id")
    gen_at = results.get("generated_at")

    # 被测模型 + 运行标识（让报告自包含、可追溯，杜绝「测了哪个模型」歧义）
    rows = [
        ('被测模型', _esc(str(mc.get("model_name_or_path") or overview.get("model") or "—"))),
        ('模型类型', 'API' if mc.get("is_api") else ('本地 vLLM' if mc.get("is_api") is False else '—')),
    ]
    if mc.get("api_provider"):
        rows.append(('API provider', _esc(str(mc["api_provider"]))))
    if run_id:
        rows.append(('run_id', _esc(str(run_id))))
    if gen_at:
        rows.append(('评测时间', _esc(str(gen_at))))

    # 生成参数真值（可复现的关键）—— 直接落 evalspec 里用过的值，缺失标 —
    def g(k):
        v = mc.get(k)
        return _esc(str(v)) if v is not None else '—'
    gen_line = (f'temperature={g("temperature")} · top_p={g("top_p")} · '
                f'max_tokens={g("max_tokens")} · seed={g("seed")}')
    rows.append(('生成参数', gen_line))

    sm = rt.get("smoke")
    ms = rt.get("max_samples")
    sampling = ('smoke（每 bench 3 条抽样）' if sm
                else (f'全量' if ms in (None, 0) else f'每 bench 截断 {ms} 条'))
    rows.append(('采样规模', f'{sampling}；各 bench 实际模式见上方「模式」列'))
    rows.append(('使用 metric', '主分来自 metric stage' + (' + 见「多维度 Metric」节' if paths.get('metrics') else '；未提供 metric_results 时仅展示 fallback')))
    rows.append(('eval_results', f'<span class="path">{_esc(paths["results"])}</span>'))
    if paths.get("metrics"):
        rows.append(('metric_results', f'<span class="path">{_esc(paths["metrics"])}</span>'))
    rows.append(('leaderboard 分数表', f'<span class="path">{_esc(paths["scores"])}</span>'))
    rows.append(('本报告', f'<span class="path">{_esc(paths["out"])}</span>'))

    body = "".join(f'<div class="kv"><b>{k}</b>{v}</div>' for k, v in rows)
    note = ('seed + temperature=0 用于可复现；部分 API 不保证 seed 严格生效。'
            if str(mc.get("temperature")) == "0.0" or mc.get("temperature") == 0
            else '当前 temperature 非 0，重复评测分数可能抖动。')
    return ('<h2>附：评测设置（供判断分数可比性 / 复现）</h2>'
            f'<div class="callout" style="border-left-color:var(--muted)">{note}'
            ' 参数取自本次 evalspec.yaml，已随结果落盘。</div>'
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


SAMPLE_JS = """
(function(){
  var dataEl = document.getElementById('sample-dashboard-data');
  var root = document.getElementById('sample-dashboard');
  var DATA = {};
  try { DATA = dataEl ? JSON.parse(dataEl.textContent || '{}') : {}; } catch(e) { DATA = {}; }
  var benches = DATA.benches || [];
  if (!benches.length) return;
  var state = {bench:0, query:'', errorOnly:false, hideEmpty:false, page:1, pageSize:50, selectedId:null};
  function q(id){ return document.getElementById(id); }
  function txt(v){
    if (v === null || v === undefined) return '';
    if (typeof v === 'object') {
      try { return JSON.stringify(v, null, 2); } catch(e) { return String(v); }
    }
    return String(v);
  }
  function esc(s){
    return String(s == null ? '' : s).replace(/[&<>"']/g, function(c){
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
    });
  }
  function empty(v){
    return v === null || v === undefined || (typeof v === 'string' && v.trim() === '') ||
      (Array.isArray(v) && v.length === 0);
  }
  function bad(row){
    var valid = row.primary_metric_valid ?? row.eval_valid;
    var score = Number(row.primary_score ?? row.primary_metric_score ?? row.eval_score);
    return valid === false || valid === 'false' || (!Number.isNaN(score) && score === 0) ||
      !!(row.primary_metric_error || row.eval_error || row.error || row.__parse_error);
  }
  function scoreText(v){ return typeof v === 'number' ? v.toFixed(4) : '—'; }
  function showPage(page){
    document.querySelectorAll('.report-page').forEach(function(el){
      el.classList.toggle('active', el.id === 'page-' + page);
    });
  }
  function setActiveNav(name){
    document.querySelectorAll('.report-nav-btn').forEach(function(btn){
      var isOverview = name === '__overview__' && btn.getAttribute('data-report-page') === 'overview';
      var isBench = btn.getAttribute('data-report-bench-name') === name;
      btn.classList.toggle('active', isOverview || isBench);
    });
  }
  function benchIndexByName(name){
    return benches.findIndex(function(b){ return b.bench_name === name; });
  }
  function pushHash(hash, push){
    if (!push) return;
    if (window.location.hash === hash) return;
    history.pushState(null, '', hash);
  }
  function showOverview(push){
    showPage('overview');
    setActiveNav('__overview__');
    pushHash('#overview', push);
  }
  function showBenchByName(name, push){
    var idx = benchIndexByName(name);
    if (idx < 0 || !root) return;
    state.bench = idx;
    state.page = 1;
    state.selectedId = null;
    showPage('bench');
    setActiveNav(name);
    render();
    pushHash('#bench/' + encodeURIComponent(name), push);
  }
  function filteredRows(bench){
    var rows = bench.rows || [];
    var query = state.query.trim().toLowerCase();
    if (state.errorOnly) rows = rows.filter(bad);
    if (query) rows = rows.filter(function(row){ return txt(row).toLowerCase().indexOf(query) >= 0; });
    return rows;
  }
  function visibleColumns(bench, rows){
    var cols = bench.columns || [];
    if (!state.hideEmpty) return cols;
    return cols.filter(function(col){ return rows.some(function(row){ return !empty(row[col]); }); });
  }
  function firstValue(row, keys){
    for (var i=0; i<keys.length; i++) {
      var v = row[keys[i]];
      if (!empty(v)) return v;
    }
    return '';
  }
  function rowId(row, fallback){
    return String(row.__sample_index ?? fallback ?? '');
  }
  function clip(value, limit){
    var s = txt(value).replace(/\s+/g, ' ').trim();
    if (!s) return '—';
    return s.length > limit ? s.slice(0, limit - 1) + '…' : s;
  }
  function caseTitle(row){
    return firstValue(row, ['question','problem','prompt','input','query','goal','claim','context','generated_ans']) || '—';
  }
  function caseMeta(row, bench){
    var items = [];
    items.push(bench.bench_name || '—');
    var ans = firstValue(row, ['answer','answers','label','target','reference','reference_answer']);
    if (!empty(ans)) items.push('gold ' + clip(ans, 28));
    if (!empty(row.primary_answer)) items.push('primary answer ' + clip(row.primary_answer, 28));
    var fieldCount = Object.keys(row).length;
    items.push(fieldCount + ' fields');
    return items;
  }
  function primaryScore(row){
    var raw = row.primary_score ?? row.primary_metric_score ?? row.eval_score;
    var score = Number(raw);
    return Number.isNaN(score) ? null : score;
  }
  function fieldType(value){
    if (Array.isArray(value)) return 'array';
    if (value === null) return 'null';
    return typeof value;
  }
  function renderPager(bench, rows){
    var cols = visibleColumns(bench, rows);
    var totalPages = Math.max(1, Math.ceil(rows.length / state.pageSize));
    state.page = Math.min(Math.max(1, state.page), totalPages);
    q('sample-pager-info').textContent = rows.length + ' 条匹配 / ' + bench.loaded_rows +
      ' 条已载入 · 第 ' + state.page + '/' + totalPages + ' 页 · ' + cols.length + ' 列';
    q('sample-prev').disabled = state.page <= 1;
    q('sample-next').disabled = state.page >= totalPages;
    return cols;
  }
  function currentPageRows(rows){
    var start = (state.page - 1) * state.pageSize;
    return rows.slice(start, start + state.pageSize);
  }
  function renderCaseList(bench, rows){
    var list = q('sample-case-list');
    var pageRows = currentPageRows(rows);
    if (pageRows.length && !pageRows.some(function(row, i){ return rowId(row, i) === state.selectedId; })) {
      state.selectedId = rowId(pageRows[0], 0);
    }
    if (!pageRows.length) state.selectedId = null;
    list.textContent = '';
    pageRows.forEach(function(row){
      var id = rowId(row, 0);
      var score = primaryScore(row);
      var card = document.createElement('button');
      card.type = 'button';
      card.className = 'case-card' + (id === state.selectedId ? ' active' : '');
      card.onclick = function(){ state.selectedId = id; render(); };
      var badgeClass = score === null ? '' : (score > 0 ? ' good' : ' bad');
      var badges = '<span class="case-badge' + badgeClass + '">score ' +
        (score === null ? '—' : scoreText(score)) + '</span>';
      if (!empty(row.primary_answer)) {
        badges += '<span class="case-badge">ans ' + esc(clip(row.primary_answer, 20)) + '</span>';
      }
      card.innerHTML =
        '<div class="case-card-top"><span class="case-index">#' + esc(row.__sample_index ?? '') + '</span>' +
        '<span class="case-badges">' + badges + '</span></div>' +
        '<div class="case-question">' + esc(clip(caseTitle(row), 180)) + '</div>' +
        '<div class="case-meta">' + caseMeta(row, bench).map(function(x){ return '<span>' + esc(x) + '</span>'; }).join('') + '</div>';
      list.appendChild(card);
    });
  }
  function renderCaseDetail(bench, rows, cols){
    var detail = q('sample-case-detail');
    var row = rows.find(function(r, i){ return rowId(r, i) === state.selectedId; });
    if (!row) {
      detail.innerHTML = '<div class="case-detail-head"><div class="case-detail-title">没有匹配的 case</div></div>';
      return;
    }
    var score = primaryScore(row);
    var meta = [
      'case #' + (row.__sample_index ?? '—'),
      'score ' + (score === null ? '—' : scoreText(score)),
      !empty(row.primary_answer) ? 'primary answer ' + clip(row.primary_answer, 60) : '',
      cols.length + ' fields'
    ].filter(Boolean);
    var fields = cols.map(function(col){
      var value = row[col];
      var cls = value === true || value === 'true' ? ' sample-true' :
        (value === false || value === 'false' ? ' sample-false' : '');
      return '<div class="case-field"><div class="case-field-key"><span>' + esc(col) +
        '</span><span class="case-field-type">' + esc(fieldType(value)) + '</span></div>' +
        '<div class="case-field-value' + cls + '">' + esc(empty(value) ? '—' : txt(value)) + '</div></div>';
    }).join('');
    detail.innerHTML =
      '<div class="case-detail-head"><div class="case-detail-title">' + esc(clip(caseTitle(row), 240)) + '</div>' +
      '<div class="case-detail-meta">' + meta.map(function(x){ return '<span>' + esc(x) + '</span>'; }).join('') +
      '</div></div><div class="case-fields">' + fields + '</div>';
  }
  function render(){
    var bench = benches[state.bench] || {rows:[], columns:[]};
    q('sample-current-title').textContent = bench.bench_name || '—';
    q('sample-current-meta').innerHTML =
      '<span>eval_type: <b>' + esc(bench.eval_type || '—') + '</b></span>' +
      '<span>mode: <b>' + esc(bench.mode || '—') + '</b></span>' +
      '<span>score: <b>' + scoreText(bench.score) + '</b></span>' +
      '<span>sample_source: <b>' + esc(bench.sample_path_kind || '—') + '</b></span>' +
      '<span>sample_path: <span class="path">' + esc(bench.sample_path || bench.detail_path || '—') + '</span></span>';
    var notes = [];
    if (!bench.sample_path && !bench.detail_path) notes.push('该 bench 没有逐样本明细路径，无法展示样本。');
    else if (!bench.file_exists) notes.push('逐样本明细文件不存在，可能产物被移动或尚未生成。');
    if (bench.load_error) notes.push('读取失败：' + bench.load_error);
    if (bench.truncated) notes.push('当前报告只嵌入前 ' + bench.loaded_rows + ' 条；命令行传入了 max rows 限制。');
    if (bench.parse_errors && bench.parse_errors.length) notes.push('JSONL 解析错误 ' + bench.parse_errors.length + ' 条。');
    var notice = q('sample-notice');
    notice.style.display = notes.length ? 'block' : 'none';
    notice.textContent = notes.join(' ');
    var rows = filteredRows(bench);
    var cols = renderPager(bench, rows);
    renderCaseList(bench, rows);
    renderCaseDetail(bench, rows, cols);
  }
  function chooseBench(name){
    showBenchByName(name, true);
  }
  var search = q('sample-search');
  var errorOnly = q('sample-error-only');
  var hideEmpty = q('sample-hide-empty');
  var pageSize = q('sample-page-size');
  var prev = q('sample-prev');
  var next = q('sample-next');
  var exportBtn = q('sample-export');
  if (search) search.addEventListener('input', function(e){ state.query = e.target.value; state.page = 1; state.selectedId = null; render(); });
  if (errorOnly) errorOnly.addEventListener('change', function(e){ state.errorOnly = e.target.checked; state.page = 1; state.selectedId = null; render(); });
  if (hideEmpty) hideEmpty.addEventListener('change', function(e){ state.hideEmpty = e.target.checked; state.selectedId = null; render(); });
  if (pageSize) pageSize.addEventListener('change', function(e){ state.pageSize = Number(e.target.value); state.page = 1; state.selectedId = null; render(); });
  if (prev) prev.addEventListener('click', function(){ state.page -= 1; state.selectedId = null; render(); });
  if (next) next.addEventListener('click', function(){ state.page += 1; state.selectedId = null; render(); });
  if (exportBtn) exportBtn.addEventListener('click', function(){
    var bench = benches[state.bench];
    var rows = filteredRows(bench);
    var blob = new Blob([rows.map(function(r){ return JSON.stringify(r); }).join('\\n') + '\\n'], {type:'application/jsonl'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (bench.bench_name || 'bench') + '_filtered.jsonl';
    a.click();
    URL.revokeObjectURL(a.href);
  });
  document.querySelectorAll('.report-nav-btn[data-report-page="overview"]').forEach(function(btn){
    btn.addEventListener('click', function(){ showOverview(true); });
  });
  document.querySelectorAll('.report-nav-btn[data-report-bench-name]').forEach(function(btn){
    btn.addEventListener('click', function(){ showBenchByName(btn.getAttribute('data-report-bench-name'), true); });
  });
  document.querySelectorAll('[data-sample-bench]').forEach(function(link){
    link.addEventListener('click', function(e){
      e.preventDefault();
      chooseBench(link.getAttribute('data-sample-bench'));
    });
  });
  function applyHash(){
    var hash = window.location.hash || '';
    if (hash.indexOf('#bench/') === 0) {
      showBenchByName(decodeURIComponent(hash.slice(7)), false);
    } else {
      showOverview(false);
    }
  }
  window.addEventListener('hashchange', applyHash);
  window.addEventListener('popstate', applyHash);
  applyHash();
})();
"""


def build_sample_dashboard_section(sample_data: dict | None) -> str:
    if not sample_data or not sample_data.get("benches"):
        return ""
    bench_count = len(sample_data.get("benches") or [])
    row_count = sum(int(b.get("loaded_rows") or 0) for b in sample_data.get("benches") or [])
    data_script = (
        '<script id="sample-dashboard-data" type="application/json">'
        f'{_json_for_script(sample_data)}</script>'
    )
    return (
        f'{data_script}'
        '<section id="page-bench" class="report-page">'
        '<div class="wide-wrap">'
        '<h1 id="sample-dashboard">Bench 明细看板</h1>'
        '<div class="lb-hint">优先内嵌 primary metric step3 JSONL；缺失时回退到 DataFlow step2。'
        '左侧切换不同 bench；当前按每个 bench 的字段全集展示，后续可把关键字段白名单沉到 gallery 后再做精简视图。</div>'
        '<div class="sample-dashboard">'
        '<div class="sample-top">'
        f'<div class="sample-title">样本明细 <span class="src">（{bench_count} benches · {row_count} rows）</span></div>'
        '<div id="sample-current-meta" class="sample-meta"></div>'
        '<div class="sample-toolbar">'
        '<input id="sample-search" placeholder="搜索当前 bench 的所有字段">'
        '<label><input id="sample-error-only" type="checkbox"> 只看 primary score 0</label>'
        '<label><input id="sample-hide-empty" type="checkbox"> 隐藏全空列</label>'
        '<select id="sample-page-size">'
        '<option value="50" selected>50 / 页</option><option value="100">100 / 页</option>'
        '<option value="250">250 / 页</option><option value="500">500 / 页</option>'
        '</select>'
        '<button id="sample-export">导出当前筛选 JSONL</button>'
        '</div></div>'
        '<div class="sample-body">'
        '<div class="sample-main">'
        '<h2 id="sample-current-title" style="margin-top:0">—</h2>'
        '<div id="sample-notice" class="sample-notice" style="display:none"></div>'
        '<div class="sample-pager"><div id="sample-pager-info"></div>'
        '<div class="sample-page-buttons"><button id="sample-prev">上一页</button>'
        '<button id="sample-next">下一页</button></div></div>'
        '<div class="case-layout">'
        '<div class="case-list-pane"><div id="sample-case-list" class="case-list"></div></div>'
        '<div id="sample-case-detail" class="case-detail-pane"></div>'
        '</div>'
        '</div></div></div>'
        '</div></section>'
    )


def render_html(results: dict, metrics: dict, scores: dict, args, paths: dict, sample_data: dict | None = None) -> str:
    overview = extract_overview(results, metrics)
    mx = extract_metrics(metrics)
    lb = build_leaderboard_data(overview, scores)

    overview_page = ('<section id="page-overview" class="report-page active"><div class="wrap">'
            + build_overview(overview, lb)
            + build_leaderboard(lb)
            + build_metric_section(mx)
            + build_details(overview, paths)
            + build_settings(results, overview, args, paths)
            + '<footer>由 One-Eval skill 自动生成 · '
            f'<a href="{REPO_URL}" target="_blank" rel="noopener">github.com/OpenDCAI/One-Eval</a>'
            ' · 排名仅供粗略定位，公开分以各自来源为准</footer></div></section>')
    body = (
        '<div class="report-app">'
        + build_sidebar(overview, sample_data)
        + '<main class="report-main">'
        + build_header(overview["model"])
        + overview_page
        + build_sample_dashboard_section(sample_data)
        + '</main></div>'
    )
    return (f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>One-Eval 评测报告 · {_esc(overview["model"])}</title>'
            f'<style>{CSS}</style></head><body>{body}'
            f'<script>{JS}</script><script>{SAMPLE_JS}</script></body></html>')


def main(argv=None):
    p = argparse.ArgumentParser(description="生成自包含 HTML 评测报告")
    p.add_argument("--results", required=True, help="eval_results.json 路径")
    p.add_argument("--metrics", help="metric_results.json 路径（可选）")
    p.add_argument("--scores", default=str(DEFAULT_SCORES), help="公开分数表 json 路径")
    p.add_argument("--out", help="输出 html 路径（默认 eval_outputs/report.html）")
    p.add_argument("--sample-dashboard-max-rows", type=int, help="逐样本看板每个 bench 最多嵌入多少行；默认全量")
    p.add_argument("--no-sample-dashboard", action="store_true", help="不在报告内嵌逐样本明细看板")
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

    sample_data = {}
    if not args.no_sample_dashboard:
        try:
            import render_sample_dashboard
            sample_data = render_sample_dashboard.build_dashboard_data(
                args.results,
                max_rows_per_bench=args.sample_dashboard_max_rows,
                metrics=metrics,
            )
            rows = sum(int(b.get("loaded_rows") or 0) for b in sample_data.get("benches", []))
            print(f"✓ 已内嵌样本明细看板数据: benches={len(sample_data.get('benches', []))} rows={rows}")
        except Exception as e:
            print(f"⚠ 样本明细看板数据读取失败: {type(e).__name__}: {e}", file=sys.stderr)

    paths = {
        "results": str(Path(args.results).absolute()),
        "metrics": str(Path(args.metrics).absolute()) if args.metrics and Path(args.metrics).exists() else "",
        "scores": str(Path(args.scores).absolute()),
        "out": str(out.absolute()),
    }
    html_text = render_html(results, metrics, scores, args, paths, sample_data=sample_data)
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
