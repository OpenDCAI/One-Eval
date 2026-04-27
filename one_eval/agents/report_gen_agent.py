from __future__ import annotations

import json
import os
import time
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState, BenchInfo
from one_eval.logger import get_logger

log = get_logger("ReportGenAgent")

EVAL_TYPE_LABELS_ZH = [
    ("knowledge", "知识与百科"),
    ("reasoning", "逻辑推理"),
    ("mathematics", "数学能力"),
    ("coding", "代码能力"),
    ("language", "理解与生成"),
    ("instruction", "指令遵循"),
    ("safety", "安全与对齐"),
]

EVAL_TYPE_LABELS_EN = [
    ("knowledge", "Knowledge"),
    ("reasoning", "Reasoning"),
    ("mathematics", "Mathematics"),
    ("coding", "Coding"),
    ("language", "Language"),
    ("instruction", "Instruction"),
    ("safety", "Safety"),
]


# 关键词匹配规则（作为 metadata 缺失时的兜底）
# 格式: {关键词: [维度列表]}
# 注意：匹配时会检查 bench_name 是否包含该关键词（不区分大小写）
BENCH_KEYWORD_RULES = {
    # 特殊数据集 (Known Special Cases)
    "gsm8k": ["mathematics", "reasoning"],
    "math": ["mathematics"],
    "humaneval": ["coding"],
    "mbpp": ["coding"],
    "mmlu": ["knowledge"],
    "ceval": ["knowledge"],
    "cmmlu": ["knowledge"],
    "bbh": ["reasoning"],
    "arc": ["reasoning"],
    "ifeval": ["instruction"],
    "alpaca": ["instruction"],

    # 通用关键词 (General Keywords)
    "code": ["coding"],
    "program": ["coding"],
    "sql": ["coding"],
    "python": ["coding"],
    "reason": ["reasoning"],
    "logic": ["reasoning"],
    "arithmetic": ["mathematics"],
    "algebra": ["mathematics"],
    "geometry": ["mathematics"],
    "calculus": ["mathematics"],
    "qa": ["knowledge"],
    "exam": ["knowledge"],
    "knowledge": ["knowledge"],
    "safe": ["safety"],
    "align": ["safety"],
    "harm": ["safety"],
    "bias": ["safety"],
    "instruct": ["instruction"],
    "chat": ["language"],
    "dialog": ["language"],
    "summary": ["language"],
    "translat": ["language"],
    "nli": ["language"],
    "reading": ["language"],
    "comprehension": ["language"],
}

DOMAIN_KEYWORD_RULES = {
    "finance": ["finance", "financial", "finqa", "fin-bench", "finbench", "convfinqa", "financeqa", "stock", "bank", "fiscal"],
    "medical": ["medical", "med", "medicine", "clinical", "health", "biomed", "usmle"],
    "legal": ["legal", "law", "juris", "court", "contract", "statute"],
    "math": ["math", "mathematics", "algebra", "geometry", "calculus", "arithmetic", "theorem"],
    "coding": ["code", "coding", "program", "software", "humaneval", "mbpp", "leetcode"],
    "reasoning": ["reason", "logic", "deduction", "inference"],
    "safety": ["safety", "harm", "bias", "alignment", "jailbreak"],
    "knowledge": ["knowledge", "qa", "exam", "mmlu", "ceval", "cmmlu"],
}


class ReportGenAgent(CustomAgent):
    @property
    def role_name(self) -> str:
        return "ReportGenAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return ""

    @property
    def task_prompt_template_name(self) -> str:
        return ""

    async def run(self, state: NodeState) -> NodeState:
        benches: List[BenchInfo] = getattr(state, "benches", []) or []
        eval_results: Dict[str, Any] = getattr(state, "eval_results", {}) or {}
        metric_plan: Dict[str, Any] = getattr(state, "metric_plan", {}) or {}

        if not benches or not eval_results:
            log.warning("缺少 benches 或 eval_results，跳过报告生成")
            return state

        bench_summaries = self._build_bench_summaries(benches, eval_results, metric_plan)
        bench_profile = self._build_bench_profile_view(bench_summaries)
        bench_rows = bench_profile.get("rows", [])
        overall_score = self._compute_overall_score(bench_rows)
        domain_performance = self._build_domain_performance_view(bench_rows)

        lang = self._get_lang(state)
        macro_view = self._build_macro_view(bench_summaries, eval_results, lang)
        diagnostic_view = self._build_diagnostic_view(benches, eval_results, metric_plan, lang)
        analyst_view = self._collect_analyst_outputs(benches, eval_results)
        analyst_compact = self._compact_analyst_view(analyst_view)

        summary_payload = {
            "overall_score": overall_score,
            "benches": bench_rows[:20],
            "benchmark_profiles": bench_rows[:20],
            "domain_performance": domain_performance.get("rows", [])[:10],
            "error_distribution": diagnostic_view.get("error_distribution", []),
            "metric_summaries": analyst_compact.get("metric_summary", []),
            "case_studies": analyst_compact.get("case_study", []),
        }
        llm_summary = await self._generate_llm_summary(summary_payload, state)

        report = {
            "version": "v1.1",
            "generated_at": time.time(),
            "model": self._get_model_name(state),
            "overall": {
                "score": overall_score,
                "bench_summaries": bench_summaries,
                "num_benches": len(bench_rows),
                "num_samples": sum(int(r.get("num_samples", 0) or 0) for r in bench_rows),
            },
            "bench_results": {"rows": bench_rows},
            "benchmark_profiles": bench_profile,
            "domain_performance": domain_performance,
            "macro": macro_view,
            "diagnostic": {
                "error_distribution": diagnostic_view.get("error_distribution", []),
                "length_histogram": diagnostic_view.get("length_histogram", {}),
            },
            "cases": {
                "columns": ["bench", "question", "model_output", "ground_truth", "error_type", "score"],
                "rows": diagnostic_view.get("cases", []),
            },
            "analyst": analyst_view,
            "llm_summary": llm_summary,
        }

        if not getattr(state, "reports", None):
            state.reports = {}
        state.reports["default"] = report

        log.info(f"[ReportGen] report={json.dumps(report, ensure_ascii=False)}")

        if not getattr(state, "result", None):
            state.result = {}
        state.result[self.role_name] = {"report_key": "default"}

        return state

    def _get_model_name(self, state: NodeState) -> str:
        if getattr(state, "target_model_name", None):
            return state.target_model_name
        if getattr(state, "target_model", None) and getattr(state.target_model, "model_name_or_path", None):
            return state.target_model.model_name_or_path
        if getattr(state, "model_type", None):
            return state.model_type
        return "unknown_model"

    def _build_bench_summaries(
        self,
        benches: List[BenchInfo],
        eval_results: Dict[str, Any],
        metric_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        summaries = []
        for bench in benches:
            bench_name = bench.bench_name
            bench_result = eval_results.get(bench_name) or {}
            metrics = bench_result.get("metrics", {}) or {}
            bench_meta = bench.meta or {}
            plan = metric_plan.get(bench_name, []) or []
            primary_name = self._get_primary_metric_name(plan, metrics) or "accuracy"
            bench_score, score_source = self._resolve_report_score(bench, bench_result, primary_name)

            num_samples = int(
                bench_result.get("num_samples")
                or (bench_meta.get("eval_result") or {}).get("valid_samples")
                or (bench_meta.get("eval_result") or {}).get("total_samples")
                or 0
            )
            valid_samples = int((bench_meta.get("eval_result") or {}).get("valid_samples") or 0)
            total_samples = int((bench_meta.get("eval_result") or {}).get("total_samples") or num_samples or 0)

            meta_dims = bench_meta.get("capabilities") or bench_meta.get("capability_dims") or bench_meta.get("radar_dimensions")
            if isinstance(meta_dims, str):
                meta_dims = [meta_dims]
            if not isinstance(meta_dims, list) or not meta_dims:
                meta_dims = self._map_bench_to_dimensions(bench_name, list(metrics.keys()))
            meta_dims = [str(d).strip() for d in meta_dims if isinstance(d, str) and str(d).strip()] or ["knowledge"]

            domain_tags = self._infer_domain_tags(bench_name, bench_meta, meta_dims)
            primary_domain = domain_tags[0] if domain_tags else "general"
            description = bench_meta.get("description_zh") or bench_meta.get("description") or ""
            warnings = []
            abnormality = bench_meta.get("eval_abnormality")
            if isinstance(abnormality, dict) and abnormality.get("is_abnormal"):
                warnings.append(abnormality)

            for dim in meta_dims:
                summaries.append({
                    "bench": bench_name,
                    "eval_type": dim,
                    "capability": dim,
                    "domain": primary_domain,
                    "domains": domain_tags,
                    "domain_tags": domain_tags,
                    "task_type": bench_meta.get("task_type"),
                    "eval_dataflow_type": getattr(bench, "bench_dataflow_eval_type", None),
                    "bench_category": bench_meta.get("category"),
                    "bench_description": description,
                    "num_samples": num_samples,
                    "valid_samples": valid_samples,
                    "total_samples": total_samples,
                    "primary_metric": primary_name,
                    "primary_score": bench_score,
                    "score_source": score_source,
                    "warnings": warnings,
                })
        return summaries

    def _resolve_report_score(self, bench: BenchInfo, bench_result: Dict[str, Any], primary_name: Optional[str]) -> Tuple[float, str]:
        # Report-level scores must reflect DataFlow Eval's general accuracy, not downstream metric scores.
        meta_eval = ((bench.meta or {}).get("eval_result") or {})
        for key in ["accuracy", "score"]:
            if meta_eval.get(key) is not None:
                return self._safe_float(meta_eval.get(key)), f"dataflow_eval:{key}"
        return 0.0, "dataflow_eval:missing"

    def _build_bench_profile_view(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        bench_index: Dict[str, Dict[str, Any]] = {}
        for s in summaries:
            bench_name = str(s.get("bench") or "")
            if not bench_name:
                continue
            score = self._safe_float(s.get("primary_score"))
            sample = int(s.get("num_samples", 0) or 0)

            if bench_name not in bench_index:
                bench_index[bench_name] = {
                    "bench": bench_name,
                    "domain": s.get("domain") or "general",
                    "domains": s.get("domains") or s.get("domain_tags") or [],
                    "domain_tags": s.get("domain_tags") or [],
                    "capabilities": [s.get("capability") or s.get("eval_type")],
                    "task_type": s.get("task_type"),
                    "eval_type": s.get("eval_dataflow_type"),
                    "category": s.get("bench_category"),
                    "description": s.get("bench_description") or "",
                    "num_samples": sample,
                    "valid_samples": int(s.get("valid_samples", 0) or 0),
                    "total_samples": int(s.get("total_samples", 0) or 0),
                    "primary_metric": s.get("primary_metric") or "accuracy",
                    "primary_score": score,
                    "score_source": s.get("score_source") or "unknown",
                    "warnings": s.get("warnings") or [],
                }
                continue

            row = bench_index[bench_name]
            row["num_samples"] = max(int(row.get("num_samples", 0) or 0), sample)
            row["valid_samples"] = max(int(row.get("valid_samples", 0) or 0), int(s.get("valid_samples", 0) or 0))
            row["total_samples"] = max(int(row.get("total_samples", 0) or 0), int(s.get("total_samples", 0) or 0))
            if score > self._safe_float(row.get("primary_score")):
                row["primary_score"] = score
                row["primary_metric"] = s.get("primary_metric") or row.get("primary_metric")
                row["score_source"] = s.get("score_source") or row.get("score_source")
            merged_tags = list(dict.fromkeys((row.get("domain_tags") or []) + (s.get("domain_tags") or [])))
            row["domain_tags"] = merged_tags
            row["domains"] = list(dict.fromkeys((row.get("domains") or []) + (s.get("domains") or [])))
            caps = [c for c in (row.get("capabilities") or []) if c]
            cap = s.get("capability") or s.get("eval_type")
            if cap:
                caps.append(cap)
            row["capabilities"] = list(dict.fromkeys(caps))
            if row.get("domain") == "general" and s.get("domain"):
                row["domain"] = s.get("domain")
            if not row.get("description") and s.get("bench_description"):
                row["description"] = s.get("bench_description")

        rows = sorted(
            bench_index.values(),
            key=lambda x: (self._safe_float(x.get("primary_score")), int(x.get("num_samples", 0) or 0)),
            reverse=True,
        )
        return {"rows": rows}

    def _build_domain_performance_view(self, bench_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        groups: Dict[str, Dict[str, Any]] = {}
        for row in bench_rows:
            domains = row.get("domains") or row.get("domain_tags") or [row.get("domain") or "general"]
            if isinstance(domains, str):
                domains = [domains]
            domains = [str(d).strip().lower() for d in domains if str(d).strip()] or ["general"]
            score = self._safe_float(row.get("primary_score"))
            num = int(row.get("num_samples", 0) or 0)
            bench_name = str(row.get("bench") or "")
            weight = num if num > 0 else 1
            for domain in domains:
                if domain not in groups:
                    groups[domain] = {
                        "score_sum": 0.0,
                        "weight": 0,
                        "benches": {},
                    }
                groups[domain]["score_sum"] += score * weight
                groups[domain]["weight"] += weight
                if bench_name:
                    groups[domain]["benches"][bench_name] = score

        rows: List[Dict[str, Any]] = []
        for domain, item in groups.items():
            benches_map = item.get("benches", {})
            benches_sorted: List[Tuple[str, float]] = sorted(
                benches_map.items(),
                key=lambda x: self._safe_float(x[1]),
                reverse=True,
            )
            avg_score = item["score_sum"] / item["weight"] if item["weight"] > 0 else 0.0
            best_bench = benches_sorted[0][0] if benches_sorted else None
            worst_bench = benches_sorted[-1][0] if benches_sorted else None
            rows.append({
                "domain": domain,
                "avg_score": avg_score,
                "score": avg_score,
                "num_samples": int(item.get("weight", 0) or 0),
                "bench_count": len(benches_sorted),
                "benches": [b[0] for b in benches_sorted],
                "best_bench": best_bench,
                "worst_bench": worst_bench,
            })

        rows = sorted(
            rows,
            key=lambda x: (self._safe_float(x.get("avg_score")), int(x.get("num_samples", 0) or 0)),
            reverse=True,
        )
        return {"rows": rows}

    def _map_bench_to_dimensions(self, bench_name: str, metric_names: List[str]) -> List[str]:
        # 1. 关键词规则匹配（名称中包含关键词即可）
        name_lower = bench_name.lower()
        
        # 优先匹配长关键词（例如优先匹配 "gsm8k" 而不是 "math"）
        sorted_keys = sorted(BENCH_KEYWORD_RULES.keys(), key=len, reverse=True)
        
        for key in sorted_keys:
            if key in name_lower:
                return BENCH_KEYWORD_RULES[key]
        
        # 2. 如果没有匹配，尝试根据 Metric 名称推断
        for m in metric_names:
            m_lower = m.lower()
            if "rouge" in m_lower or "bleu" in m_lower:
                return ["language"]
            if "pass@k" in m_lower:
                return ["coding"]
        
        # 3. 默认归类
        return ["knowledge"]

    def _compute_overall_score(self, summaries: List[Dict[str, Any]]) -> float:
        total = 0.0
        weight = 0
        for s in summaries:
            num = int(s.get("num_samples", 0) or 0)
            score = self._safe_float(s.get("primary_score"))
            if num > 0:
                total += score * num
                weight += num
        if weight > 0:
            return total / weight
        if summaries:
            return sum(self._safe_float(s.get("primary_score")) for s in summaries) / len(summaries)
        return 0.0

    def _build_macro_view(self, summaries: List[Dict[str, Any]], eval_results: Dict[str, Any], lang: str) -> Dict[str, Any]:
        radar = self._build_radar(summaries, lang)
        sunburst = self._build_sunburst(summaries)
        table = self._build_macro_table(summaries, eval_results)
        return {"radar": radar, "sunburst": sunburst, "table": table}

    def _build_radar(self, summaries: List[Dict[str, Any]], lang: str) -> Dict[str, Any]:
        labels_cfg = EVAL_TYPE_LABELS_EN if str(lang).lower().startswith("en") else EVAL_TYPE_LABELS_ZH
        accum = {k: {"score": 0.0, "weight": 0} for k, _ in labels_cfg}
        for s in summaries:
            eval_type = s.get("eval_type")
            if eval_type not in accum:
                continue
            num = int(s.get("num_samples", 0) or 0)
            score = self._safe_float(s.get("primary_score"))
            if num <= 0:
                num = 1
            accum[eval_type]["score"] += score * num
            accum[eval_type]["weight"] += num
        labels = [label for _, label in labels_cfg]
        scores = []
        for key, _ in labels_cfg:
            if accum[key]["weight"] > 0:
                scores.append(accum[key]["score"] / accum[key]["weight"])
            else:
                scores.append(0.0)
        return {"labels": labels, "scores": scores}

    def _get_lang(self, state: NodeState) -> str:
        req = getattr(state, "request", None)
        if isinstance(req, dict):
            return str(req.get("language") or "zh")
        if req is not None:
            return str(getattr(req, "language", "zh") or "zh")
        return "zh"

    def _build_sunburst(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        rows = []
        for s in summaries:
            bench = s.get("bench")
            num = int(s.get("num_samples", 0) or 0)
            score = self._safe_float(s.get("primary_score"))
            path = self._split_bench_path(bench, s.get("domain"))
            rows.append({"path": path, "value": num, "score": score})
        return {"rows": rows}

    def _build_macro_table(self, summaries: List[Dict[str, Any]], eval_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        for s in summaries:
            bench = s.get("bench")
            bench_result = eval_results.get(bench) or {}
            metrics = bench_result.get("metrics", {}) or {}
            num_samples = int(bench_result.get("num_samples", 0) or 0)
            for name, m in metrics.items():
                rows.append({
                    "bench": bench,
                    "metric": name,
                    "score": self._safe_float(m.get("score")),
                    "priority": m.get("priority"),
                    "num_samples": num_samples,
                })
        return rows

    def _build_diagnostic_view(
        self,
        benches: List[BenchInfo],
        eval_results: Dict[str, Any],
        metric_plan: Dict[str, Any],
        lang: str = "zh",
    ) -> Dict[str, Any]:
        error_counter = Counter()
        correct_lengths = []
        incorrect_lengths = []
        cases = []

        for bench in benches:
            bench_name = bench.bench_name
            bench_result = eval_results.get(bench_name) or {}
            metrics = bench_result.get("metrics", {}) or {}
            plan = metric_plan.get(bench_name, []) or []
            primary_name = self._get_primary_metric_name(plan, metrics)
            primary_details = self._get_details(metrics, primary_name)

            records_path = bench.meta.get("eval_step3_path") or bench.meta.get("eval_detail_path")
            records = self._load_records(records_path)
            if not records:
                continue

            extraction_details = self._get_details(metrics, "extraction_rate")
            missing_details = self._get_details(metrics, "missing_answer_rate")
            format_details = self._get_details(metrics, "format_compliance_score")

            has_record_metrics = any(
                isinstance(r, dict) and isinstance(r.get("metric_details"), dict)
                for r in records[:5]
            )
            if not primary_details and not has_record_metrics:
                continue

            max_len = len(records)
            if primary_details:
                max_len = min(max_len, len(primary_details))

            for idx in range(max_len):
                rec = records[idx]
                record_metrics = rec.get("metric_details") if isinstance(rec, dict) else None

                primary_value = None
                if isinstance(record_metrics, dict) and primary_name in record_metrics:
                    primary_value = self._get_metric_value(record_metrics.get(primary_name))
                if primary_value is None and primary_details and idx < len(primary_details):
                    primary_value = primary_details[idx]

                score = self._safe_float(primary_value)
                is_correct = score >= 0.5
                pred = self._get_pred(rec, bench.meta.get("pred_key"))
                ref = self._get_ref(rec, bench.meta.get("ref_key"))
                question = self._get_question(rec, self._get_question_key_hint(bench.meta))

                pred_len = len(str(pred)) if pred is not None else 0
                if is_correct:
                    correct_lengths.append(pred_len)
                else:
                    incorrect_lengths.append(pred_len)

                extraction_value = None
                missing_value = None
                format_value = None
                if isinstance(record_metrics, dict):
                    extraction_value = self._get_metric_value(record_metrics.get("extraction_rate"))
                    missing_value = self._get_metric_value(record_metrics.get("missing_answer_rate"))
                    format_value = self._get_metric_value(record_metrics.get("format_compliance_score"))

                if extraction_value is None and extraction_details and idx < len(extraction_details):
                    extraction_value = extraction_details[idx]
                if missing_value is None and missing_details and idx < len(missing_details):
                    missing_value = missing_details[idx]
                if format_value is None and format_details and idx < len(format_details):
                    format_value = format_details[idx]

                error_info = self._classify_error(
                    is_correct,
                    self._safe_float(extraction_value) if extraction_value is not None else None,
                    self._safe_float(missing_value) if missing_value is not None else None,
                    self._safe_float(format_value) if format_value is not None else None,
                    lang=lang,
                )

                if not is_correct:
                    error_id = error_info.get("id", "incorrect")
                    error_counter[error_id] += 1
                    if len(cases) < 5:
                        cases.append({
                            "bench": bench_name,
                            "question": question,
                            "model_output": pred,
                            "ground_truth": ref,
                            "error_type": error_info.get("label"),
                            "error_id": error_id,
                            "score": score
                        })

        error_distribution = []
        total_err = sum(error_counter.values())
        for k, v in error_counter.most_common():
            ratio = (v / total_err) if total_err > 0 else 0.0
            error_distribution.append({
                "id": k,
                "type": self._error_label(k, lang),
                "label": self._error_label(k, lang),
                "count": v,
                "ratio": ratio,
            })

        length_histogram = self._build_length_hist(correct_lengths, incorrect_lengths)

        return {
            "error_distribution": error_distribution,
            "length_histogram": length_histogram,
            "cases": cases,
        }

    def _collect_analyst_outputs(
        self,
        benches: List[BenchInfo],
        eval_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        metric_summary = {}
        case_study = {}
        for bench in benches:
            bench_name = bench.bench_name
            bench_result = eval_results.get(bench_name) or {}
            metrics = bench_result.get("metrics", {}) or {}
            summary_metric = metrics.get("metric_summary_analyst", {}) or {}
            case_metric = metrics.get("case_study_analyst", {}) or {}
            summary = summary_metric.get("summary")
            analysis = case_metric.get("analysis")
            if isinstance(summary, str) and summary.strip():
                metric_summary[bench_name] = summary
            elif isinstance(summary_metric.get("error"), str) and summary_metric.get("error"):
                metric_summary[bench_name] = f"Metric summary unavailable: {summary_metric.get('error')}"
            if isinstance(analysis, str) and analysis.strip():
                case_study[bench_name] = analysis
            elif isinstance(case_metric.get("error"), str) and case_metric.get("error"):
                case_study[bench_name] = f"Case study unavailable: {case_metric.get('error')}"
        return {
            "metric_summary": metric_summary,
            "case_study": case_study,
        }

    def _compact_analyst_view(self, analyst_view: Dict[str, Dict[str, str]], limit: int = 5, max_len: int = 800) -> Dict[str, List[Dict[str, str]]]:
        metric_summary = []
        case_study = []
        for bench, text in list(analyst_view.get("metric_summary", {}).items())[:limit]:
            metric_summary.append({"bench": bench, "text": self._truncate_text(text, max_len)})
        for bench, text in list(analyst_view.get("case_study", {}).items())[:limit]:
            case_study.append({"bench": bench, "text": self._truncate_text(text, max_len)})
        return {"metric_summary": metric_summary, "case_study": case_study}

    def _truncate_text(self, text: str, max_len: int) -> str:
        if not isinstance(text, str):
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _get_primary_metric_name(self, plan: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Optional[str]:
        for m in plan:
            if m.get("priority") == "primary" and m.get("name") in metrics:
                return m.get("name")
        for m in plan:
            if m.get("name") in metrics:
                return m.get("name")
        if metrics:
            return next(iter(metrics.keys()))
        return None

    def _get_details(self, metrics: Dict[str, Any], name: Optional[str]) -> Optional[List[Any]]:
        if not name or name not in metrics:
            return None
        details = metrics[name].get("details")
        if isinstance(details, list) and len(details) > 0:
            return details
        return None

    def _get_metric_value(self, metric_value: Any) -> Optional[float]:
        if metric_value is None:
            return None
        if isinstance(metric_value, dict):
            if "score" in metric_value:
                return self._safe_float(metric_value.get("score"))
            return None
        return self._safe_float(metric_value)

    def _classify_error(
        self,
        is_correct: bool,
        extraction_value: Optional[float],
        missing_value: Optional[float],
        format_value: Optional[float],
        lang: str = "zh",
    ) -> Dict[str, str]:
        if is_correct:
            error_id = "correct"
        elif extraction_value is not None and extraction_value <= 0:
            error_id = "extraction_error"
        elif missing_value is not None and missing_value >= 1:
            error_id = "refusal_empty"
        elif format_value is not None and format_value < 0.5:
            error_id = "format_error"
        else:
            error_id = "incorrect_reasoning"
        return {"id": error_id, "label": self._error_label(error_id, lang)}

    def _error_label(self, error_id: str, lang: str = "zh") -> str:
        is_en = str(lang).lower().startswith("en")
        labels = {
            "correct": ("正确", "Correct"),
            "extraction_error": ("抽取错误", "Extraction Error"),
            "refusal_empty": ("拒答 / 空输出", "Refusal / Empty Response"),
            "format_error": ("格式错误", "Format Error"),
            "incorrect_reasoning": ("答案错误（推理/逻辑）", "Incorrect Answer (Reasoning/Logic)"),
        }
        zh, en = labels.get(error_id, labels["incorrect_reasoning"])
        return en if is_en else zh

    def _build_length_hist(self, correct: List[int], incorrect: List[int], bins: int = 10) -> Dict[str, Any]:
        max_len = max(correct + incorrect) if (correct or incorrect) else 0
        if max_len <= 0:
            return {"bins": [0], "correct": [0], "incorrect": [0]}
        bin_size = max(1, math.ceil(max_len / bins))
        edges = [(i + 1) * bin_size for i in range(bins)]
        corr_counts = [0] * bins
        err_counts = [0] * bins
        for l in correct:
            idx = min(l // bin_size, bins - 1)
            corr_counts[idx] += 1
        for l in incorrect:
            idx = min(l // bin_size, bins - 1)
            err_counts[idx] += 1
        return {"bins": edges, "correct": corr_counts, "incorrect": err_counts}

    def _split_bench_path(self, bench: str, domain: Optional[str]) -> List[str]:
        if isinstance(bench, str) and "/" in bench:
            return [p for p in bench.split("/") if p]
        if isinstance(bench, str) and "__" in bench:
            return [p for p in bench.split("__") if p]
        if domain:
            return [str(domain), bench]
        return [bench]

    def _infer_domain_tags(self, bench_name: str, bench_meta: Dict[str, Any], eval_dims: List[str]) -> List[str]:
        tags = bench_meta.get("tags")
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = []

        explicit_domains = bench_meta.get("domains") or bench_meta.get("domain")
        explicit_domain_values: List[str] = []
        raw_candidates: List[str] = []
        if isinstance(explicit_domains, str):
            explicit_domain_values.append(explicit_domains)
            raw_candidates.append(explicit_domains)
        elif isinstance(explicit_domains, list):
            explicit_domain_values.extend([d for d in explicit_domains if isinstance(d, str)])
            raw_candidates.extend(explicit_domain_values)

        task_type = bench_meta.get("task_type")
        if isinstance(task_type, str):
            raw_candidates.append(task_type)
        elif isinstance(task_type, list):
            raw_candidates.extend([t for t in task_type if isinstance(t, str)])

        category = bench_meta.get("category")
        if isinstance(category, str):
            raw_candidates.append(category)

        raw_candidates.extend([t for t in tags if isinstance(t, str)])
        raw_candidates.extend([d for d in eval_dims if isinstance(d, str)])
        description = bench_meta.get("description") or ""
        if isinstance(description, str):
            raw_candidates.append(description)
        raw_candidates.append(bench_name or "")

        blob = " ".join(raw_candidates).lower()
        normalized: List[str] = []
        known_domains = set(DOMAIN_KEYWORD_RULES.keys())
        for d in explicit_domain_values:
            dn = d.strip().lower()
            if dn in known_domains:
                normalized.append(dn)

        for domain, keywords in DOMAIN_KEYWORD_RULES.items():
            for kw in keywords:
                if kw and kw in blob:
                    normalized.append(domain)
                    break

        # fallback: 用简单分词，补捉可能未覆盖的显式 domain 值
        tokens = set(re.findall(r"[a-zA-Z]+", blob))
        if "finance" in tokens or "financial" in tokens or "finqa" in tokens:
            normalized.append("finance")
        if "medical" in tokens or "clinical" in tokens:
            normalized.append("medical")
        if "legal" in tokens or "law" in tokens:
            normalized.append("legal")

        if not normalized:
            normalized.append("general")
        return list(dict.fromkeys(normalized))

    def _get_question_key_hint(self, bench_meta: Dict[str, Any]) -> Optional[str]:
        if not isinstance(bench_meta, dict):
            return None
        for key in ["question_key", "query_key", "prompt_key"]:
            value = bench_meta.get(key)
            if isinstance(value, str) and value.strip():
                return value
        key_mapping = bench_meta.get("key_mapping")
        if isinstance(key_mapping, dict):
            for key in ["input_question_key", "input_text_key", "input_context_key"]:
                value = key_mapping.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return None

    def _extract_path_value(self, obj: Any, path: Optional[str]) -> Any:
        if not path or not isinstance(path, str):
            return None
        cur = obj
        for part in path.split("."):
            if isinstance(cur, dict):
                if part not in cur:
                    return None
                cur = cur.get(part)
                continue
            if isinstance(cur, list) and part.isdigit():
                idx = int(part)
                if idx < 0 or idx >= len(cur):
                    return None
                cur = cur[idx]
                continue
            return None
        return cur

    def _get_question(self, rec: Dict[str, Any], key_hint: Optional[str] = None) -> Any:
        hinted = self._extract_path_value(rec, key_hint)
        if hinted is not None and str(hinted).strip():
            return hinted
        for k in ["question", "query", "prompt", "instruction", "input", "text", "context", "problem", "task"]:
            if k in rec and str(rec[k]).strip():
                return rec[k]
        return ""

    def _get_pred(self, rec: Dict[str, Any], key_hint: Optional[str] = None) -> Any:
        hinted = self._extract_path_value(rec, key_hint)
        if hinted is not None:
            return hinted
        for k in ["predict", "prediction", "output", "response", "pred", "generated_ans", "model_output", "completion", "generated_text", "eval_pred"]:
            if k in rec:
                return rec[k]
        return None

    def _get_ref(self, rec: Dict[str, Any], key_hint: Optional[str] = None) -> Any:
        hinted = self._extract_path_value(rec, key_hint)
        if hinted is not None:
            return hinted
        for k in ["target", "reference", "ground_truth", "label", "labels", "targets", "answer", "solution", "correct_answer", "gold"]:
            if k in rec:
                return rec[k]
        return None

    def _load_records(self, path: Optional[str]) -> List[Dict[str, Any]]:
        if not path or not isinstance(path, str):
            return []
        if not os.path.exists(path):
            return []
        if path.endswith(".jsonl"):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        records.append(obj)
            return records
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                for k in ["records", "predictions", "labels", "data", "items"]:
                    if k in obj and isinstance(obj[k], list):
                        return [x for x in obj[k] if isinstance(x, dict)]
        return []

    def _safe_float(self, value: Any) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return 0.0
            try:
                return float(s)
            except Exception:
                return 0.0
        if isinstance(value, dict):
            for v in value.values():
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    s = v.strip()
                    if not s:
                        continue
                    try:
                        return float(s)
                    except Exception:
                        continue
        return 0.0

    async def _generate_llm_summary(self, payload: Dict[str, Any], state: NodeState) -> str:
        lang = self._get_lang(state)
        if lang.lower().startswith("en"):
            system_prompt = "You are an evaluation report analyst."
            user_prompt = (
                "Summarize the evaluation in concise English. "
                "You must include benchmark profile information and domain-specific performance analysis "
                "(e.g., finance-domain strengths/weaknesses). "
                "Do not use markdown horizontal rules like --- . "
                f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
            )
        else:
            system_prompt = "你是评测报告分析专家。"
            user_prompt = (
                "请用简洁中文总结模型表现。"
                "必须包含 benchmark 本身信息（如领域、任务、数据规模）以及领域表现分析"
                "（例如金融领域的优势/短板）。"
                "不要输出 markdown 分割线（例如 ---）。"
                f"输入如下：\n{json.dumps(payload, ensure_ascii=False)}"
            )

        try:
            llm = self.create_llm(state)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            resp = await llm.call(messages, bind_post_tools=False)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            log.warning(f"LLM summary failed: {e}")
            return self._fallback_summary(payload, lang)

    def _fallback_summary(self, payload: Dict[str, Any], lang: str = "zh") -> str:
        benches = payload.get("benches") or []
        if not benches:
            return "暂无可用评测结果。" if not str(lang).lower().startswith("en") else "No evaluation results available yet."
        sorted_benches = sorted(benches, key=lambda x: self._safe_float(x.get("primary_score")), reverse=True)
        best = sorted_benches[0]
        worst = sorted_benches[-1]
        overall = self._safe_float(payload.get("overall_score"))
        if str(lang).lower().startswith("en"):
            return f"Overall score is about {overall:.4f}. Best benchmark is {best.get('bench')}, weakest is {worst.get('bench')}."
        return f"整体得分约 {overall:.4f}。最佳数据集为 {best.get('bench')}，较弱数据集为 {worst.get('bench')}。"
