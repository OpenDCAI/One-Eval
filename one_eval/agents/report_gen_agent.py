from __future__ import annotations

import json
import os
import time
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState, BenchInfo
from one_eval.logger import get_logger

log = get_logger("ReportGenAgent")

EVAL_TYPE_LABELS = [
    ("key1_text_score", "文本评分"),
    ("key2_qa", "单参考问答"),
    ("key2_q_ma", "多参考问答"),
    ("key3_q_choices_a", "单选题"),
    ("key3_q_choices_as", "多选题"),
    ("key3_q_a_rejected", "偏好对比"),
]


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
        overall_score = self._compute_overall_score(bench_summaries)

        macro_view = self._build_macro_view(bench_summaries, eval_results)
        diagnostic_view = self._build_diagnostic_view(benches, eval_results, metric_plan)
        analyst_view = self._collect_analyst_outputs(benches, eval_results)
        analyst_compact = self._compact_analyst_view(analyst_view)

        summary_payload = {
            "overall_score": overall_score,
            "benches": bench_summaries[:20],
            "error_distribution": diagnostic_view.get("error_distribution", []),
            "metric_summaries": analyst_compact.get("metric_summary", []),
            "case_studies": analyst_compact.get("case_study", []),
        }
        llm_summary = await self._generate_llm_summary(summary_payload, state)

        report = {
            "version": "v1",
            "generated_at": time.time(),
            "model": self._get_model_name(state),
            "overall": {
                "score": overall_score,
                "bench_summaries": bench_summaries,
            },
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
            plan = metric_plan.get(bench_name, []) or []
            primary_name = self._get_primary_metric_name(plan, metrics)
            primary = metrics.get(primary_name, {}) if primary_name else {}
            score = self._safe_float(primary.get("score"))
            num_samples = int(bench_result.get("num_samples", 0) or 0)
            summaries.append({
                "bench": bench_name,
                "eval_type": bench.bench_dataflow_eval_type or bench.meta.get("eval_type"),
                "domain": bench.meta.get("domain"),
                "task_type": bench.meta.get("task_type"),
                "num_samples": num_samples,
                "primary_metric": primary_name,
                "primary_score": score,
            })
        return summaries

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

    def _build_macro_view(self, summaries: List[Dict[str, Any]], eval_results: Dict[str, Any]) -> Dict[str, Any]:
        radar = self._build_radar(summaries)
        sunburst = self._build_sunburst(summaries)
        table = self._build_macro_table(summaries, eval_results)
        return {"radar": radar, "sunburst": sunburst, "table": table}

    def _build_radar(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        accum = {k: {"score": 0.0, "weight": 0} for k, _ in EVAL_TYPE_LABELS}
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
        labels = [label for _, label in EVAL_TYPE_LABELS]
        scores = []
        for key, _ in EVAL_TYPE_LABELS:
            if accum[key]["weight"] > 0:
                scores.append(accum[key]["score"] / accum[key]["weight"])
            else:
                scores.append(0.0)
        return {"labels": labels, "scores": scores}

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
        metric_plan: Dict[str, Any]
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
            if not primary_details:
                continue

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
                question = self._get_question(rec)

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

                error_type = self._classify_error(
                    is_correct,
                    self._safe_float(extraction_value) if extraction_value is not None else None,
                    self._safe_float(missing_value) if missing_value is not None else None,
                    self._safe_float(format_value) if format_value is not None else None,
                )

                if not is_correct:
                    error_counter[error_type] += 1
                    if len(cases) < 5:
                        cases.append({
                            "bench": bench_name,
                            "question": question,
                            "model_output": pred,
                            "ground_truth": ref,
                            "error_type": error_type,
                            "score": score
                        })

        error_distribution = []
        total_err = sum(error_counter.values())
        for k, v in error_counter.most_common():
            ratio = (v / total_err) if total_err > 0 else 0.0
            error_distribution.append({"type": k, "count": v, "ratio": ratio})

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
            summary = metrics.get("metric_summary_analyst", {}).get("summary")
            analysis = metrics.get("case_study_analyst", {}).get("analysis")
            if isinstance(summary, str) and summary.strip():
                metric_summary[bench_name] = summary
            if isinstance(analysis, str) and analysis.strip():
                case_study[bench_name] = analysis
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
    ) -> str:
        if is_correct:
            return "Correct"
        if extraction_value is not None and extraction_value <= 0:
            return "Extraction Error"
        if missing_value is not None and missing_value >= 1:
            return "Refusal / Empty Response" 
        if format_value is not None and format_value < 0.5:
            return "Format Error" 

        return "Incorrect Answer (Reasoning/Logic)" 

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

    def _get_question(self, rec: Dict[str, Any]) -> Any:
        for k in ["question", "query", "prompt", "instruction", "input", "text"]:
            if k in rec:
                return rec[k]
        return ""

    def _get_pred(self, rec: Dict[str, Any], key_hint: Optional[str] = None) -> Any:
        if key_hint and key_hint in rec:
            return rec[key_hint]
        for k in ["predict", "prediction", "output", "response", "pred", "generated_ans", "model_output", "completion", "generated_text", "eval_pred"]:
            if k in rec:
                return rec[k]
        return None

    def _get_ref(self, rec: Dict[str, Any], key_hint: Optional[str] = None) -> Any:
        if key_hint and key_hint in rec:
            return rec[key_hint]
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
        if isinstance(value, dict):
            for v in value.values():
                if isinstance(v, (int, float)):
                    return float(v)
        return 0.0

    async def _generate_llm_summary(self, payload: Dict[str, Any], state: NodeState) -> str:
        system_prompt = "你是评测报告分析专家，请用一段话总结模型表现。"
        user_prompt = f"评测结果摘要如下，请给出简洁中文总结：\n{json.dumps(payload, ensure_ascii=False)}"

        try:
            llm = self.create_llm(state)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            resp = await llm.call(messages, bind_post_tools=False)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            log.warning(f"LLM summary failed: {e}")
            return self._fallback_summary(payload)

    def _fallback_summary(self, payload: Dict[str, Any]) -> str:
        benches = payload.get("benches") or []
        if not benches:
            return "暂无可用评测结果。"
        sorted_benches = sorted(benches, key=lambda x: self._safe_float(x.get("primary_score")), reverse=True)
        best = sorted_benches[0]
        worst = sorted_benches[-1]
        overall = self._safe_float(payload.get("overall_score"))
        return f"整体得分约 {overall:.4f}。最佳数据集为 {best.get('bench')}，较弱数据集为 {worst.get('bench')}。"