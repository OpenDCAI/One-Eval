from __future__ import annotations

import json
import os
import copy
from pathlib import Path
from typing import Any, Dict, List

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState, BenchInfo
from one_eval.logger import get_logger
from one_eval.metrics.runner import MetricRunner

log = get_logger("ScoreCalcAgent")


class ScoreCalcAgent(CustomAgent):
    """
    Step 3 Agent: Score计算
    """
    
    @property
    def role_name(self) -> str:
        return "ScoreCalcAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return ""

    @property
    def task_prompt_template_name(self) -> str:
        return ""

    def _load_records(self, path: str) -> List[Dict[str, Any]]:
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

    def _write_records(self, path: str, records: List[Dict[str, Any]]) -> None:
        if path.endswith(".jsonl"):
            with open(path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return
        if path.endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)

    def _build_step3_path(self, records_path: str) -> str:
        p = Path(records_path)
        stem = p.stem
        if stem.endswith("step2"):
            new_stem = stem[:-1] + "3"
        else:
            parts = stem.split("_")
            if len(parts) >= 2 and parts[-2] == "step" and parts[-1].isdigit():
                parts[-1] = "3"
                new_stem = "_".join(parts)
            elif stem.endswith("_step3"):
                new_stem = stem
            else:
                new_stem = f"{stem}_step3"
        return str(p.with_name(new_stem + p.suffix))

    def _attach_metric_details(self, records: List[Dict[str, Any]], metrics: Dict[str, Any]) -> bool:
        if not records or not metrics:
            return False
        total = len(records)
        updated = False
        for name, m in metrics.items():
            details = m.get("details")
            if not isinstance(details, list) or len(details) != total:
                continue
            for i, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                md = rec.get("metric_details")
                if not isinstance(md, dict):
                    md = {}
                    rec["metric_details"] = md
                md[name] = details[i]
            updated = True
        return updated

    def _strip_dataflow_eval_fields(self, records: List[Dict[str, Any]]) -> None:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            keys = [k for k in rec.keys() if isinstance(k, str) and k.startswith("eval_")]
            for k in keys:
                rec.pop(k, None)

    def _write_step3_file(self, bench: BenchInfo, bench_result: Dict[str, Any]) -> str | None:
        meta = bench.meta or {}
        records_path = meta.get("eval_detail_path") or meta.get("artifact_paths", {}).get("records_path")
        records = self._load_records(records_path)
        if not records:
            return None
        self._strip_dataflow_eval_fields(records)
        metrics = bench_result.get("metrics", {}) or {}
        if not self._attach_metric_details(records, metrics):
            return None
        step3_path = self._build_step3_path(records_path)
        self._write_records(step3_path, records)
        return step3_path

    def _get_lang(self, state: NodeState) -> str:
        req = getattr(state, "request", None)
        if isinstance(req, dict):
            return str(req.get("language") or "zh")
        if req is not None:
            return str(getattr(req, "language", "zh") or "zh")
        return "zh"

    async def run(self, state: NodeState) -> NodeState:
        benches: List[BenchInfo] = getattr(state, "benches", []) or []
        metric_plan: Dict[str, Any] = getattr(state, "metric_plan", {}) or {}

        if not benches:
            log.warning("state.benches 为空，跳过 score 计算")
            return state

        if not metric_plan:
            log.warning("state.metric_plan 为空，跳过 score 计算")
            return state

        if not getattr(state, "eval_results", None):
            state.eval_results = {}

        runner = MetricRunner()
        report_lang = self._get_lang(state)

        computed: List[str] = []
        failed: List[Dict[str, Any]] = []

        total_benches = len(benches)
        log.info(f"Start processing {total_benches} benches for score calculation.")

        for i, bench in enumerate(benches):
            bench_name = bench.bench_name
            log.info(f"[{bench_name}] Processing ({i+1}/{total_benches})...")
            
            # Check eval status first
            if getattr(bench, "eval_status", None) == "failed":
                log.warning(f"[{bench_name}] eval_status is 'failed', skipping score calculation.")
                failed.append({"bench": bench_name, "error": "Previous evaluation failed"})
                continue

            # === 自动关联 DataFlowEvalNode 的结果 ===
            if bench.meta and bench.meta.get("eval_detail_path"):
                detail_path = bench.meta["eval_detail_path"]
                if "artifact_paths" not in bench.meta:
                    bench.meta["artifact_paths"] = {}
                # 将 detail_path (step2 result) 注册为 records_path
                # MetricRunner 会优先读取 records_path
                bench.meta["artifact_paths"]["records_path"] = detail_path
                log.info(f"[{bench_name}] Linked eval_detail_path to artifact_paths['records_path']: {detail_path}")
            else:
                log.warning(f"[{bench_name}] Missing eval_detail_path. MetricRunner might fail or fallback to raw dataset.")

            plan = metric_plan.get(bench_name, []) or []
            if not plan:
                log.warning(f"[{bench_name}] No metric plan found, skipping.")
                continue

            runtime_plan = copy.deepcopy(plan)
            for metric_cfg in runtime_plan:
                if not isinstance(metric_cfg, dict):
                    continue
                args = metric_cfg.get("args")
                if not isinstance(args, dict):
                    args = {}
                args["language"] = report_lang
                metric_cfg["args"] = args

            log.info(f"[{bench_name}] Running metrics: {[p.get('name') for p in runtime_plan if isinstance(p, dict)]}")
            
            try:
                bench_result = runner.run_bench(bench, runtime_plan)
            except Exception as e:
                log.error(f"[{bench_name}] Critical error in runner.run_bench: {e}", exc_info=True)
                bench_result = {"error": str(e)}
                
            state.eval_results[bench_name] = bench_result

            metrics = bench_result.get("metrics", {})
            num_samples = bench_result.get("num_samples", 0)
            
            # Extract simple metrics for meta display
            simple_metrics = {mname: mres.get("score") for mname, mres in metrics.items() if isinstance(mres, dict) and "score" in mres}

            # Update bench.meta["eval_result"] while protecting DataFlow base score fields
            if "eval_result" not in bench.meta or not isinstance(bench.meta["eval_result"], dict):
                bench.meta["eval_result"] = {}
            protected_keys = {"score", "accuracy", "exact_match", "valid_samples", "total_samples"}
            for mname, mscore in simple_metrics.items():
                if mname in protected_keys:
                    bench.meta["eval_result"][f"metric_{mname}"] = mscore
                else:
                    bench.meta["eval_result"][mname] = mscore

            # Update bench.meta["metric_summary"] so frontend can display them
            metric_summary_text = metrics.get("metric_summary_analyst", {}).get("summary")
            if metric_summary_text:
                bench.meta["metric_summary"] = metric_summary_text
            
            summary = {
                "bench": bench_name,
                "num_samples": num_samples,
                "metrics": simple_metrics,
                "analyst": {
                    "metric_summary": metrics.get("metric_summary_analyst", {}).get("summary"),
                    "case_study": metrics.get("case_study_analyst", {}).get("analysis"),
                },
            }
            log.info(f"[{bench_name}] Summary: {summary}")

            step3_path = self._write_step3_file(bench, bench_result)
            if step3_path:
                if not bench.meta:
                    bench.meta = {}
                bench.meta["eval_step3_path"] = step3_path
                log.info(f"[{bench_name}] Step3 records saved: {step3_path}")

            if isinstance(bench_result, dict) and bench_result.get("error"):
                failed.append({"bench": bench_name, "error": bench_result.get("error")})
            else:
                computed.append(bench_name)

        if not getattr(state, "result", None):
            state.result = {}

        state.result[self.role_name] = {
            "computed": computed,
            "failed": failed,
        }

        return state
