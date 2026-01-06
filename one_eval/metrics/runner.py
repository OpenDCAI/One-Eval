# one_eval/metrics/runner.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from one_eval.core.state import BenchInfo
from one_eval.logger import get_logger
from .registry import get_metric_fn

log = get_logger("MetricRunner")


class MetricRunner:
    def __init__(self):
        pass

    def run_bench(self, bench: BenchInfo, metrics_cfg: List[Dict[str, Any]]) -> Dict[str, Any]:
        inputs = self._resolve_inputs(bench)
        if not inputs:
            return {"error": "missing_inputs"}

        try:
            preds, refs, align_info = self._load_pred_ref(inputs, bench)
        except Exception as e:
            return {"error": f"load_failed: {str(e)}"}

        results: Dict[str, Any] = {
            "num_samples": len(refs),
            "alignment": align_info,
            "metrics": {},
        }

        for cfg in metrics_cfg:
            name = cfg.get("name")
            fn = get_metric_fn(name)
            if not fn:
                results["metrics"][name] = {"error": "metric_not_implemented", "score": 0.0}
                continue

            try:
                res = fn(preds, refs, **(cfg.get("args", {}) or {}))
                results["metrics"][name] = {
                    **res,
                    "priority": cfg.get("priority", "secondary"),
                    "desc": cfg.get("desc", ""),
                }
            except Exception as e:
                log.error(f"Metric {name} error: {e}")
                results["metrics"][name] = {"error": str(e), "score": 0.0}

        return results

    def _resolve_inputs(self, bench: BenchInfo) -> Optional[Dict[str, Any]]:
        p = getattr(bench, "dataset_cache", None)
        if p and isinstance(p, str) and p.strip():
            path = Path(p)
            if path.is_dir():
                rec = self._first_existing(path, ["records.jsonl", "records.json"])
                if rec:
                    return {"mode": "records", "records_path": rec}

                pred = self._first_existing(
                    path,
                    [
                        "predict.jsonl",
                        "pred.jsonl",
                        "predictions.jsonl",
                        "predict.json",
                        "pred.json",
                        "predictions.json",
                    ],
                )
                gt = self._first_existing(
                    path,
                    [
                        "ground_truth.jsonl",
                        "gt.jsonl",
                        "labels.jsonl",
                        "ground_truth.json",
                        "gt.json",
                        "labels.json",
                    ],
                )
                if pred and gt:
                    return {"mode": "split", "pred_path": pred, "gt_path": gt}

                return {"mode": "records", "records_path": path}

            return {"mode": "records", "records_path": path}

        meta = getattr(bench, "meta", {}) or {}
        ap = meta.get("artifact_paths") or {}

        rp = ap.get("records") or ap.get("records_path")
        if isinstance(rp, str) and rp.strip():
            return {"mode": "records", "records_path": Path(rp)}

        pred = (
            ap.get("predict")
            or ap.get("pred")
            or ap.get("prediction")
            or ap.get("predict_file")
            or ap.get("pred_file")
            or ap.get("prediction_file")
        )
        gt = (
            ap.get("ground_truth")
            or ap.get("gt")
            or ap.get("labels")
            or ap.get("ground_truth_file")
            or ap.get("gt_file")
            or ap.get("labels_file")
        )
        if isinstance(pred, str) and pred.strip() and isinstance(gt, str) and gt.strip():
            return {"mode": "split", "pred_path": Path(pred), "gt_path": Path(gt)}

        return None

    def _first_existing(self, root: Path, names: List[str]) -> Optional[Path]:
        for name in names:
            cand = root / name
            if cand.exists():
                return cand
        return None

    def _load_pred_ref(self, inputs: Dict[str, Any], bench: BenchInfo) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
        mode = inputs.get("mode")
        if mode == "records":
            records_path: Path = inputs["records_path"]
            records = self._load_records(records_path)
            preds = [self._get_pred(r) for r in records]
            refs = [self._get_ref(r) for r in records]
            return preds, refs, {"mode": "records", "path": str(records_path)}

        pred_path: Path = inputs["pred_path"]
        gt_path: Path = inputs["gt_path"]

        pred_items = self._load_records(pred_path)
        gt_items = self._load_records(gt_path)

        meta = getattr(bench, "meta", {}) or {}
        id_key = meta.get("id_key")
        if not isinstance(id_key, str) or not id_key.strip():
            id_key = self._guess_id_key(gt_items) or self._guess_id_key(pred_items)

        if not id_key:
            raise ValueError("missing_id_key")

        pred_index = self._index_by_id(pred_items, id_key)
        gt_index = self._index_by_id(gt_items, id_key)

        preds: List[Any] = []
        refs: List[Any] = []

        missing_pred = 0
        extra_pred = 0

        for sid, gt_rec in gt_index.items():
            pred_rec = pred_index.get(sid)
            if pred_rec is None:
                missing_pred += 1
                preds.append(None)
            else:
                preds.append(self._get_pred(pred_rec))
            refs.append(self._get_ref(gt_rec))

        for sid in pred_index.keys():
            if sid not in gt_index:
                extra_pred += 1

        return preds, refs, {
            "mode": "split",
            "pred_path": str(pred_path),
            "gt_path": str(gt_path),
            "id_key": id_key,
            "gt_samples": len(gt_index),
            "pred_samples": len(pred_index),
            "missing_pred": missing_pred,
            "extra_pred": extra_pred,
        }

    def _guess_id_key(self, items: List[Dict[str, Any]]) -> Optional[str]:
        if not items:
            return None
        cand = ["sample_id", "id", "uid", "uuid"]
        first = items[0]
        for k in cand:
            if k in first:
                return k
        return None

    def _index_by_id(self, items: List[Dict[str, Any]], id_key: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for rec in items:
            if not isinstance(rec, dict):
                continue
            sid = rec.get(id_key)
            if sid is None:
                continue
            s = str(sid)
            if s in out:
                raise ValueError(f"duplicate_id: {s}")
            out[s] = rec
        return out

    def _load_records(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(str(path))

        if path.suffix.lower() == ".jsonl":
            records: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        records.append(obj)
            return records

        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                if "records" in obj and isinstance(obj["records"], list):
                    return [x for x in obj["records"] if isinstance(x, dict)]
                if "predictions" in obj and isinstance(obj["predictions"], list):
                    return [x for x in obj["predictions"] if isinstance(x, dict)]
                if "labels" in obj and isinstance(obj["labels"], list):
                    return [x for x in obj["labels"] if isinstance(x, dict)]

        raise ValueError(f"unsupported_file: {path}")

    def _get_pred(self, rec: Dict[str, Any]) -> Any:
        keys = ["predict", "prediction", "output", "response", "pred"]
        for k in keys:
            if k in rec:
                return rec[k]
        return None

    def _get_ref(self, rec: Dict[str, Any]) -> Any:
        keys = ["target", "reference", "ground_truth", "label", "labels", "targets", "answer"]
        for k in keys:
            if k in rec:
                return rec[k]
        return None
