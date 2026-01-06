from typing import List, Dict, Any
from one_eval.utils.extractor import extract_first_number

def compute_numerical_match(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    atol = float(kwargs.get("atol", 1e-6))
    scores = []
    pred_vals = []
    ref_vals = []

    for p, r in zip(preds, refs):
        # 1. 在这里调用 extractor
        pv = extract_first_number(p)
        rv = extract_first_number(r)
        
        pred_vals.append(pv)
        ref_vals.append(rv)

        if pv is None or rv is None:
            scores.append(0.0)
        else:
            scores.append(1.0 if abs(pv - rv) <= atol else 0.0)

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores,
        "artifacts": {"pred_vals": pred_vals, "ref_vals": ref_vals}
    }
