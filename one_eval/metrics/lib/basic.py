from typing import List, Dict, Any
from one_eval.utils.extractor import normalize_text

def compute_exact_match(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    strict = kwargs.get("strict", False)
    scores = []
    
    for p, r in zip(preds, refs):
        # 处理 Multi-reference (r 可能是 list)
        r_list = r if isinstance(r, list) else [r]
        
        p_norm = str(p) if strict else normalize_text(p).lower()
        
        match = 0.0
        for gold in r_list:
            g_norm = str(gold) if strict else normalize_text(gold).lower()
            if p_norm == g_norm:
                match = 1.0
                break
        scores.append(match)

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores
    }
