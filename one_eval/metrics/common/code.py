from typing import List, Any, Dict
from one_eval.metrics.core import register_metric, MetricCategory

@register_metric(
    name="pass_at_k",
    desc="Pass@k (Code Execution)",
    usage="代码生成",
    categories=[MetricCategory.QA_SINGLE],
    groups={
        "code": "primary"
    },
    priority="primary"
)
def compute_pass_at_k(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    Pass@k implementation placeholder.
    Real pass@k requires sandboxed execution which is risky and complex.
    """
    return {
        "score": 0.0, 
        "error": "Pass@k requires sandboxed execution environment which is not currently enabled."
    }

@register_metric(
    name="code_similarity",
    desc="代码相似度 (BLEU-based)",
    usage="代码生成",
    categories=[MetricCategory.QA_SINGLE],
    groups={
        "code_sim": "primary",
        "code": "secondary"
    },
    priority="secondary"
)
def compute_code_similarity(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    Proxy for code similarity using BLEU.
    """
    try:
        from .text_gen import compute_bleu
        return compute_bleu(preds, refs, **kwargs)
    except ImportError:
        return {"score": 0.0, "error": "Importing text_gen failed."}