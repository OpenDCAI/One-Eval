from typing import List, Any, Dict, Optional
from one_eval.core.metric_registry import register_metric, MetricCategory
from one_eval.utils.extractor import AnswerExtractor

try:
    from math_verify import parse, verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False

# 初始化提取器实例
extractor = AnswerExtractor()

def _try_math_verify_compare(answer: Any, ground_truth: Any) -> Optional[bool]:
    """尝试使用 math_verify 库进行数学等价性校验"""
    if not HAS_MATH_VERIFY:
        return None
    try:
        # 尝试直接解析
        return verify(parse(str(ground_truth)), parse(str(answer)))
    except Exception:
        try:
            # 备用解析策略
            return verify(parse(ground_truth), parse(answer))
        except Exception:
            return None

@register_metric(
    name="math_verify",
    desc="数学等价性校验 (Hybrid: Text Match + Math Verify)",
    usage="数学问题/QA (GSM8K, MATH)",
    categories=[MetricCategory.QA_SINGLE],
    groups={
        "math": "primary",
        "qa": "primary"
    },
    priority="primary"
)
def compute_math_verify(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    DataFlow 核心评测逻辑移植：
    采用混合策略计算准确率：
    1. 文本包含匹配 (Text Containment, Normalized)
    2. 数学等价性校验 (Math Equivalence via math_verify)
    
    只要满足其一即判为正确 (Score=1.0)。
    """
    scores = []
    details = []
    
    for p_raw, r in zip(preds, refs):
        # 兼容处理：ref 可能是单个值也可能是列表（多参考答案）
        targets = r if isinstance(r, list) else [r]
        
        # 1. 提取答案 (使用移植过来的强大提取器)
        p_extracted = extractor.extract_answer(p_raw, None)
        
        is_match = False
        match_type = "none"
        
        for gt in targets:
            if gt is None:
                continue
                
            # A. 文本匹配检查 (检查原始预测和提取后预测)
            text_ok = extractor.text_contains_match(p_raw, gt) or extractor.text_contains_match(p_extracted, gt)
            
            # B. 数学等价性检查
            math_res = _try_math_verify_compare(p_extracted, gt)
            
            if text_ok:
                is_match = True
                match_type = "text_match"
                break
            elif math_res is True:
                is_match = True
                match_type = "math_verify"
                break
        
        scores.append(1.0 if is_match else 0.0)
        details.append({
            "score": 1.0 if is_match else 0.0,
            "match_type": match_type,
            "extracted": str(p_extracted),
            "raw_pred": str(p_raw)[:100]  # 记录部分原始预测以便调试
        })

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": details
    }