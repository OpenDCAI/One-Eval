# one_eval/metrics/lib/diagnostic.py
from typing import List, Any, Dict
from one_eval.utils.extractor import extract_first_number, extract_choice

def compute_extraction_rate(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算提取率：有多少样本能成功提取出有效格式（数字/选项）。
    Args:
        preds: 预测结果列表
        refs: (占位符，不使用)
        kwargs: 
            - extractor: "number" (默认) | "choice"
    """
    extractor_type = str(kwargs.get("extractor", "number"))
    
    valid_count = 0
    extracted_values = []
    details = []

    for p in preds:
        val = None
        # 根据配置选择提取器
        if extractor_type == "choice":
            val = extract_choice(p)
        else:
            # 默认为 number
            val = extract_first_number(p)
            
        extracted_values.append(val)
        
        # 只要提取结果不是 None，就视为提取成功
        if val is not None:
            valid_count += 1
            details.append(1.0)
        else:
            details.append(0.0)

    score = valid_count / len(preds) if preds else 0.0

    return {
        "score": score,
        "details": details,
        "artifacts": {
            "extracted_values": extracted_values,
            "extractor_used": extractor_type
        }
    }

def compute_missing_answer_rate(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算丢失率 (1 - Extraction Rate)
    """
    # 复用上面的逻辑
    result = compute_extraction_rate(preds, refs, **kwargs)
    return {
        "score": 1.0 - result["score"],
        "details": [1.0 - d for d in result["details"]],
        "artifacts": result["artifacts"]
    }
