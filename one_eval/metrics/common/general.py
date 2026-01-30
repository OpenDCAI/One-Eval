# one_eval/metrics/lib/core.py
from typing import List, Dict, Any, Optional, Set, Tuple
from one_eval.utils.extractor import (
    normalize_text, 
    extract_first_number, 
    extract_choice, 
    extract_multi_choice,
    AnswerExtractor,
    safe_float
)
from one_eval.core.metric_registry import register_metric, MetricCategory

@register_metric(
    name="exact_match",
    desc="抽取式答案完全匹配 (EM)",
    usage="短答案/抽取式 QA",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI],
    aliases=["em"],
    groups={
        "qa_extractive": "primary",
        "long_context_qa": "secondary"
    },
    priority="primary"
)
def compute_exact_match(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    strict = kwargs.get("strict", False)
    use_containment = kwargs.get("use_containment", False) # 新增参数
    scores = []
    
    for p, r in zip(preds, refs):
        # 处理 Multi-reference (r 可能是 list)
        r_list = r if isinstance(r, list) else [r]
        
        p_norm = str(p) if strict else normalize_text(p).lower()
        
        match = 0.0
        for gold in r_list:
            g_norm = str(gold) if strict else normalize_text(gold).lower()
            # 1. Exact Match (原有逻辑)
            if p_norm == g_norm:
                match = 1.0
                break
            
            # 2. Containment Match (新增: DataFlow 风格的模糊匹配)
            if use_containment and not strict:
                if AnswerExtractor.text_contains_match(p, gold):
                    match = 1.0
                    break
                    
        scores.append(match)

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores
    }

@register_metric(
    name="containment_match",
    desc="文本包含匹配 (DataFlow style)",
    usage="QA/Reasoning where answer is short and contained in pred",
    categories=[MetricCategory.QA_SINGLE],
    priority="secondary"
)
def compute_containment_match(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    检查参考答案是否包含在预测结果中 (Fuzzy Match).
    使用 DataFlow 的 AnswerExtractor.text_contains_match 逻辑.
    """
    return compute_exact_match(preds, refs, strict=False, use_containment=True, **kwargs)


@register_metric(
    name="strict_match",
    desc="原始字符串严格匹配",
    usage="格式严格的任务",
    categories=[MetricCategory.QA_SINGLE],
    groups={
        "numerical": "secondary", 
        "symbolic": "secondary"
    },
    priority="secondary"
)
def compute_strict_match(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """严格匹配 (Case-sensitive, no normalization)"""
    return compute_exact_match(preds, refs, strict=True, **kwargs)


# --- From numerical.py ---

@register_metric(
    name="numerical_match",
    desc="数值软匹配(1.0 == 1，容忍浮点误差)",
    usage="算术题/数值填空",
    categories=[MetricCategory.QA_SINGLE],
    groups={
        "numerical": "primary",
        "math_numeric": "primary"
    },
    priority="primary"
)
def compute_numerical_match(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    atol = float(kwargs.get("atol", 1e-6))
    scores = []
    pred_vals = []
    ref_vals = []
    
    # 使用增强版提取器 (支持 CoT/Last Number Extraction)
    extractor = AnswerExtractor()

    for p, r in zip(preds, refs):
        # 1. 尝试使用 AnswerExtractor 提取 (默认策略: 优先 Boxed, 否则 Last Number)
        # AnswerExtractor 返回的是清洗后的字符串
        pv_str = extractor.extract_answer(p, use_last_number=True)
        rv_str = extractor.extract_answer(r, use_last_number=True)
        
        # 2. 转为 float
        pv = safe_float(pv_str)
        rv = safe_float(rv_str)
        
        # Fallback: 如果新提取器没提取到，尝试旧的 extract_first_number (仅作保险，通常不需要)
        if pv is None:
            pv = extract_first_number(p)
        if rv is None:
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


# --- From choice.py ---

@register_metric(
    name="choice_accuracy",
    desc="选项字母或离散标签准确率",
    usage="选择题 (A/B/C/D)",
    categories=[MetricCategory.CHOICE_SINGLE, MetricCategory.CHOICE_MULTI],
    groups={
        "choice": "primary"
    },
    priority="primary"
)
def compute_choice_accuracy(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算选项准确率 (Choice Accuracy)
    Args:
        preds: 预测结果列表 (Raw strings)
        refs: 参考答案列表 (Raw strings or List of strings)
    """
    scores: List[float] = []
    pred_choices: List[Optional[str]] = []
    ref_choices: List[Any] = [] # 用于 artifact 展示，可能是 str 或 list

    for p, r in zip(preds, refs):
        # 1. 提取预测结果的选项 (A/B/C/D)
        pc = extract_choice(p)
        pred_choices.append(pc)

        # 2. 如果没提取出来，直接判错
        if pc is None:
            scores.append(0.0)
            ref_choices.append(str(r)) # 记录一下原始 ref 方便 debug
            continue

        # 3. 处理参考答案 (支持多选/多解)
        is_match = False
        
        if isinstance(r, list):
            # 如果 ref 是列表，说明有多个正确答案，命中任何一个都算对
            golds = []
            for item in r:
                g = extract_choice(item)
                if g: golds.append(g)
            
            if pc in golds:
                is_match = True
            ref_choices.append(golds)
            
        else:
            # 单个 ref
            gc = extract_choice(r)
            if gc is not None and pc == gc:
                is_match = True
            ref_choices.append(gc)

        scores.append(1.0 if is_match else 0.0)

    score = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "score": score,
        "details": scores,
        "artifacts": {
            "pred_choices": pred_choices,
            "ref_choices": ref_choices
        }
    }


# --- From diagnostic.py ---

@register_metric(
    name="extraction_rate",
    desc="正则提取成功率 (强烈建议)",
    usage="所有需要从长输出提取答案的任务",
    categories=[
        MetricCategory.QA_SINGLE, 
        MetricCategory.QA_MULTI,
        MetricCategory.CHOICE_SINGLE,
        MetricCategory.CHOICE_MULTI
    ],
    groups={
        "numerical": "diagnostic",
        "symbolic": "diagnostic",
        "choice": "diagnostic",
        "qa_extractive": "diagnostic"
    },
    priority="diagnostic"
)
def compute_extraction_rate(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算提取率：有多少样本能成功提取出有效格式（数字/选项）。
    Args:
        extractor: "number" (默认) | "choice" | "generic"
    """
    extractor_type = str(kwargs.get("extractor", "number"))
    
    valid_count = 0
    extracted_values = []
    details = []

    # Helper for generic extraction
    answer_extractor = AnswerExtractor() if extractor_type == "generic" else None

    for p in preds:
        val = None
        if extractor_type == "choice":
            val = extract_choice(p)
        elif extractor_type == "generic":
            # Use AnswerExtractor to try to get *something*
            val = answer_extractor.extract_answer(p, use_last_number=False)
            if not val: # Empty string means failure
                val = None
        else:
            val = extract_first_number(p)
            
        extracted_values.append(val)
        
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

@register_metric(
    name="missing_answer_rate",
    desc="未输出有效选项/标签的比例 (诊断用)",
    usage="监控模型拒答率",
    categories=[MetricCategory.CHOICE_SINGLE, MetricCategory.CHOICE_MULTI],
    priority="diagnostic"
)
def compute_missing_answer_rate(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算丢失率 (1 - Extraction Rate)
    """
    result = compute_extraction_rate(preds, refs, **kwargs)
    return {
        "score": 1.0 - result["score"],
        "details": [1.0 - d for d in result["details"]],
        "artifacts": result["artifacts"]
    }

@register_metric(
    name="format_compliance_score",
    desc="格式遵循度评分 (有效内容占比)",
    usage="评估模型是否输出冗余废话",
    categories=[
        MetricCategory.QA_SINGLE, 
        MetricCategory.QA_MULTI,
        MetricCategory.CHOICE_SINGLE
    ],
    groups={
        "diagnostic": "secondary"
    },
    priority="diagnostic"
)
def compute_format_compliance_score(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算格式遵循度。
    主要指标：
    1. Conciseness: len(extracted) / len(total)
    2. Cleanliness: 是否包含 ```json 等标记 (如果要求纯文本)
    
    Args:
        extractor: "number" | "choice" | "none" (default: auto detect or none)
    """
    extractor_type = str(kwargs.get("extractor", "none"))
    scores = []
    artifacts = []

    for p in preds:
        pred_str = str(p).strip()
        if not pred_str:
            scores.append(0.0)
            artifacts.append({"ratio": 0.0, "issue": "empty"})
            continue

        # 1. Determine extracted content
        extracted = None
        if extractor_type == "choice":
            extracted = extract_choice(pred_str)
        elif extractor_type == "number":
            extracted = extract_first_number(pred_str)
            if extracted is not None: extracted = str(extracted)
        else:
            # If no extractor specified, assume the whole string should be the answer
            # In this case, we check for markdown artifacts
            pass

        # 2. Calculate Ratio
        if extracted:
            ratio = len(str(extracted)) / len(pred_str)
        else:
            # 如果没提取出来，或者没指定 extractor
            # 我们检查是否有 markdown code block
            if "```" in pred_str:
                # 包含了 markdown block，扣分
                # 假设有效内容是去掉 ``` 后的长度
                clean_len = len(pred_str.replace("```", "").replace("json", "").replace("python", ""))
                ratio = clean_len / len(pred_str) * 0.8 # Penalty
            else:
                ratio = 1.0 # Assume perfect if no artifacts found
        
        # Clamp
        ratio = min(1.0, max(0.0, ratio))
        scores.append(ratio)
        artifacts.append({"ratio": ratio, "extracted": extracted})

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores,
        "artifacts": artifacts
    }



# --- From multilabel.py ---

def _get_sets(p: Any, r: Any) -> Tuple[Set[str], Set[str]]:
    p_set = extract_multi_choice(p)
    r_set = set()
    if isinstance(r, list):
        for item in r:
            r_set.update(extract_multi_choice(item))
    else:
        r_set = extract_multi_choice(r)
    return p_set, r_set

@register_metric(
    name="multilabel_f1",
    desc="多标签分类 F1",
    usage="多标签分类任务",
    priority="primary"
)
def compute_multilabel_f1(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    scores = []
    for p, r in zip(preds, refs):
        p_set, r_set = _get_sets(p, r)
        
        if not p_set and not r_set:
            f1 = 1.0
        elif not p_set or not r_set:
            f1 = 0.0
        else:
            tp = len(p_set & r_set)
            fp = len(p_set - r_set)
            fn = len(r_set - p_set)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        scores.append(f1)
        
    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores
    }

def compute_jaccard_index(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    scores = []
    for p, r in zip(preds, refs):
        p_set, r_set = _get_sets(p, r)
        
        union = len(p_set | r_set)
        inter = len(p_set & r_set)
        
        if union == 0:
            score = 1.0 # Both empty
        else:
            score = inter / union
        scores.append(score)
        
    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores
    }