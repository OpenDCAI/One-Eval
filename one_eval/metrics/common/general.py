"""
general.py —— 通用维度指标:正确性(correctness) + 格式遵循(format)。

设计原则:每个指标只服务一个清晰的质量维度,不做参数变体的重复注册。
- correctness:exact_match / numerical_match / choice_accuracy / multilabel_f1
- format    :extraction_rate(可抽取性) / format_compliance(答案分隔规范度)
- diagnostic:missing_answer_rate(弃答率,= 1 - 抽取率,供归因)
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from one_eval.utils.extractor import (
    normalize_text,
    extract_first_number,
    extract_choice,
    extract_multi_choice,
    AnswerExtractor,
)
from one_eval.core.metric_registry import register_metric, MetricCategory, MetricDimension
from one_eval.metrics.parsers import parse_value


# ============================ 正确性维度 ============================

@register_metric(
    name="exact_match",
    desc="答案完全匹配 (EM)，支持归一化/包含/严格三种模式",
    usage="短答案、抽取式 QA。strict=True 走大小写敏感的原样匹配；use_containment=True 允许参考答案被预测包含即算对",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI],
    aliases=["em"],
    dimension=MetricDimension.CORRECTNESS,
)
def compute_exact_match(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """归一化完全匹配，多参考答案命中其一即算对。

    kwargs:
        strict: True 则大小写敏感、不归一化 (默认 False)
        use_containment: True 则参考答案被预测文本包含也算对 (默认 False)
    """
    strict = kwargs.get("strict", False)
    use_containment = kwargs.get("use_containment", False)
    scores = []

    for p, r in zip(preds, refs):
        r_list = r if isinstance(r, list) else [r]
        p_norm = str(p) if strict else normalize_text(p).lower()

        match = 0.0
        for gold in r_list:
            g_norm = str(gold) if strict else normalize_text(gold).lower()
            if p_norm == g_norm:
                match = 1.0
                break
            if use_containment and not strict and AnswerExtractor.text_contains_match(p, gold):
                match = 1.0
                break
        scores.append(match)

    return {"score": sum(scores) / len(scores) if scores else 0.0, "details": scores}


@register_metric(
    name="numerical_match",
    desc="数值软匹配 (1.0 == 1，容忍浮点误差)",
    usage="算术题/数值填空。先抽取答案再按 atol 比较数值",
    categories=[MetricCategory.QA_SINGLE],
    dimension=MetricDimension.CORRECTNESS,
)
def compute_numerical_match(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    atol = float(kwargs.get("atol", 1e-6))
    scores, pred_vals, ref_vals = [], [], []
    extractor = AnswerExtractor()
    from one_eval.utils.extractor import safe_float

    for p, r in zip(preds, refs):
        pv = safe_float(extractor.extract_answer(p, use_last_number=True))
        rv = safe_float(extractor.extract_answer(r, use_last_number=True))
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
        "artifacts": {"pred_vals": pred_vals, "ref_vals": ref_vals},
    }


@register_metric(
    name="choice_accuracy",
    desc="选择题选项准确率 (自动抽取 A/B/C/D)",
    usage="单选/多解选择题。自动从预测里抽出选项字母再比对",
    categories=[MetricCategory.CHOICE_SINGLE, MetricCategory.CHOICE_MULTI],
    aliases=["acc", "accuracy"],
    dimension=MetricDimension.CORRECTNESS,
)
def compute_choice_accuracy(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    parser_cfg = kwargs.get("parser")
    if isinstance(parser_cfg, dict) and parser_cfg.get("type"):
        return _compute_choice_accuracy_with_parser(preds, refs, **kwargs)

    scores: List[float] = []
    pred_choices: List[Optional[str]] = []
    ref_choices: List[Any] = []

    for p, r in zip(preds, refs):
        pc = extract_choice(p)
        pred_choices.append(pc)
        if pc is None:
            scores.append(0.0)
            ref_choices.append(str(r))
            continue

        is_match = False
        if isinstance(r, list):
            golds = [g for g in (extract_choice(x) for x in r) if g]
            is_match = pc in golds
            ref_choices.append(golds)
        else:
            gc = extract_choice(r)
            is_match = gc is not None and pc == gc
            ref_choices.append(gc)
        scores.append(1.0 if is_match else 0.0)

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores,
        "artifacts": {"pred_choices": pred_choices, "ref_choices": ref_choices},
    }


def _compute_choice_accuracy_with_parser(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    parser_cfg = kwargs.get("parser") or {"type": "choice_letter", "choices": "A-D"}
    denominator = str(kwargs.get("denominator") or "total")
    failure_policy = kwargs.get("failure_policy") or {}
    parse_failed_policy = failure_policy.get("parse_failed", "score_zero")
    empty_policy = failure_policy.get("empty_output", "score_zero")
    invalid_ref_policy = failure_policy.get("invalid_reference", "exclude")
    records = kwargs.get("records") or [None] * len(preds)

    details: List[Optional[float]] = []
    pred_choices: List[Optional[str]] = []
    ref_choices: List[Optional[str]] = []
    parse_results: List[Dict[str, Any]] = []

    valid_predictions = 0
    parse_failed = 0
    empty_output = 0
    invalid_references = 0
    denominator_count = 0
    score_sum = 0.0

    for idx, (pred, ref) in enumerate(zip(preds, refs)):
        record = records[idx] if idx < len(records) else None
        pred_parse = parse_value(pred, parser_cfg, record if isinstance(record, dict) else None)
        ref_parse = parse_value(ref, parser_cfg, record if isinstance(record, dict) else None)

        pred_choices.append(pred_parse.normalized if pred_parse.ok else None)
        ref_choices.append(ref_parse.normalized if ref_parse.ok else None)
        parse_results.append({
            "pred": pred_parse.to_dict(),
            "ref": ref_parse.to_dict(),
        })

        if pred_parse.ok:
            valid_predictions += 1
        elif pred_parse.error == "empty_output":
            empty_output += 1
        else:
            parse_failed += 1

        if not ref_parse.ok:
            invalid_references += 1
            if invalid_ref_policy == "exclude":
                details.append(None)
                continue

        should_count = denominator == "total"
        if denominator == "valid_only":
            should_count = pred_parse.ok and ref_parse.ok
        elif denominator == "official":
            should_count = ref_parse.ok

        if not should_count:
            details.append(None)
            continue

        denominator_count += 1
        if pred_parse.ok and ref_parse.ok:
            score = 1.0 if pred_parse.normalized == ref_parse.normalized else 0.0
        elif pred_parse.error == "empty_output" and empty_policy == "exclude":
            denominator_count -= 1
            details.append(None)
            continue
        elif pred_parse.error != "empty_output" and parse_failed_policy == "exclude":
            denominator_count -= 1
            details.append(None)
            continue
        else:
            score = 0.0

        details.append(score)
        score_sum += score

    score = score_sum / denominator_count if denominator_count > 0 else 0.0
    return {
        "score": score,
        "details": details,
        "total_samples": len(preds),
        "scored_samples": denominator_count,
        "valid_predictions": valid_predictions,
        "parse_failed": parse_failed,
        "empty_output": empty_output,
        "invalid_references": invalid_references,
        "denominator": denominator,
        "parser": parser_cfg,
        "failure_policy": failure_policy,
        "artifacts": {
            "pred_choices": pred_choices,
            "ref_choices": ref_choices,
            "parse_results": parse_results,
        },
    }


def _get_sets(p: Any, r: Any) -> Tuple[Set[str], Set[str]]:
    p_set = extract_multi_choice(p)
    r_set: Set[str] = set()
    if isinstance(r, list):
        for item in r:
            r_set.update(extract_multi_choice(item))
    else:
        r_set = extract_multi_choice(r)
    return p_set, r_set


@register_metric(
    name="multilabel_f1",
    desc="多标签/多选集合 F1",
    usage="多选题、多标签分类。预测与参考都解析成选项集合后算 F1",
    categories=[MetricCategory.CHOICE_MULTI, MetricCategory.QA_MULTI],
    dimension=MetricDimension.CORRECTNESS,
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
    return {"score": sum(scores) / len(scores) if scores else 0.0, "details": scores}

# __APPEND_FORMAT_DIMENSION__


# ============================ 格式遵循维度 ============================

def _extract_by_type(p: Any, extractor_type: str, answer_extractor=None) -> Optional[Any]:
    """按指定方式从预测里抽取“有效答案”，抽不到返回 None。"""
    if extractor_type == "choice":
        return extract_choice(p)
    if extractor_type == "generic":
        val = answer_extractor.extract_answer(p, use_last_number=False)
        return val or None
    return extract_first_number(p)


@register_metric(
    name="extraction_rate",
    desc="可抽取率:有多少样本能解析出有效答案 (数字/选项)",
    usage="所有需从长输出里提取答案的任务。低分说明模型没按格式输出，会拖累正确性",
    categories=[
        MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI,
        MetricCategory.CHOICE_SINGLE, MetricCategory.CHOICE_MULTI,
    ],
    dimension=MetricDimension.FORMAT,
)
def compute_extraction_rate(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """kwargs.extractor: "number"(默认) | "choice" | "generic"。"""
    extractor_type = str(kwargs.get("extractor", "number"))
    answer_extractor = AnswerExtractor() if extractor_type == "generic" else None

    details, extracted_values = [], []
    for p in preds:
        val = _extract_by_type(p, extractor_type, answer_extractor)
        extracted_values.append(val)
        details.append(1.0 if val is not None else 0.0)

    return {
        "score": sum(details) / len(details) if details else 0.0,
        "details": details,
        "artifacts": {"extracted_values": extracted_values, "extractor_used": extractor_type},
    }


@register_metric(
    name="missing_answer_rate",
    desc="弃答率:未能解析出有效答案的比例 (= 1 - 可抽取率)",
    usage="诊断模型拒答/跑题。配合 extraction_rate 做正确性归因",
    categories=[
        MetricCategory.QA_SINGLE, MetricCategory.CHOICE_SINGLE, MetricCategory.CHOICE_MULTI,
    ],
    dimension=MetricDimension.DIAGNOSTIC,
)
def compute_missing_answer_rate(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    result = compute_extraction_rate(preds, refs, **kwargs)
    return {
        "score": 1.0 - result["score"],
        "details": [1.0 - d for d in result["details"]],
        "artifacts": result["artifacts"],
    }


@register_metric(
    name="format_compliance_score",
    desc="格式规范度:答案是否被清晰分隔出来 (含 boxed/####/答案标记)",
    usage="评估输出是否把答案显式标出、便于定位。低分=答案淹没在啰嗦文本里",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.CHOICE_SINGLE],
    dimension=MetricDimension.FORMAT,
)
def compute_format_compliance_score(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """规则化打分,结果在 [0,1]。只看“答案是否被显式标记出来”这一可遵循的格式信号。

    评分项:
      +0.5 基础分(非空)
      +0.5 含明确答案分隔标记(\\boxed{}, ####, "答案:"/"answer:"/"final answer" 等)
    不再对代码块扣分:任务可能本来就要输出代码,代码块不是格式污染。
    """
    markers = [r"\boxed", "####", "answer:", "答案", "the answer is", "final answer"]
    scores, arts = [], []

    for p in preds:
        s = str(p).strip()
        if not s:
            scores.append(0.0)
            arts.append({"issue": "empty"})
            continue

        low = s.lower()
        has_marker = any(m in low for m in markers)
        score = 0.5 + (0.5 if has_marker else 0.0)
        scores.append(score)
        arts.append({"has_marker": has_marker})

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores,
        "artifacts": arts,
    }


# ============================ 流畅度维度 ============================

@register_metric(
    name="repetition_rate",
    desc="退化重复率:输出里 n-gram 复读的严重程度 (0=无重复, 越高越退化)",
    usage="生成健康度,无需 ref。抓“复读机/卡循环”这类失败模式。建议作诊断,分越低越好",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI, MetricCategory.TEXT_SCORE],
    dimension=MetricDimension.FLUENCY,
)
def compute_repetition_rate(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """逐条算 1 - distinct_n,即重复 n-gram 占比。只看预测本身,与参考无关。

    kwargs.n: n-gram 阶数 (默认 3)。文本太短(token 数 < n)记 0(无从判断重复)。
    distinct_n = 去重 n-gram 数 / 总 n-gram 数;复读越多 distinct 越低,重复率越高。
    """
    n = int(kwargs.get("n", 3))
    scores, arts = [], []

    for p in preds:
        tokens = normalize_text(p).split()
        if len(tokens) < n + 1:
            scores.append(0.0)
            arts.append({"rep": 0.0, "tokens": len(tokens), "note": "too_short"})
            continue
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total = len(grams)
        distinct = len(set(grams))
        rep = 1.0 - (distinct / total) if total > 0 else 0.0
        scores.append(rep)
        arts.append({"rep": rep, "distinct": distinct, "total": total})

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores,
        "artifacts": arts,
    }
