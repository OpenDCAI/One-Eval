"""
general.py —— 通用维度指标:正确性(correctness) + 格式遵循(format)。

设计原则:每个指标只服务一个清晰的质量维度,不做参数变体的重复注册。
- correctness:exact_match / numerical_match / choice_accuracy / multilabel_f1
- format    :extraction_rate(可抽取性) / format_compliance(答案分隔规范度) / json_validity
- diagnostic:missing_answer_rate(弃答率,= 1 - 抽取率,供归因) / empty_or_whitespace_rate
- fluency   :repetition_rate(退化重复) / garbled_text_rate(乱码/异常编码)
"""
import json
import re
import unicodedata
from typing import List, Dict, Any, Optional, Set, Tuple
from one_eval.utils.extractor import (
    normalize_text,
    extract_first_number,
    extract_choice,
    extract_multi_choice,
    extract_answer_set,
    AnswerExtractor,
)
from one_eval.core.metric_registry import register_metric, MetricCategory, MetricDimension


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


@register_metric(
    name="set_f1",
    desc="开放答案集合 F1",
    usage="列表型 QA、实体抽取、多答案短答。将预测与参考解析为开放文本集合后计算 F1",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI],
    dimension=MetricDimension.CORRECTNESS,
)
def compute_set_f1(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    scores = []
    artifacts = {"pred_sets": [], "ref_sets": [], "tp": [], "fp": [], "fn": []}

    for p, r in zip(preds, refs):
        p_set = extract_answer_set(p)
        r_set = extract_answer_set(r)
        tp = p_set & r_set
        fp = p_set - r_set
        fn = r_set - p_set

        if not p_set and not r_set:
            f1 = 1.0
        elif not p_set or not r_set:
            f1 = 0.0
        else:
            prec = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
            rec = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        scores.append(f1)
        artifacts["pred_sets"].append(sorted(p_set))
        artifacts["ref_sets"].append(sorted(r_set))
        artifacts["tp"].append(sorted(tp))
        artifacts["fp"].append(sorted(fp))
        artifacts["fn"].append(sorted(fn))

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores,
        "artifacts": artifacts,
    }

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


# ============================ 诊断维度 ============================

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
    name="empty_or_whitespace_rate",
    desc="空输出率:输出为 None、空字符串或纯空白的比例",
    usage="失败归因诊断。低分说明模型/服务根本没有产出可评内容；分越低越好",
    categories=[
        MetricCategory.TEXT_SCORE,
        MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI,
        MetricCategory.CHOICE_SINGLE, MetricCategory.CHOICE_MULTI,
        MetricCategory.PAIRWISE,
    ],
    dimension=MetricDimension.DIAGNOSTIC,
)
def compute_empty_or_whitespace_rate(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """计算空输出/纯空白输出比例。"""
    details = [1.0 if p is None or str(p).strip() == "" else 0.0 for p in preds]
    empty_indices = [idx for idx, score in enumerate(details) if score == 1.0]

    total = len(details)
    empty_count = len(empty_indices)
    return {
        "score": empty_count / total if total else 0.0,
        "details": details,
        "artifacts": {
            "empty_count": empty_count,
            "total": total,
            "empty_indices": empty_indices,
        },
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


def _extract_json_candidate(text: str) -> Tuple[str, bool]:
    """Return a JSON candidate and whether it was extracted from surrounding text."""
    s = text.strip()
    if not s:
        return s, False

    fenced = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)```", s)
    if fenced:
        return fenced.group(1).strip(), True

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(s[idx:])
        except json.JSONDecodeError:
            continue
        return s[idx:idx + end].strip(), idx != 0 or idx + end != len(s)

    return s, False


def _parse_json_prediction(pred: Any, allow_extraction: bool) -> Tuple[bool, Dict[str, Any]]:
    s = "" if pred is None else str(pred).strip()
    candidate = s
    extracted = False
    if allow_extraction:
        candidate, extracted = _extract_json_candidate(s)

    art: Dict[str, Any] = {
        "extracted": extracted,
        "candidate": candidate[:500],
    }
    if not candidate:
        art["error"] = "empty"
        return False, art

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        art["error"] = e.msg
        art["error_pos"] = e.pos
        return False, art

    art["parsed_type"] = type(parsed).__name__
    return True, art


@register_metric(
    name="json_validity",
    desc="JSON 合法率:输出是否可被 json.loads 解析",
    usage=(
        "结构化输出、tool calling、agent 任务、JSON-only 指令。默认严格要求整段输出是 JSON；"
        "allow_extraction=True 时允许从 markdown fenced block 或周围文本中提取首个 JSON 对象/数组再解析"
    ),
    categories=[
        MetricCategory.TEXT_SCORE,
        MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI,
        MetricCategory.CHOICE_SINGLE, MetricCategory.CHOICE_MULTI,
    ],
    dimension=MetricDimension.FORMAT,
)
def compute_json_validity(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """逐条检查预测是否是合法 JSON。

    kwargs.allow_extraction:
      False(默认): str(pred).strip() 必须整体可被 json.loads 解析。
      True: 可从 ```json fenced block``` 或文本中第一个 JSON object/array 提取后解析。
    """
    allow_extraction = bool(kwargs.get("allow_extraction", False))
    parsed = [_parse_json_prediction(p, allow_extraction=allow_extraction) for p in preds]
    details = [1.0 if ok else 0.0 for ok, _ in parsed]

    return {
        "score": sum(details) / len(details) if details else 0.0,
        "details": details,
        "artifacts": {
            "allow_extraction": allow_extraction,
            "items": [art for _, art in parsed],
        },
    }


# ============================ 流畅度维度 ============================

_ALLOWED_CONTROL_CHARS = {"\n", "\r", "\t"}
_STRONG_MOJIBAKE_PATTERNS = [
    "â€™", "â€œ", "â€\x9d", "â€¦", "â€“", "â€”",
    "ä¸", "ä½", "å›", "å¤", "æ˜", "æœ",
    "è¿", "è¯", "çš", "ç”", "ï¼", "ï½",
]


def _garbled_reasons(text: str) -> List[str]:
    reasons = []
    if "\ufffd" in text:
        reasons.append("replacement_char")

    for ch in text:
        code = ord(ch)
        category = unicodedata.category(ch)
        if category == "Cc" and ch not in _ALLOWED_CONTROL_CHARS:
            reasons.append("control_char")
            break
        if category == "Cs":
            reasons.append("surrogate")
            break
        if 0xFDD0 <= code <= 0xFDEF or (code & 0xFFFE) == 0xFFFE:
            reasons.append("noncharacter")
            break

    if any(pattern in text for pattern in _STRONG_MOJIBAKE_PATTERNS):
        reasons.append("mojibake")

    return reasons


@register_metric(
    name="garbled_text_rate",
    desc="乱码率:输出中明显编码损坏/异常 Unicode 的比例",
    usage="生成健康度,无需 ref。保守检测乱码和异常字符；分越低越好",
    categories=[
        MetricCategory.TEXT_SCORE,
        MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI,
        MetricCategory.CHOICE_SINGLE, MetricCategory.CHOICE_MULTI,
        MetricCategory.PAIRWISE,
    ],
    dimension=MetricDimension.FLUENCY,
)
def compute_garbled_text_rate(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """保守检测乱码/异常编码。空输出记 0,交给 empty_or_whitespace_rate 归因。"""
    scores, bad_indices, reasons_by_index = [], [], {}

    for idx, p in enumerate(preds):
        s = "" if p is None else str(p)
        if not s.strip():
            scores.append(0.0)
            continue

        reasons = _garbled_reasons(s)
        score = 1.0 if reasons else 0.0
        scores.append(score)
        if reasons:
            bad_indices.append(idx)
            reasons_by_index[str(idx)] = reasons

    garbled_count = int(sum(scores))
    return {
        "score": garbled_count / len(scores) if scores else 0.0,
        "details": scores,
        "artifacts": {
            "garbled_count": garbled_count,
            "total": len(scores),
            "garbled_indices": bad_indices,
            "reasons_by_index": reasons_by_index,
        },
    }


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

