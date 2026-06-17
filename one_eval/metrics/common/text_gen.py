from typing import List, Any, Dict
from one_eval.utils.extractor import normalize_text, AnswerExtractor
from rouge_score import rouge_scorer
import sacrebleu
from one_eval.core.metric_registry import register_metric, MetricCategory, MetricDimension

@register_metric(
    name="bleu",
    desc="sacreBLEU 主指标",
    usage="翻译/生成",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI],
    dimension=MetricDimension.SIMILARITY
)
def compute_bleu(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算 BLEU Score (Wrapper around sacrebleu).
    Args:
        preds: List of predicted strings.
        refs: List of reference strings (or list of list of references).
        kwargs:
            - tokenize: '13a' (default), 'zh', etc.
    """

    # sacrebleu expect refs to be List[List[str]] where inner list is parallel references
    # But our input refs might be List[str] or List[List[str]] (per sample)
    
    # Transpose refs: List[Sample_Refs] -> List[Ref1_List, Ref2_List, ...]
    # Determine max number of references
    max_refs = 0
    formatted_refs = []
    formatted_preds = [str(p) for p in preds]
    
    # First pass to find max refs
    clean_refs = []
    for r in refs:
        if isinstance(r, list):
            clean_refs.append([str(x) for x in r])
            max_refs = max(max_refs, len(r))
        else:
            clean_refs.append([str(r)])
            max_refs = max(max_refs, 1)
            
    # Pad and transpose
    transposed_refs = [[] for _ in range(max_refs)]
    for r_list in clean_refs:
        for i in range(max_refs):
            if i < len(r_list):
                transposed_refs[i].append(r_list[i])
            else:
                # Sacrebleu handles missing refs if we handle it right, but usually empty string might skew
                # For safety, repeat the last ref or use empty string? 
                # Sacrebleu recommends variable number of refs not to be padded with empty strings if possible
                # But corpus_bleu expects parallel lists of same length.
                # If we have variable refs, it's tricky.
                # Strategy: Pad with empty string, but this might lower score. 
                # Better Strategy: Just use the first reference for everyone if simplified, but that's wrong.
                # Correct Strategy: Pad with None or empty, sacrebleu might ignore?
                transposed_refs[i].append("") 
    
    tokenize = kwargs.get("tokenize", "13a")
    
    # Compute Corpus BLEU
    # sacrebleu.corpus_bleu(sys_stream, ref_streams)
    bleu = sacrebleu.corpus_bleu(formatted_preds, transposed_refs, tokenize=tokenize)
    
    return {
        "score": min(1.0, bleu.score / 100.0), # Convert 0-100 to 0-1 and clamp
        "details": [], # BLEU is corpus level usually
        "artifacts": {
            "sacrebleu_score": bleu.score,
            "counts": bleu.counts,
            "totals": bleu.totals,
            "precisions": bleu.precisions,
            "bp": bleu.bp,
            "sys_len": bleu.sys_len,
            "ref_len": bleu.ref_len
        }
    }

@register_metric(
    name="rouge_l",
    desc="ROUGE-L F1",
    usage="摘要/生成",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI],
    aliases=["rouge"],
    dimension=MetricDimension.SIMILARITY
)
def compute_rouge(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算 ROUGE Score (Wrapper around rouge-score).
    Default: ROUGE-L
    """

    rouge_types = kwargs.get("rouge_types", ["rougeL"])
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    scores = []
    details = []
    
    for p, r in zip(preds, refs):
        p_str = str(p)
        
        # Handle multi-ref: take max score
        r_list = r if isinstance(r, list) else [r]
        
        best_fmeasure = 0.0
        for gold in r_list:
            res = scorer.score(str(gold), p_str)
            # Usually we care about the primary metric, e.g. rougeL
            # res is Dict[str, Score(precision, recall, fmeasure)]
            current_f = res[rouge_types[0]].fmeasure
            if current_f > best_fmeasure:
                best_fmeasure = current_f
        
        scores.append(best_fmeasure)
        details.append(best_fmeasure)

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": details
    }

@register_metric(
    name="chrf",
    desc="CHRF Score",
    usage="翻译/生成",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI],
    dimension=MetricDimension.SIMILARITY
)
def compute_chrf(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算 CHRF Score.
    """
    
    # Prepare refs similar to BLEU, but chrf usually takes one list of refs or multiple?
    # sacrebleu.corpus_chrf(sys_stream, ref_streams)
    # Reuse logic from BLEU for transposing refs
    max_refs = 0
    clean_refs = []
    for r in refs:
        if isinstance(r, list):
            clean_refs.append([str(x) for x in r])
            max_refs = max(max_refs, len(r))
        else:
            clean_refs.append([str(r)])
            max_refs = max(max_refs, 1)
            
    transposed_refs = [[] for _ in range(max_refs)]
    for r_list in clean_refs:
        for i in range(max_refs):
            if i < len(r_list):
                transposed_refs[i].append(r_list[i])
            else:
                transposed_refs[i].append("")

    chrf = sacrebleu.corpus_chrf([str(p) for p in preds], transposed_refs)
    
    return {
        "score": chrf.score / 100.0,
        "details": [],
        "artifacts": {
            "chrf_score": chrf.score
        }
    }

@register_metric(
    name="token_f1",
    desc="token 级 F1 (匹配程度)",
    usage="长答案/部分匹配",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI],
    dimension=MetricDimension.SIMILARITY,
    aliases=["f1"]
)
def compute_token_f1(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算 Token-level F1 (SQuAD style).
    不依赖外部重型库。
    """
    scores = []
    
    for p, r in zip(preds, refs):
        p_str = str(p)
        r_list = r if isinstance(r, list) else [r]
        
        best_f1 = 0.0
        for gold in r_list:
            f1 = _compute_f1_single(p_str, str(gold))
            if f1 > best_f1:
                best_f1 = f1
        scores.append(best_f1)
        
    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores
    }

@register_metric(
    name="jaccard_similarity",
    desc="去重 token 集合 Jaccard 相似度",
    usage="关键词覆盖/检索式 QA/事实列表。只看去重 token 重叠，不看词频与语序",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI],
    dimension=MetricDimension.SIMILARITY,
    aliases=["jaccard"]
)
def compute_jaccard_similarity(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算 token 集合 Jaccard similarity: |pred ∩ ref| / |pred ∪ ref|。
    多参考答案时取最高分，与 token_f1/rouge_l 保持一致。
    """
    scores = []

    for p, r in zip(preds, refs):
        p_str = str(p)
        r_list = r if isinstance(r, list) else [r]
        scores.append(max(
            (_compute_jaccard_single(p_str, str(gold)) for gold in r_list),
            default=0.0,
        ))

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores
    }

def _compute_f1_single(prediction: str, truth: str) -> float:
    from collections import Counter
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 1.0 if pred_tokens == truth_tokens else 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def _compute_jaccard_single(prediction: str, truth: str) -> float:
    pred_tokens = set(normalize_text(prediction).split())
    truth_tokens = set(normalize_text(truth).split())

    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0

    union = pred_tokens | truth_tokens
    return len(pred_tokens & truth_tokens) / len(union)

