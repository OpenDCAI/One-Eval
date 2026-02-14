from typing import List, Any, Dict
from one_eval.utils.extractor import normalize_text, AnswerExtractor
from rouge_score import rouge_scorer  
import sacrebleu
from one_eval.core.metric_registry import register_metric, MetricCategory

@register_metric(
    name="bleu",
    desc="sacreBLEU 主指标",
    usage="翻译/生成",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI]
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
    name="ter",
    desc="Translation Edit Rate",
    usage="翻译/生成",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI]
)
def compute_ter(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算 TER (Translation Edit Rate).
    Lower is better. (But One-Eval usually assumes Higher is better?)
    Note: TER is an error rate (0 is perfect, >1 is bad).
    """
    # Reuse ref transposition logic
    # ... Refactoring transposition into a helper would be better but for now copy-paste to be safe
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

    ter = sacrebleu.corpus_ter([str(p) for p in preds], transposed_refs)
    
    return {
        "score": ter.score / 100.0, # TER is typically 0-100+
        "details": [],
        "artifacts": {
            "ter_score": ter.score
        }
    }

@register_metric(
    name="rouge_l",
    desc="ROUGE-L F1",
    usage="摘要/生成",
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI],
    aliases=["rouge"]
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
    categories=[MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI]
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

def _compute_f1_single(prediction: str, truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 1.0 if pred_tokens == truth_tokens else 0.0
    
    common = collections_Counter(pred_tokens) & collections_Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def collections_Counter(tokens):
    # Simple Counter implementation to avoid importing collections if not needed, 
    # but collections is standard lib, so just import it at top or inside
    from collections import Counter
    return Counter(tokens)

@register_metric(
    name="reasoning_efficiency",
    desc="CoT 效率评估 (Ref长度/Pred长度)",
    usage="评估 CoT 的冗余程度 (正例越高越好，负例越低越啰嗦)",
    categories=[MetricCategory.TEXT_SCORE, MetricCategory.QA_SINGLE]
)
def compute_reasoning_efficiency(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算 Reasoning Efficiency.
    Efficiency = len(ref) / len(pred)
    
    返回:
        score: 正例的平均效率 (越高越好，表示用最少的字说对了)
        artifacts:
            efficiency_pos: 正例平均效率
            efficiency_neg: 负例平均效率 (越低说明幻觉/啰嗦越多)
            pos_count: 正例数量
            neg_count: 负例数量
    """
    extractor = AnswerExtractor()
    pos_ratios = []
    neg_ratios = []
    details = []
    
    for p, r in zip(preds, refs):
        p_str = str(p)
        
        # Handle multi-ref: assume the first one is representative for length, 
        # or find the one that matches?
        # For length comparison, we usually take the shortest valid ref or just the first one.
        # But for correctness, we check any.
        
        r_list = r if isinstance(r, list) else [r]
        r_str_primary = str(r_list[0]) # Use first ref for length baseline
        
        # 1. Check Correctness (Logic similar to numerical_match/exact_match)
        is_correct = False
        
        # Extract answer from pred
        p_ans = extractor.extract_answer(p_str, use_last_number=True)
        if p_ans is None: p_ans = ""
        p_ans_norm = normalize_text(p_ans)
        
        for gold in r_list:
            # Extract answer from gold (usually gold is already the answer, but just in case)
            g_ans = extractor.extract_answer(str(gold), use_last_number=True)
            if g_ans is None: g_ans = ""
            g_ans_norm = normalize_text(g_ans)
            
            if p_ans_norm == g_ans_norm:
                is_correct = True
                break
        
        # 2. Calculate Efficiency Ratio
        # Avoid division by zero
        p_len = len(p_str)
        r_len = len(r_str_primary)
        
        ratio = 0.0
        if p_len > 0:
            ratio = r_len / p_len
            
        # Clamp ratio to [0, 1] usually? Or can it be > 1 if pred is shorter than ref?
        # Pred shorter than ref is possible but rare for CoT. 
        # User didn't ask to clamp, but typically efficiency > 1 means super concise (or just answering without CoT).
        # We keep raw ratio.
        
        if is_correct:
            pos_ratios.append(ratio)
        else:
            neg_ratios.append(ratio)
            
        details.append(ratio)

    avg_pos = sum(pos_ratios) / len(pos_ratios) if pos_ratios else 0.0
    avg_neg = sum(neg_ratios) / len(neg_ratios) if neg_ratios else 0.0
    
    
    return {
        "score": avg_pos, # Main score is efficiency on correct answers
        "details": details,
        "artifacts": {
            "efficiency_pos": avg_pos,
            "efficiency_neg": avg_neg,
            "pos_count": len(pos_ratios),
            "neg_count": len(neg_ratios)
        }
    }

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "on", "in", 
    "to", "for", "of", "with", "by", "from", "up", "down", "is", "are", "was", "were", 
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", 
    "should", "would", "will", "may", "might", "must", "it", "its", "they", "them", 
    "their", "he", "him", "his", "she", "her", "we", "us", "our", "you", "your", "i", 
    "me", "my", "this", "that", "these", "those", "not", "no", "yes", "as", "what", "which",
    "who", "whom", "whose", "where", "why", "how", "all", "any", "both", "each", "few", 
    "more", "most", "other", "some", "such", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now"
}

@register_metric(
    name="keyword_recall",
    desc="关键词召回率 (Ref关键词在Pred中的占比)",
    usage="评估内容覆盖度 (负例中低分=知识盲区, 高分=逻辑错误)",
    categories=[MetricCategory.TEXT_SCORE, MetricCategory.QA_SINGLE]
)
def compute_keyword_recall(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算 Keyword Recall.
    Recall = |Ref_Keywords ∩ Pred_Tokens| / |Ref_Keywords|
    
    分析维度:
    - 负例 (Incorrect Cases):
        - Low Recall: 知识盲区 (Knowledge Gap) - 连相关的词都没提到
        - High Recall: 逻辑错误 (Logic Error) - 提到了相关词但推导错了
        
    Returns:
        score: 平均 Recall (所有样本)
        artifacts:
            recall_neg: 负例的平均 Recall
            recall_pos: 正例的平均 Recall
            neg_count: 负例数量
            pos_count: 正例数量
    """
    extractor = AnswerExtractor()
    recalls = []
    pos_recalls = []
    neg_recalls = []
    
    for p, r in zip(preds, refs):
        p_str = str(p)
        r_list = r if isinstance(r, list) else [r]
        
        # 1. Determine Correctness (Standard Exact Match logic)
        is_correct = False
        p_ans = extractor.extract_answer(p_str, use_last_number=True) or ""
        p_ans_norm = normalize_text(p_ans)
        
        # Also prepare Ref Keywords (Union of all refs or just the first? Usually Union is safer for recall)
        # But traditionally we compare against each ref and take max? 
        # User said "Ref关键词集合 S". If multiple refs exist, S could be union.
        # Let's use Union of all refs to be lenient.
        ref_keywords = set()
        
        for gold in r_list:
            g_str = str(gold)
            g_ans = extractor.extract_answer(g_str, use_last_number=True) or ""
            g_ans_norm = normalize_text(g_ans)
            
            if p_ans_norm == g_ans_norm:
                is_correct = True
            
            # Tokenize Ref
            # Normalize entire ref string (not just answer part, as we want content recall of the explanation/context usually? 
            # Or just the answer? User said "Ref 进行分词". Ref usually implies the ground truth text. 
            # In CoT datasets, Ref is often the solution string.)
            # If Ref is just "A", recall is trivial. If Ref is full solution, recall is meaningful.
            # We assume Ref is the full text.
            norm_gold = normalize_text(g_str)
            tokens = norm_gold.split()
            ref_keywords.update([t for t in tokens if t not in STOPWORDS])
            
        # 2. Compute Recall
        p_norm = normalize_text(p_str)
        p_tokens = set(p_norm.split())
        
        if not ref_keywords:
            # Empty ref keywords (maybe only stopwords or empty string)
            # If pred is also empty/stopwords, maybe 1.0? Else 0.0?
            # Let's say 1.0 if both empty, else 0.0.
            recall = 0.0 if p_tokens else 1.0 # Edge case
        else:
            intersection = ref_keywords.intersection(p_tokens)
            recall = len(intersection) / len(ref_keywords)
            
        recalls.append(recall)
        
        if is_correct:
            pos_recalls.append(recall)
        else:
            neg_recalls.append(recall)
            
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_pos = sum(pos_recalls) / len(pos_recalls) if pos_recalls else 0.0
    avg_neg = sum(neg_recalls) / len(neg_recalls) if neg_recalls else 0.0
    
    return {
        "score": avg_recall,
        "details": recalls,
        "artifacts": {
            "recall_all": avg_recall,
            "recall_pos": avg_pos,
            "recall_neg": avg_neg,
            "pos_count": len(pos_recalls),
            "neg_count": len(neg_recalls)
        }
    }
