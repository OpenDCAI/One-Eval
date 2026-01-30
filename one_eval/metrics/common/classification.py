from typing import List, Any, Dict, Union, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import numpy as np
from one_eval.utils.extractor import safe_float, AnswerExtractor
from one_eval.core.metric_registry import register_metric, MetricCategory

@register_metric(
    name="mcc",
    desc="Matthews Correlation Coefficient",
    usage="二分类/相关性任务 (GLUE)",
    categories=[MetricCategory.CHOICE_SINGLE],
    groups={
        "classification": "secondary"
    },
    priority="secondary"
)
def compute_mcc(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    Matthews Correlation Coefficient (MCC).
    Common in GLUE (CoLA).
    """
        
    try:
        # MCC needs clean labels. 
        # If preds/refs are strings "0"/"1" or ints, sklearn handles it if they match.
        # We assume preds/refs are already aligned labels.
        score = matthews_corrcoef(refs, preds)
        return {"score": float(score), "details": []}
    except Exception as e:
        return {"score": 0.0, "error": f"MCC failed: {str(e)}"}

@register_metric(
    name="pearson",
    desc="Pearson Correlation",
    usage="语义相似度/回归",
    categories=[MetricCategory.TEXT_SCORE],
    groups={
        "correlation": "secondary"
    },
    priority="secondary"
)
def compute_pearson(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    Pearson Correlation.
    Common in STS (Semantic Textual Similarity).
    Expects float inputs.
    """
        
    try:
        p_floats = [safe_float(x) or 0.0 for x in preds]
        r_floats = [safe_float(x) or 0.0 for x in refs]
        
        val, _ = pearsonr(p_floats, r_floats)
        return {"score": float(val), "details": []}
    except Exception as e:
        return {"score": 0.0, "error": f"Pearson failed: {str(e)}"}

@register_metric(
    name="spearman",
    desc="Spearman Correlation",
    usage="语义相似度/排序",
    categories=[MetricCategory.TEXT_SCORE],
    groups={
        "correlation": "secondary"
    },
    priority="secondary"
)
def compute_spearman(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    Spearman Correlation.
    """
        
    try:
        p_floats = [safe_float(x) or 0.0 for x in preds]
        r_floats = [safe_float(x) or 0.0 for x in refs]
        
        val, _ = spearmanr(p_floats, r_floats)
        return {"score": float(val), "details": []}
    except Exception as e:
        return {"score": 0.0, "error": f"Spearman failed: {str(e)}"}

@register_metric(
    name="auc_roc",
    desc="AUC-ROC ×100",
    usage="二分类/多分类任务",
    categories=[MetricCategory.CHOICE_SINGLE],
    groups={
        "auc_roc": "primary",
        "classification": "primary"
    },
    priority="primary"
)
def compute_auc_roc(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算 AUC-ROC。
    
    Args:
        preds: 预测概率列表。
               - 如果是二分类，应为 List[float] (positive class probs)。
               - 如果是多分类，应为 List[List[float]] (probs for each class) 或 List[Dict] (metadata)。
        refs: 真实标签列表。
        kwargs:
            - multi_class: 'ovr' (default) or 'ovo'
            - labels: List of labels for multiclass
    """
    
    try:
        # 1. 尝试标准化 preds 为数值
        clean_preds = []
        for p in preds:
            # Case A: 直接是 float (二分类概率)
            val = safe_float(p)
            if val is not None:
                clean_preds.append(val)
                continue
                
            # Case B: 是字典，尝试提取 'probs' 或 'logits'
            if isinstance(p, dict):
                # 假设 inference 阶段把概率存在了 "probs" 字段
                # 如果是多分类，这应该是一个 list
                if "probs" in p:
                    clean_preds.append(p["probs"])
                elif "prob" in p:
                    clean_preds.append(p["prob"])
                else:
                    # 无法提取，填 0.0 或报错
                    clean_preds.append(0.0)
                continue
            
            # Case C: 是 List (多分类概率分布)
            if isinstance(p, list):
                clean_preds.append(p)
                continue
                
            clean_preds.append(0.0) # Fallback

        # 2. 计算 AUC
        # 处理 refs, sklearn 期望 array-like
        # multi_class handling
        multi_class = kwargs.get("multi_class", "ovr")
        
        # 简单处理：如果是二分类，refs 应该是 0/1 或 False/True
        # 如果 refs 是字符串类别，需要 LabelEncoder，这里暂且假设用户已经处理好，或者 refs 是简单的 index
        
        score = roc_auc_score(refs, clean_preds, multi_class=multi_class)
        
        return {
            "score": float(score),
            "details": [], # AUC 是全局指标
        }
        
    except Exception as e:
        return {"score": 0.0, "error": f"AUC calculation failed: {str(e)}"}

@register_metric(
    name="accuracy",
    desc="通用 accuracy (当 evaluator 支持时)",
    usage="分类任务辅助指标",
    categories=[MetricCategory.CHOICE_SINGLE],
    aliases=["acc", "acc_norm"],
    groups={
        "auc_roc": "secondary",
        "classification": "secondary"
    },
    priority="secondary"
)
def compute_accuracy(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    通用 Accuracy (基于完全匹配)。
    不同于 choice_accuracy (会自动提取 A/B/C/D)，这个函数做简单的相等性检查。
    """
    scores = []
    for p, r in zip(preds, refs):
        # Handle multi-ref
        r_list = r if isinstance(r, list) else [r]
        
        # 只要命中一个 ref 就算对
        is_match = False
        p_str = str(p).strip()
        for gold in r_list:
            if p_str == str(gold).strip():
                is_match = True
                break
        
        scores.append(1.0 if is_match else 0.0)
        
    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores
    }

@register_metric(
    name="micro_f1",
    desc="多选集合 Micro-F1",
    usage="多选题/多标签分类",
    categories=[MetricCategory.CHOICE_MULTI],
    groups={
        "classification": "primary"
    },
    priority="primary"
)
def compute_micro_f1(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    Computes Micro-F1 for multi-select questions.
    Parses prediction string into a set of choices and compares with reference set.
    
    Ref: DataFlow UnifiedBenchDatasetEvaluator._eval_mc_multi
    """
    scores = []
    details = []
    
    # Assuming num_choices is passed in kwargs or we infer a safe upper bound (e.g. 26 for A-Z)
    num_choices = kwargs.get("num_choices", 26) 
    
    for p, r in zip(preds, refs):
        # Parse Reference
        gold_set = set()
        if isinstance(r, list):
            for x in r:
                if isinstance(x, int):
                    gold_set.add(x)
                elif isinstance(x, str) and len(x) == 1 and x.isalpha():
                    gold_set.add(ord(x.upper()) - ord("A"))
        elif isinstance(r, str):
             s_r = AnswerExtractor.parse_multiselect_set(r, num_choices)
             if s_r:
                 gold_set = s_r
        
        # Parse Prediction
        pred_set = AnswerExtractor.parse_multiselect_set(str(p), num_choices)
        if pred_set is None:
            pred_set = set()
            
        # Compute F1
        if not pred_set and not gold_set:
            f1 = 1.0
        else:
            inter = len(pred_set & gold_set)
            prec = inter / len(pred_set) if len(pred_set) > 0 else 0.0
            rec = inter / len(gold_set) if len(gold_set) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            
        scores.append(f1)
        details.append({
            "score": f1,
            "pred_set": list(pred_set),
            "gold_set": list(gold_set)
        })
        
    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": details
    }