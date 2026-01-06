from typing import List, Any, Dict, Optional
from one_eval.utils.extractor import extract_choice

def compute_choice_accuracy(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    计算选项准确率 (Choice Accuracy)
    
    Args:
        preds: 预测结果列表 (Raw strings)
        refs: 参考答案列表 (Raw strings or List of strings)
        kwargs: 
            - ignore_case: bool (default True)
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
        # ref 可能是 "A" 或者 ["A", "B"] (如果有多解)
        is_match = False
        
        if isinstance(r, list):
            # 如果 ref 是列表，说明有多个正确答案，命中任何一个都算对
            # 注意：这里假设 ref 列表里的已经是干净的 "A", "B" 等，或者需要再次 extract
            # 为了稳健，建议对 ref 也做一次 extract_choice 清洗
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

    # 4. 计算平均分
    score = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "score": score,
        "details": scores,
        "artifacts": {
            "pred_choices": pred_choices,
            "ref_choices": ref_choices
        }
    }
