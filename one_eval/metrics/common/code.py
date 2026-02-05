from typing import List, Any, Dict
from one_eval.core.metric_registry import register_metric, MetricCategory

@register_metric(
    name="pass_at_k",
    desc="Pass@k (Code Execution)",
    usage="代码生成",
    categories=[MetricCategory.QA_SINGLE]
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
    categories=[MetricCategory.QA_SINGLE]
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

@register_metric(
    name="soft_code_execution",
    desc="静态代码分析 (语法/复杂度)",
    usage="代码生成 (无需沙箱)",
    categories=[MetricCategory.QA_SINGLE]
)
def compute_soft_code_execution(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """
    静态代码分析：
    1. 语法检查 (AST Parse)
    2. 基本复杂度分析
    """
    import ast
    
    scores = []
    details_list = []
    
    for p in preds:
        code_str = str(p)
        # Try to clean markdown code blocks
        if "```" in code_str:
            lines = code_str.split('\n')
            clean_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    clean_lines.append(line)
            if clean_lines:
                code_str = "\n".join(clean_lines)
            else:
                # Maybe it was just inline code or single block
                code_str = code_str.replace("```python", "").replace("```", "")

        try:
            # 1. Syntax Check
            tree = ast.parse(code_str)
            
            # 2. Complexity Heuristic
            # Count functions and classes
            func_count = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            class_count = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
            
            # Score logic:
            # - Syntax Valid: +0.5
            # - Has Function/Class: +0.5 (Assume valid code usually defines something)
            
            score = 0.5
            if func_count > 0 or class_count > 0:
                score += 0.5
            elif len(tree.body) > 0:
                # Script with body but no funcs
                score += 0.3
                
            scores.append(score)
            details_list.append({
                "valid": True, 
                "funcs": func_count, 
                "classes": class_count
            })
            
        except SyntaxError as e:
            scores.append(0.0)
            details_list.append({"valid": False, "error": str(e)})
        except Exception as e:
            scores.append(0.0)
            details_list.append({"valid": False, "error": f"Unknown error: {str(e)}"})

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores,
        "artifacts": details_list
    }
