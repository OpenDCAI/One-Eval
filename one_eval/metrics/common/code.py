"""
code.py —— 代码生成维度指标。

说明:真正的 Pass@k 需要沙箱执行(有安全风险且依赖环境),本仓库不内置。
这里只保留无需沙箱、确定可跑的静态分析指标。需要 Pass@k 时由用户在受控环境
自定义实现(见 skill custom_metrics 机制)。
"""
from typing import List, Any, Dict
from one_eval.core.metric_registry import register_metric, MetricCategory, MetricDimension


@register_metric(
    name="code_validity",
    desc="代码合法性 (AST 可解析 + 是否定义函数/类)",
    usage="代码生成,无沙箱。注意:只验“能不能解析/有没有结构”,不代表逻辑正确。要真 Pass@k 请在受控环境自写 custom metric",
    categories=[MetricCategory.QA_SINGLE],
    aliases=["soft_code_execution"],
    dimension=MetricDimension.VALIDITY
)
def compute_code_validity(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """语法检查(AST 可解析) + 结构启发式打分。衡量“产物是否是合法可用的代码”,非正确性。"""
    import ast

    scores, details_list = [], []
    for p in preds:
        code_str = str(p)
        if "```" in code_str:
            lines, clean_lines, in_block = code_str.split("\n"), [], False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    clean_lines.append(line)
            code_str = "\n".join(clean_lines) if clean_lines else \
                code_str.replace("```python", "").replace("```", "")

        try:
            tree = ast.parse(code_str)
            func_count = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
            class_count = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
            score = 0.5
            if func_count > 0 or class_count > 0:
                score += 0.5
            elif len(tree.body) > 0:
                score += 0.3
            scores.append(score)
            details_list.append({"valid": True, "funcs": func_count, "classes": class_count})
        except SyntaxError as e:
            scores.append(0.0)
            details_list.append({"valid": False, "error": str(e)})
        except Exception as e:
            scores.append(0.0)
            details_list.append({"valid": False, "error": f"Unknown error: {str(e)}"})

    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "details": scores,
        "artifacts": details_list,
    }
