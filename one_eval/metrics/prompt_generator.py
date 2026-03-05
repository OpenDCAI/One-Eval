# one_eval/metrics/prompt_generator.py
from typing import List, Dict, Any
import json
from one_eval.core.metric_registry import MetricMeta, MetricCategory

class MetricPromptGenerator:
    """
    负责生成用于 Prompt 的指标相关文档。
    """
    
    DECISION_RULES = [
        {
            "condition": "通用前提",
            "rules": [
                "本步骤输入来自上一步 inference 的输出：包含 BenchInfo/meta，以及落盘的 predict 与 ground truth 内容。",
                "优先使用 meta 中的 `bench_dataflow_eval_type` 来判定评测类型；若缺失则根据样本字段(schema)推断。"
            ],
        },
        {
            "condition": "文本打分 (key1_text_score): keys=[text]",
            "rules": [
                "若评测是对输出文本本身打分/检测 -> 使用 `llm_judge_score` / `toxicity_max` / `truth_score` 这类 score 型指标。"
            ],
        },
        {
            "condition": "生成式：单参考答案 (key2_qa): keys=[question,target]",
            "rules": [
                "默认 -> `exact_match`(primary) + `extraction_rate`(diagnostic)，必要时补 `f1`(secondary)。",
                "若答案是数值/算术 -> `numerical_match`(primary)；若含 LaTeX/符号推导 -> `symbolic_match`(primary) + `strict_match`(secondary)。",
                "若是摘要/翻译类 -> `rouge_l` 或 `bleu` 作为主指标。"
            ],
        },
        {
            "condition": "生成式：多参考答案 (key2_q_ma): keys=[question,targets[]]",
            "rules": [
                "使用与单参考相同的指标族，但 evaluator 需支持多参考（多 gold）聚合。",
                "默认仍建议加 `extraction_rate` 监控答案提取/对齐成功率。"
            ],
        },
        {
            "condition": "选择题：单正确 (key3_q_choices_a): keys=[question,choices[],label]",
            "rules": [
                "使用 `choice_accuracy`(primary)；诊断可加 `missing_answer_rate`。",
                "二分类/多分类可选 `auc_roc`(primary) + `accuracy`(secondary)。"
            ],
        },
        {
            "condition": "选择题：多正确 (key3_q_choices_as): keys=[question,choices[],labels[]]",
            "rules": [
                "优先选择支持多标签/多选规则的实现；当前 registry 以 `choice_accuracy`/`missing_answer_rate` 占位，具体聚合规则由 evaluator 定义。"
            ],
        },
        {
            "condition": "偏好/排序：成对比较 (key3_q_a_rejected): keys=[question,better,rejected]",
            "rules": [
                "使用 `win_rate_against_baseline`(primary)，含义为 pairwise preference 的取胜率。"
            ],
        },
    ]

    EVAL_TYPE_SPECS = {
        MetricCategory.TEXT_SCORE: {"title": "文本打分"},
        MetricCategory.QA_SINGLE: {"title": "生成式：单参考答案"},
        MetricCategory.QA_MULTI: {"title": "生成式：多参考答案"},
        MetricCategory.CHOICE_SINGLE: {"title": "选择题：单正确"},
        MetricCategory.CHOICE_MULTI: {"title": "选择题：多正确"},
        MetricCategory.PAIRWISE: {"title": "偏好/排序：成对比较"},
    }

    @classmethod
    def get_decision_logic_doc(cls) -> str:
        """
        动态生成 Prompt 中的 '决策逻辑' 文档
        """
        doc_lines = []
        for idx, item in enumerate(cls.DECISION_RULES, 1):
            doc_lines.append(f"{idx}. **若是 {item['condition']}**：")
            for rule in item['rules']:
                doc_lines.append(f"   - {rule}")
        return "\n".join(doc_lines)

    @classmethod
    def get_metric_library_doc(cls, metas: List[MetricMeta]) -> str:
        """
        动态生成 Prompt 中的 '支持的指标库' 文档。
        """
        definitions_by_id: Dict[str, List[Dict[str, Any]]] = {k: [] for k in cls.EVAL_TYPE_SPECS.keys()}
        
        # Group metrics by category
        for meta in metas:
            metric_entry = {
                "name": meta.name,
                "desc": meta.desc,
                "usage": meta.usage,
                "args": {} # Placeholder if we want to support args introspection later
            }
            
            # Ensure at least one category to avoid missing metrics
            categories = meta.categories if meta.categories else ["Uncategorized"]
            
            for category_id in categories:
                if category_id not in definitions_by_id:
                    definitions_by_id[category_id] = []
                
                # Check for existing to avoid duplicates if categories overlap weirdly
                existing_list = definitions_by_id[category_id]
                existing_idx = next((i for i, x in enumerate(existing_list) if x["name"] == meta.name), -1)
                
                if existing_idx != -1:
                    existing_list[existing_idx] = metric_entry
                else:
                    existing_list.append(metric_entry)

        # Build display dictionary with titles
        final_definitions = {}
        for key_id, metrics in definitions_by_id.items():
            if key_id in cls.EVAL_TYPE_SPECS:
                title = cls.EVAL_TYPE_SPECS[key_id]["title"]
                display_key = f"{title} ({key_id})"
                final_definitions[display_key] = metrics
            else:
                final_definitions[key_id] = metrics

        # Generate markdown
        doc_lines = []
        idx = 1
        for category, metrics in final_definitions.items():
            if not metrics:
                continue
            doc_lines.append(f"{idx}. **{category}**")
            for m in metrics:
                line = f"   - `{m['name']}`: {m['desc']}"
                if "usage" in m:
                    line += f" [适用: {m['usage']}]"
                if m.get("args"):
                    line += f" (默认参数: {json.dumps(m['args'])})"
                doc_lines.append(line)
            doc_lines.append("") # Empty line separator
            idx += 1
        return "\n".join(doc_lines)