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
                "仅选择 1 个主指标，其余尽可能多推荐可执行的辅/诊断指标。",
                "优先结合样例输出的格式特征（数值/符号/长文本）选择指标。",
                "若多种指标都可能适用，请说明取舍依据。"
            ],
        },
        {
            "condition": "生成式：多参考答案 (key2_q_ma): keys=[question,targets[]]",
            "rules": [
                "选择能支持多参考聚合的指标；优先从库中挑选最能反映任务目标的主指标。",
                "除主指标外，尽可能多推荐可执行的辅/诊断指标。"
            ],
        },
        {
            "condition": "选择题：单正确 (key3_q_choices_a): keys=[question,choices[],label]",
            "rules": [
                "从指标库中选择最能反映选择题正确性的主指标，并说明原因。",
                "除主指标外，尽可能多推荐可执行的辅/诊断指标。"
            ],
        },
        {
            "condition": "选择题：多正确 (key3_q_choices_as): keys=[question,choices[],labels[]]",
            "rules": [
                "选择支持多选/多标签的指标作为主指标，并尽可能多补充诊断项。"
            ],
        },
        {
            "condition": "偏好/排序：成对比较 (key3_q_a_rejected): keys=[question,better,rejected]",
            "rules": [
                "选择能反映偏好/排序稳定性的主指标，并尽可能多补充诊断指标。"
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