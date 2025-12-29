from __future__ import annotations
from typing import List, Dict, Any, Optional
from one_eval.logger import get_logger
import re, json

log = get_logger(__name__)

class MetricRegistry:
    """
    指标注册表：存储已知数据集的标准评测指标配置。
    
    改进说明：
    1. 细化了 accuracy 的类型 (numerical vs symbolic vs choice)。
    2. 为每个数据集增加了 'extraction_rate' 作为诊断指标。
    3. 区分了 strict_match (格式严格) 和 soft_match (数值/逻辑正确)。
    """

    def __init__(self):
        """根据 opencompass 的 evaluator 粒度，预定义一批 metric 模板。"""
        # --- 初始化metric定义 ---
        self._definitions = {
            "数学与逻辑 (Math & Logic)": [
                {"name": "numerical_match", "desc": "数值软匹配(1.0 == 1，容忍浮点误差)", "usage": "算术题、小学数学"},
                {"name": "symbolic_match", "desc": "SymPy / LaTeX 等价性校验", "usage": "高等数学、代数推导"},
                {"name": "strict_match", "desc": "原始字符串严格匹配", "usage": "格式严格的任务"}
            ],
            "选择与分类 (Choice & Classification)": [
                {"name": "choice_accuracy", "desc": "选项字母或离散标签准确率", "usage": "选择题 (A/B/C/D)"},
                {"name": "missing_answer_rate", "desc": "未输出有效选项/标签的比例 (诊断用)", "usage": "监控模型拒答率"},
                {"name": "auc_roc", "desc": "AUC-ROC ×100", "usage": "二分类/多分类任务"}
            ],
            "代码生成 (Code Generation)": [
                {"name": "pass_at_k", "desc": "代码通过率 (需指定 k)", "args": {"k": 1}, "usage": "HumanEval, MBPP"}
            ],
            "文本生成与摘要 (Generation & Summarization)": [
                {"name": "rouge_l", "desc": "ROUGE-L F1", "usage": "摘要、翻译"},
                {"name": "bleu", "desc": "sacreBLEU 主指标", "usage": "翻译任务"},
                {"name": "bert_score", "desc": "语义相似度", "usage": "开放生成 (可选)"}
            ],
            "问答 (QA)": [
                {"name": "exact_match", "desc": "抽取式答案完全匹配 (EM)", "usage": "SQuAD, 抽取式QA"},
                {"name": "f1", "desc": "token 级 F1 (匹配程度)", "usage": "长上下文 QA"},
                {"name": "llm_judge_score", "desc": "LLM 裁判打分 (0~100)", "usage": "开放式主观问答"}
            ],
            "长文本与检索 (Long Context & Retrieval)": [
                {"name": "retrieval_accuracy", "desc": "是否检索到正确段落/索引", "usage": "RAG, NeedleBench"},
                {"name": "count_accuracy", "desc": "计数覆盖精度", "usage": "计数类任务"}
            ],
            "安全与评估 (Safety & Evaluation)": [
                {"name": "toxicity_max", "desc": "样本毒性分数的最大值", "usage": "安全检测"},
                {"name": "truth_score", "desc": "真实度评分 (LLM-based)", "usage": "幻觉检测"}
            ],
            "通用诊断 (General Diagnostic)": [
                {"name": "extraction_rate", "desc": "正则提取成功率 (强烈建议)", "usage": "所有非选择题任务"}
            ]
        }
        # --- 初始化metric模板 ---
        self._templates = {
            "numerical": [
                {"name": "numerical_match", "priority": "primary"},
                {"name": "strict_match", "priority": "secondary"},
                {"name": "extraction_rate", "priority": "diagnostic"}
            ],
            "symbolic": [
                {"name": "symbolic_match", "priority": "primary"},
                {"name": "strict_match", "priority": "secondary"},
                {"name": "extraction_rate", "priority": "diagnostic"}
            ],
            "choice": [
                {"name": "choice_accuracy", "priority": "primary"},
                {"name": "missing_answer_rate", "priority": "diagnostic"}
            ],
            "code": [
                {"name": "pass_at_k", "k": 1, "priority": "primary"},
                {"name": "pass_at_k", "k": 5, "priority": "secondary"}
            ],
            "generation_rouge": [
                {"name": "rouge_l", "priority": "primary"},
                {"name": "rouge_1", "priority": "secondary"},
                {"name": "rouge_2", "priority": "secondary"}
            ],
            "generation_bleu": [
                {"name": "bleu", "priority": "primary"}
            ],
            "qa_extractive": [
                {"name": "exact_match", "priority": "primary"},
                {"name": "f1", "priority": "secondary"}
            ],
            "long_context_qa": [
                {"name": "f1", "priority": "primary"},
                {"name": "exact_match", "priority": "secondary"}
            ],
            "retrieval": [
                {"name": "retrieval_accuracy", "priority": "primary"}
            ],
            "count": [
                {"name": "count_accuracy", "priority": "primary"}
            ],
            "code_sim": [
                {"name": "code_similarity", "priority": "primary"}
            ],
            "llm_judge": [
                {"name": "llm_judge_score", "priority": "primary"},
            ],
            "win_rate": [
                {"name": "win_rate_against_baseline", "priority": "primary"},
            ],
            "auc_roc": [
                {"name": "auc_roc", "priority": "primary"},
                {"name": "accuracy", "priority": "secondary"}
            ]    
        }

        # --- 初始化注册表：按 opencompass 数据集族映射到上面的模板 ---
        self._registry: Dict[str, List[Dict[str, Any]]] = {
            # --- Group A: 数值计算 (Arithmetic / Numerical) ---            
            "gsm8k": self._templates["numerical"],
            "svamp": self._templates["numerical"],
            "calc-ape210k": self._templates["numerical"],
            "calc-mawps": self._templates["numerical"],
            "calc-asdiv_a": self._templates["numerical"],
            
            # --- Group B: 符号与高难度数学 (Symbolic / Hard Math) ---            
            "math": self._templates["symbolic"],
            "hendrycks_math": self._templates["symbolic"],
            "math-500": self._templates["symbolic"],
            "competition_math": self._templates["symbolic"],
            
            # --- Group C: 选择题 (Multiple Choice / Classification) ---
            "aqua-rat": self._templates["choice"],
            "mmlu": self._templates["choice"],
            "agieval-gaokao-mathqa": self._templates["choice"],
            "math-qa": self._templates["choice"], # MathQA 虽然有步骤，但常作为选择题评测

            # --- Group D: 代码 (Code) ---
            "humaneval": self._templates["code"],
            "mbpp": self._templates["code"],

            # --- Group E: 通用文本生成 / 摘要 / QA ---
            "general_qa": self._templates["generation_rouge"],
            "summscreen": self._templates["generation_rouge"],
            "lcsts": self._templates["generation_rouge"],
            "iwslt2017": self._templates["generation_bleu"],
            "flores": self._templates["generation_bleu"],

            # 抽取式 QA / span-based QA
            "squad20": self._templates["qa_extractive"],
            "tydiqa": self._templates["qa_extractive"],
            "nq": self._templates["qa_extractive"],
            "nq_cn": self._templates["qa_extractive"],
            "qasper": self._templates["qa_extractive"],

            # LongBench / LV-Eval / Omni 长上下文 QA & 相关任务
            "longbench": self._templates["long_context_qa"],
            "lveval": self._templates["long_context_qa"],

            # --- Group F: 检索 / 计数 / 长上下文结构化任务 ---
            "needlebench": self._templates["retrieval"],
            "needlebench_v2": self._templates["retrieval"],
            "longbench_retrieval": self._templates["retrieval"],
            "longbench_count": self._templates["count"],
            "longbench_codesim": self._templates["code_sim"],

            # --- Group H: LLM 裁判 / 主观评测 ---
            # MT-Bench / Arena / Subjective 族 & LEval / LV-Eval / TEval
            "subjective": self._templates["llm_judge"],
            "arena": self._templates["llm_judge"],
            "mtbench": self._templates["llm_judge"],
            "promptbench": self._templates["llm_judge"],
            "leval": self._templates["win_rate"],
            "teval": self._templates["llm_judge"],
            "omni_math_judge": self._templates["llm_judge"],

            # --- Group I: AUC / 其他分类指标 ---
            "llm_compression": self._templates["auc_roc"],

        }
        self._decision_rules = [
            {
                "condition": "数学/计算题 (Math/Arithmetic)",
                "rules": [
                    "简单算术/应用题 -> 推荐 `numerical_match` (primary) + `extraction_rate` (diagnostic)",
                    "复杂公式/竞赛数学 (含 LaTeX) -> 推荐 `symbolic_match` (primary) + `strict_match` (secondary)"
                ]
            },
            {
                "condition": "选择题 (Multiple Choice)",
                "rules": [
                    "推荐 `choice_accuracy` (primary) + `missing_answer_rate` (diagnostic)"
                ]
            },
            {
                "condition": "代码题 (Code Generation)",
                "rules": [
                    "推荐 `pass_at_k` (k=1) (primary) + `pass_at_k` (k=5 或 10) (secondary)",
                    "注意：`args` 必须显式写出，例如 `{'k': 1}`"
                ]
            },
            {
                "condition": "抽取式QA (Extractive QA)",
                "rules": [
                    "推荐 `exact_match` (primary) + `f1` (secondary)"
                ]
            },
            {
                "condition": "长文本QA / 摘要 (Long Context/Summarization)",
                "rules": [
                    "推荐 `rouge_l` (primary) 或 `f1` (针对 LongBench 类)"
                ]
            },
            {
                "condition": "检索任务 (Retrieval)",
                "rules": [
                    "推荐 `retrieval_accuracy` (primary)"
                ]
            },
            {
                "condition": "开放式主观问答 (Open-ended)",
                "rules": [
                    "推荐 `llm_judge_score` (primary)"
                ]
            },
            {
                "condition": "安全/毒性检测 (Safety)",
                "rules": [
                    "推荐 `toxicity_max` (primary) + `toxicity_rate` (diagnostic)"
                ]
            }
        ]
    
    def get_decision_logic_doc(self) -> str:
        """
        动态生成 Prompt 中的 '决策逻辑' 文档
        """
        doc_lines = []
        for idx, item in enumerate(self._decision_rules, 1):
            doc_lines.append(f"{idx}. **若是 {item['condition']}**：")
            for rule in item['rules']:
                doc_lines.append(f"   - {rule}")
        return "\n".join(doc_lines)
    
    def get_metric_library_doc(self) -> str:
        """
        动态生成 Prompt 中的 '支持的指标库' 文档。
        """
        doc_lines = []
        idx = 1
        for category, metrics in self._definitions.items():
            doc_lines.append(f"{idx}. **{category}**")
            for m in metrics:
                # 格式化: - `name`: desc (适用场景)
                line = f"   - `{m['name']}`: {m['desc']}"
                if "usage" in m:
                    line += f" [适用: {m['usage']}]"
                if "args" in m:
                    line += f" (默认参数: {json.dumps(m['args'])})"
                doc_lines.append(line)
            doc_lines.append("") # 空行分隔
            idx += 1
        return "\n".join(doc_lines)

    def register(self, dataset_name: str, metrics: List[Dict[str, Any]]):
        """动态注册或覆盖某个数据集的指标"""
        self._registry[dataset_name.lower()] = metrics
        log.info(f"[MetricRegistry] 已注册/更新数据集 '{dataset_name}' 的指标配置")

    def _normalize_key(self, key: str) -> str:
        """
        标准化字符串：
        1. 转小写
        2. 将所有非字母数字字符替换为下划线 '_'
        3. 关键：前后加下划线，形成封闭边界，防止子串误匹配
        
        Example: 
            'math' -> '_math_'
            'math-500' -> '_math_500_'
            'openai/gsm8k' -> '_openai_gsm8k_'
        """
        # 将所有非字母数字 (a-z, 0-9) 替换为 '_'
        clean_key = re.sub(r'[^a-z0-9]', '_', key.lower())
        # 去除多余的连续下划线
        clean_key = re.sub(r'_+', '_', clean_key).strip('_')
        # 添加边界保护
        return f"_{clean_key}_"

    def get_metrics(self, dataset_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        获取数据集的指标配置。
        如果找不到匹配项，返回 None，而不是返回默认值。
        """
        raw_name = dataset_name.lower().strip()
        
        # 1. 精确匹配
        if raw_name in self._registry:
            return self._registry[raw_name]
        
        # 2. 模糊匹配
        normalized_input = self._normalize_key(raw_name)
        matched_metrics = []
        best_match_len = 0
        
        for key, metrics in self._registry.items():
            normalized_key = self._normalize_key(key)
            if normalized_key in normalized_input:
                if len(key) > best_match_len:
                    best_match_len = len(key)
                    matched_metrics = metrics
        
        if matched_metrics:
            log.info(f"[MetricRegistry] 模糊匹配成功: '{dataset_name}'")
            return matched_metrics

        # 3. 彻底没找到 -> 返回 None (删除原来的 Default Fallback)
        return None 


# 全局单例
metric_registry = MetricRegistry()
