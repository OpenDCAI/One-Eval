from __future__ import annotations
from typing import List, Dict, Any, Optional
from one_eval.logger import get_logger
import re

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

        # 1. 数值计算类 (GSM8K, SVAMP 等)
        metrics_numerical = [
            {"name": "numerical_match", "priority": "primary", "desc": "数值软匹配(1.0 == 1，容忍浮点误差)"},
            {"name": "strict_match", "priority": "secondary", "desc": "原始字符串严格匹配"},
            {"name": "extraction_rate", "priority": "diagnostic", "desc": "成功抽取出数值答案的比例"}
        ]

        # 2. 符号/公式类 (MATH, Omni-MATH 等)
        metrics_symbolic = [
            {"name": "symbolic_match", "priority": "primary", "desc": "SymPy / LaTeX 等价性校验"},
            {"name": "strict_match", "priority": "secondary", "desc": "LaTeX 字符串严格匹配"},
            {"name": "extraction_rate", "priority": "diagnostic", "desc": "从 \\boxed{} 或表达式中正确抽取答案的比例"}
        ]

        # 3. 选择题 / 分类类 (MMLU, RACE, HellaSwag 等)
        metrics_choice = [
            {"name": "choice_accuracy", "priority": "primary", "desc": "选项字母或离散标签准确率"},
            {"name": "missing_answer_rate", "priority": "diagnostic", "desc": "未输出有效选项/标签的比例"}
        ]

        # 4. 代码类 (HumanEval, MBPP, LiveCodeBench, CodeCompass 等)
        metrics_code = [
            {"name": "pass_at_k", "k": 1, "priority": "primary"},
            {"name": "pass_at_k", "k": 5, "priority": "secondary"}
        ]

        # 5. 摘要 / 通用生成类 (Rouge / Bleu)
        metrics_generation_rouge = [
            {"name": "rouge_l", "priority": "primary", "desc": "ROUGE-L F1"},
            {"name": "rouge_1", "priority": "secondary", "desc": "ROUGE-1 F1，可选"},
            {"name": "rouge_2", "priority": "secondary", "desc": "ROUGE-2 F1，可选"}
        ]
        metrics_generation_bleu = [
            {"name": "bleu", "priority": "primary", "desc": "sacreBLEU 主指标"}
        ]

        # 6. 抽取式 QA / 长上下文 QA (SQuAD, LongBench F1 等)
        metrics_qa_extractive = [
            {"name": "exact_match", "priority": "primary", "desc": "抽取式答案完全匹配 (EM)"},
            {"name": "f1", "priority": "secondary", "desc": "token 级 F1 (匹配程度)"}
        ]
        metrics_long_context_qa = [
            {"name": "f1", "priority": "primary", "desc": "长上下文问答 F1 (LongBench / LV-Eval)"},
            {"name": "exact_match", "priority": "secondary", "desc": "可选的 EM 统计"}
        ]

        # 7. 检索 / 计数 / 长上下文结构化任务 (LongBench Retrieval / Count / CodeSim)
        metrics_retrieval = [
            {"name": "retrieval_accuracy", "priority": "primary", "desc": "是否检索到正确段落/索引"}
        ]
        metrics_count = [
            {"name": "count_accuracy", "priority": "primary", "desc": "计数题中，预测数字集合覆盖真值的精度"}
        ]
        metrics_code_sim = [
            {"name": "code_similarity", "priority": "primary", "desc": "代码字符串相似度 (如 Fuzzy 匹配)"}
        ]

        # 8. 安全 / 毒性 (RealToxicPrompts / 安全相关数据集)
        metrics_toxicity = [
            {"name": "toxicity_max", "priority": "primary", "desc": "样本毒性分数的最大值"},
            {"name": "toxicity_avg", "priority": "secondary", "desc": "毒性分数均值"},
            {"name": "toxicity_rate", "priority": "diagnostic", "desc": "毒性分数超过阈值的样本比例"}
        ]

        # 9. LLM 作为裁判的主观评测 (MT-Bench, Arena, LEval, LV-Eval, TEval 等)
        metrics_llm_judge = [
            {"name": "llm_judge_score", "priority": "primary", "desc": "LLM 裁判打分或胜率 (0~100)"},
        ]
        metrics_win_rate = [
            {"name": "win_rate_against_baseline", "priority": "primary", "desc": "相对于基线模型的胜率 (如 LEvalGPTEvaluator)"},
        ]

        # 10. AUC / 其他分类指标 (AUCROCEvaluator 等)
        metrics_auc_roc = [
            {"name": "auc_roc", "priority": "primary", "desc": "AUC-ROC ×100"},
            {"name": "accuracy", "priority": "secondary", "desc": "基于 argmax 的分类准确率 ×100"}
        ]

        # --- 初始化注册表：按 opencompass 数据集族映射到上面的模板 ---
        self._registry: Dict[str, List[Dict[str, Any]]] = {
            # --- Group A: 数值计算 (Arithmetic / Numerical) ---
            "gsm8k": metrics_numerical,
            "svamp": metrics_numerical,
            "calc-ape210k": metrics_numerical,
            "calc-mawps": metrics_numerical,
            "calc-asdiv_a": metrics_numerical,

            # --- Group B: 符号与高难度数学 (Symbolic / Hard Math) ---
            "hendrycks_math": metrics_symbolic,
            "math": metrics_symbolic,          # 别名
            "math-500": metrics_symbolic,      # 子集
            "competition_math": metrics_symbolic,

            # --- Group C: 选择题 (Multiple Choice / Classification) ---
            "aqua-rat": metrics_choice,
            "mmlu": metrics_choice,
            "agieval-gaokao-mathqa": metrics_choice,
            "math-qa": metrics_choice, # MathQA 虽然有步骤，但常作为选择题评测

            # --- Group D: 代码 (Code) ---
            "humaneval": metrics_code,
            "mbpp": metrics_code,

            # --- Group E: 通用文本生成 / 摘要 / QA ---
            "general_qa": metrics_generation_rouge,
            "summscreen": metrics_generation_rouge,
            "lcsts": metrics_generation_rouge,
            "iwslt2017": metrics_generation_bleu,
            "flores": metrics_generation_bleu,

            # 抽取式 QA / span-based QA
            "squad20": metrics_qa_extractive,
            "tydiqa": metrics_qa_extractive,
            "nq": metrics_qa_extractive,
            "nq_cn": metrics_qa_extractive,
            "qasper": metrics_qa_extractive,

            # LongBench / LV-Eval / Omni 长上下文 QA & 相关任务
            "longbench": metrics_long_context_qa,
            "lveval": metrics_long_context_qa,

            # TruthfulQA 这类带 LLM 裁判的指标，可以在 One-Eval 里实现 truth_score + bleu
            "truthful_qa": [
                {"name": "truth_score", "priority": "primary", "desc": "TruthfulQAEvaluator: 真实度评分 (LLM-based)"},
                {"name": "bleu", "priority": "secondary", "desc": "生成质量的 BLEU 辅助指标"}
            ],

            # --- Group F: 检索 / 计数 / 长上下文结构化任务 ---
            "needlebench": metrics_retrieval,
            "needlebench_v2": metrics_retrieval,
            "longbench_retrieval": metrics_retrieval,
            "longbench_count": metrics_count,
            "longbench_codesim": metrics_code_sim,

            # --- Group G: 安全 / 毒性 ---
            "realtoxicprompts": metrics_toxicity,
            "safety": metrics_toxicity,

            # --- Group H: LLM 裁判 / 主观评测 ---
            # MT-Bench / Arena / Subjective 族 & LEval / LV-Eval / TEval
            "subjective": metrics_llm_judge,
            "arena": metrics_llm_judge,
            "mtbench": metrics_llm_judge,
            "promptbench": metrics_llm_judge,
            "leval": metrics_win_rate,
            "teval": metrics_llm_judge,
            "omni_math_judge": metrics_llm_judge,

            # --- Group I: AUC / 其他分类指标 ---
            "llm_compression": metrics_auc_roc,
        }

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
