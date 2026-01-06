from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
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
        # 保留结构化后的 eval_type 信息
        self._eval_type_specs: Dict[str, Dict[str, Any]] = {
            "key1_text_score": {
                "title": "文本打分",
                "required_keys": ["text"],
                "optional_keys": ["context"],
                "artifacts": ["predict"],
            },
            "key2_qa": {
                "title": "生成式：单参考答案",
                "required_keys": ["question", "target"],
                "optional_keys": ["context"],
                "artifacts": ["predict", "ground_truth"],
            },
            "key2_q_ma": {
                "title": "生成式：多参考答案",
                "required_keys": ["question", "targets"],
                "optional_keys": ["context"],
                "artifacts": ["predict", "ground_truth"],
            },
            "key3_q_choices_a": {
                "title": "选择题：单正确",
                "required_keys": ["question", "choices", "label"],
                "optional_keys": ["context"],
                "artifacts": ["predict", "ground_truth"],
            },
            "key3_q_choices_as": {
                "title": "选择题：多正确",
                "required_keys": ["question", "choices", "labels"],
                "optional_keys": ["context"],
                "artifacts": ["predict", "ground_truth"],
            },
            "key3_q_a_rejected": {
                "title": "偏好/排序：成对比较",
                "required_keys": ["question", "better", "rejected"],
                "optional_keys": ["context"],
                "artifacts": ["predict", "ground_truth"],
            },
        }

        # --- 初始化metric定义 ---
        # 以 bench_dataflow_eval_type 为一级分类，便于覆盖纯文本 bench 的评测规划。
        self._definitions = {
            "文本打分 (key1_text_score)": [
                {"name": "llm_judge_score", "desc": "LLM 裁判打分 (0~100)", "usage": "开放式主观评测/打分式任务"},
                {"name": "toxicity_max", "desc": "样本毒性分数的最大值", "usage": "安全检测/毒性评测"},
                {"name": "truth_score", "desc": "真实度评分 (LLM-based)", "usage": "幻觉检测/事实性评测"}
            ],
            "生成式：单参考答案 (key2_qa)": [
                {"name": "exact_match", "desc": "抽取式答案完全匹配 (EM)", "usage": "短答案/抽取式 QA"},
                {"name": "f1", "desc": "token 级 F1 (匹配程度)", "usage": "长答案/部分匹配"},
                {"name": "strict_match", "desc": "原始字符串严格匹配", "usage": "格式严格的任务"},
                {"name": "numerical_match", "desc": "数值软匹配(1.0 == 1，容忍浮点误差)", "usage": "算术题/数值填空"},
                {"name": "symbolic_match", "desc": "SymPy / LaTeX 等价性校验", "usage": "代数推导/高等数学"},
                {"name": "rouge_l", "desc": "ROUGE-L F1", "usage": "摘要/生成"},
                {"name": "rouge_1", "desc": "ROUGE-1 F1", "usage": "摘要/生成 (辅助)"},
                {"name": "rouge_2", "desc": "ROUGE-2 F1", "usage": "摘要/生成 (辅助)"},
                {"name": "bleu", "desc": "sacreBLEU 主指标", "usage": "翻译/生成"},
                {"name": "bert_score", "desc": "语义相似度", "usage": "开放生成 (可选)"},
                {"name": "retrieval_accuracy", "desc": "是否检索到正确段落/索引", "usage": "RAG/检索类输出"},
                {"name": "count_accuracy", "desc": "计数覆盖精度", "usage": "计数类输出"},
                {"name": "code_similarity", "desc": "代码相似度", "usage": "代码输出相似度对比"},
                {"name": "extraction_rate", "desc": "正则提取成功率 (强烈建议)", "usage": "所有需要从长输出提取答案的任务"}
            ],
            "生成式：多参考答案 (key2_q_ma)": [
                {"name": "exact_match", "desc": "多参考下的 EM (实现侧需支持多 gold)", "usage": "SQuAD 多答案/多可接受答案"},
                {"name": "f1", "desc": "多参考下的 token F1 (实现侧需支持多 gold)", "usage": "多答案/部分匹配"},
                {"name": "rouge_l", "desc": "多参考下的 ROUGE-L", "usage": "多参考生成"},
                {"name": "bleu", "desc": "多参考下的 BLEU", "usage": "多参考翻译"},
                {"name": "bert_score", "desc": "多参考下的语义相似度", "usage": "多参考开放生成 (可选)"},
                {"name": "extraction_rate", "desc": "正则提取成功率 (强烈建议)", "usage": "多参考任务的答案提取监控"}
            ],
            "选择题：单正确 (key3_q_choices_a)": [
                {"name": "choice_accuracy", "desc": "选项字母或离散标签准确率", "usage": "选择题 (A/B/C/D)"},
                {"name": "missing_answer_rate", "desc": "未输出有效选项/标签的比例 (诊断用)", "usage": "监控模型拒答率"},
                {"name": "auc_roc", "desc": "AUC-ROC ×100", "usage": "二分类/多分类任务"},
                {"name": "accuracy", "desc": "通用 accuracy (当 evaluator 支持时)", "usage": "分类任务辅助指标"}
            ],
            "选择题：多正确 (key3_q_choices_as)": [
                {"name": "choice_accuracy", "desc": "多选题准确率/命中率 (实现侧需明确规则)", "usage": "多正确答案的选择题"},
                {"name": "missing_answer_rate", "desc": "未输出有效选项/标签的比例 (诊断用)", "usage": "监控模型拒答率"}
            ],
            "偏好/排序：成对比较 (key3_q_a_rejected)": [
                {"name": "win_rate_against_baseline", "desc": "胜率/偏好取胜率", "usage": "pairwise preference / DPO 数据"}
            ]
        }
        # --- 初始化metric模板 ---
        # 暂时保留不动，后续可能会弃用
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

        self._schema_family_templates: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            "key1_text_score": {
                "safety_toxicity": [{"name": "toxicity_max", "priority": "primary"}],
                "truthfulness": [{"name": "truth_score", "priority": "primary"}],
                "llm_judge": self._templates["llm_judge"],
            },
            "key2_qa": {
                "extractive_qa": self._templates["qa_extractive"],
                "long_context_qa": self._templates["long_context_qa"],
                "math_numeric": self._templates["numerical"],
                "math_symbolic": self._templates["symbolic"],
                "translation": self._templates["generation_bleu"],
                "summarization": self._templates["generation_rouge"],
                "code": self._templates["code"],
                "retrieval": self._templates["retrieval"],
                "count": self._templates["count"],
                "open_ended": self._templates["llm_judge"],
            },
            "key2_q_ma": {
                "extractive_qa": self._templates["qa_extractive"],
                "long_context_qa": self._templates["long_context_qa"],
                "translation": self._templates["generation_bleu"],
                "summarization": self._templates["generation_rouge"],
                "open_ended": self._templates["llm_judge"],
            },
            "key3_q_choices_a": {
                "choice": self._templates["choice"],
                "auc_roc": self._templates["auc_roc"],
            },
            "key3_q_choices_as": {
                "choice": self._templates["choice"],
            },
            "key3_q_a_rejected": {
                "pairwise_preference": self._templates["win_rate"],
            },
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

    def infer_eval_type(self, bench_meta: Optional[Dict[str, Any]]) -> Optional[str]:
        if not bench_meta:
            return None
        eval_type = bench_meta.get("bench_dataflow_eval_type") or bench_meta.get("eval_type")
        if isinstance(eval_type, str) and eval_type.strip():
            return eval_type.strip()
        return None

    # 从 eval_type 分流， 基于 bench_meta 或者 prompt_template 来判断任务类型
    def infer_task_family(
        self,
        eval_type: Optional[str],
        bench_meta: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        task_domain: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        meta = bench_meta or {}
        explicit = meta.get("task_family") or meta.get("bench_task_family")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip(), 1.0

        task_type = meta.get("task_type")
        domain = meta.get("domain")
        desc = meta.get("description")
        s = " ".join([str(task_type or ""), str(domain or ""), str(desc or "")]).lower()
        p = (prompt_template or "").lower()
        td = (task_domain or "").lower()

        def has_any(text: str, keywords: List[str]) -> bool:
            return any(k in text for k in keywords)

        if eval_type in {"key3_q_choices_a", "key3_q_choices_as"}:
            if has_any(s + " " + p, ["auc", "roc", "binary", "multiclass", "logit"]):
                return "auc_roc", 0.75
            return "choice", 0.9

        if eval_type == "key3_q_a_rejected":
            return "pairwise_preference", 0.9

        if eval_type == "key1_text_score":
            if has_any(s + " " + p, ["toxicity", "toxic", "safety", "harm", "jailbreak", "毒性", "安全"]):
                return "safety_toxicity", 0.9
            if has_any(s + " " + p, ["truth", "factual", "halluc", "事实", "幻觉", "真实"]):
                return "truthfulness", 0.85
            if has_any(s + " " + p, ["judge", "评分", "打分", "score"]):
                return "llm_judge", 0.8
            return None, 0.0

        if eval_type in {"key2_qa", "key2_q_ma"}:
            if has_any(p, ["translate", "translation", "翻译", "译为", "翻成"]):
                return "translation", 0.9
            if has_any(p, ["summarize", "summary", "摘要", "总结", "tl;dr"]):
                return "summarization", 0.85
            if has_any(p, ["write a function", "write code", "implement", "def ", "class ", "函数", "代码"]):
                return "code", 0.85
            if has_any(s + " " + p, ["retrieval", "rag", "needle", "检索"]):
                return "retrieval", 0.75
            if has_any(s + " " + p, ["count", "计数"]):
                return "count", 0.7

            if has_any(s + " " + p, ["math", "arithmetic", "数学", "算术", "numerical"]):
                if has_any(s + " " + p, ["latex", "\\boxed", "equation", "符号", "推导"]):
                    return "math_symbolic", 0.85
                return "math_numeric", 0.75

            if has_any(s, ["qa", "question_answering", "extractive", "span"]):
                if has_any(s, ["long", "context", "长文本", "longbench"]):
                    return "long_context_qa", 0.7
                return "extractive_qa", 0.7

            if td == "math":
                return "math_numeric", 0.55
            if td in {"text", "nlp"} and has_any(s, ["qa", "question", "answer", "问答"]):
                return "extractive_qa", 0.55

            return None, 0.0

        return None, 0.0

    def get_default_metrics_by_eval_type(
        self,
        eval_type: str,
        bench_meta: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        task_domain: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        key = (eval_type or "").strip()
        if not key:
            return None

        family, confidence = self.infer_task_family(
            key,
            bench_meta=bench_meta,
            prompt_template=prompt_template,
            task_domain=task_domain,
        )

        if family and key in self._schema_family_templates and family in self._schema_family_templates[key]:
            return self._schema_family_templates[key][family]

        if key in {"key3_q_choices_a", "key3_q_choices_as"}:
            return self._schema_family_templates[key]["choice"]

        if key == "key3_q_a_rejected":
            return self._schema_family_templates[key]["pairwise_preference"]

        if key == "key1_text_score" and confidence >= 0.8 and family:
            return self._schema_family_templates[key][family]

        return None

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

    def get_metrics(
        self,
        dataset_name: str,
        bench_meta: Optional[Dict[str, Any]] = None,
        eval_type: Optional[str] = None,
        prompt_template: Optional[str] = None,
        task_domain: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        获取数据集的指标配置。
        - 优先：若提供 eval_type 或可从 bench_meta 推断 eval_type，则按 (eval_type, task_family) 选择模板。
        - 回退：按 dataset_name 做模糊匹配。
        - 找不到则返回 None。
        """
        if eval_type:
            metrics = self.get_default_metrics_by_eval_type(
                eval_type,
                bench_meta=bench_meta,
                prompt_template=prompt_template,
                task_domain=task_domain,
            )
            if metrics:
                return metrics

        inferred = self.infer_eval_type(bench_meta)
        if inferred:
            metrics = self.get_default_metrics_by_eval_type(
                inferred,
                bench_meta=bench_meta,
                prompt_template=prompt_template,
                task_domain=task_domain,
            )
            if metrics:
                return metrics

        # 统一标准化输入
        raw_name = dataset_name.lower().strip()
        normalized_input = self._normalize_key(raw_name)
        
        matched_metrics: List[Dict[str, Any]] = []
        best_match_len = 0

        for key, metrics in self._registry.items():
            normalized_key = self._normalize_key(key)
            if normalized_key in normalized_input:
                if len(key) > best_match_len:
                    best_match_len = len(key)
                    matched_metrics = metrics

        if matched_metrics:
            log.info(f"[MetricRegistry] 匹配成功: '{dataset_name}' (Matched Key: {len(matched_metrics)} metrics)")
            return matched_metrics

        return None


# 全局单例
metric_registry = MetricRegistry()
