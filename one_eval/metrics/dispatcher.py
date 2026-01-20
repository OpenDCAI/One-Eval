# one_eval/metrics/dispatcher.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import re
from one_eval.logger import get_logger 
from one_eval.metrics.core import (
    MetricCategory, 
    get_registered_metrics_meta, 
    load_metric_implementations
)
from one_eval.metrics.config import DATASET_METRIC_MAP_CONFIG
from one_eval.metrics.prompt_generator import MetricPromptGenerator

log = get_logger(__name__)

class MetricDispatcher:
    """
    指标调度器：负责根据数据集信息推荐合适的评测指标。
    替代了原有的 MetricRegistry 类。
    """

    def __init__(self):
        # 模板缓存: template_name -> List[Dict]
        self._templates: Dict[str, List[Dict[str, Any]]] = {}
        
        # 加载指标实现并构建模板
        self._build_templates()
        
        # 加载数据集映射配置
        self._dataset_map = DATASET_METRIC_MAP_CONFIG

    def _build_templates(self):
        """
        从注册中心拉取最新的 MetricMeta，构建推荐模板。
        """
        load_metric_implementations()
        metas = get_registered_metrics_meta()
        
        for meta in metas:
            # meta.groups 格式: {"numerical": "primary", "math": "secondary"}
            for group_name, priority in meta.groups.items():
                if group_name not in self._templates:
                    self._templates[group_name] = []
                
                template_list = self._templates[group_name]
                
                # 检查是否已存在
                existing_item = next((x for x in template_list if x["name"] == meta.name), None)
                if existing_item:
                    existing_item["priority"] = priority
                else:
                    template_list.append({
                        "name": meta.name,
                        "priority": priority,
                    })

    def get_decision_logic_doc(self) -> str:
        return MetricPromptGenerator.get_decision_logic_doc()

    def get_metric_library_doc(self) -> str:
        return MetricPromptGenerator.get_metric_library_doc(get_registered_metrics_meta())

    def register_dataset(self, dataset_name: str, template_name: str):
        """动态注册数据集映射"""
        self._dataset_map[dataset_name.lower()] = template_name
        log.info(f"[MetricDispatcher] 已注册数据集映射 '{dataset_name}' -> '{template_name}'")

    def infer_eval_type(self, bench_meta: Optional[Dict[str, Any]]) -> Optional[str]:
        if not bench_meta:
            return None
        eval_type = bench_meta.get("bench_dataflow_eval_type") or bench_meta.get("eval_type")
        if isinstance(eval_type, str) and eval_type.strip():
            return eval_type.strip()
        return None

    def infer_task_family(
        self,
        eval_type: Optional[str],
        bench_meta: Optional[Dict[str, Any]] = None,
        task_domain: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        """
        智能推断任务族 (Heuristic Logic)
        """
        meta = bench_meta or {}
        explicit = meta.get("task_family") or meta.get("bench_task_family")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip(), 1.0

        task_type = meta.get("task_type")
        domain = meta.get("domain")
        desc = meta.get("description")
        s = " ".join([str(task_type or ""), str(domain or ""), str(desc or "")]).lower()
        td = (task_domain or "").lower()

        def has_any(text: str, keywords: List[str]) -> bool:
            return any(k in text for k in keywords)

        # 这里的逻辑保留，作为 Layer 3 的智能推断
        if eval_type in {MetricCategory.CHOICE_SINGLE, MetricCategory.CHOICE_MULTI}:
            if has_any(s, ["auc", "roc", "binary", "multiclass", "logit"]):
                return "auc_roc", 0.75
            return "choice", 0.9

        if eval_type == MetricCategory.PAIRWISE:
            return "win_rate", 0.9 # Fixed: return template name directly

        if eval_type == MetricCategory.TEXT_SCORE:
            if has_any(s, ["toxicity", "toxic", "safety", "harm", "jailbreak", "毒性", "安全"]):
                return "safety_toxicity", 0.9
            if has_any(s, ["truth", "factual", "halluc", "事实", "幻觉", "真实"]):
                return "truthfulness", 0.85
            if has_any(s, ["judge", "评分", "打分", "score"]):
                return "llm_judge", 0.8
            return None, 0.0

        if eval_type in {MetricCategory.QA_SINGLE, MetricCategory.QA_MULTI}:
            if has_any(s, ["translate", "translation", "翻译", "译为", "翻成"]):
                return "generation_bleu", 0.9
            if has_any(s, ["summarize", "summary", "摘要", "总结", "tl;dr"]):
                return "generation_rouge", 0.85
            if has_any(s, ["write a function", "write code", "implement", "def ", "class ", "函数", "代码"]):
                return "code", 0.85
            if has_any(s, ["retrieval", "rag", "needle", "检索"]):
                return "retrieval", 0.75
            if has_any(s, ["count", "计数"]):
                return "count", 0.7

            if has_any(s, ["math", "arithmetic", "数学", "算术", "numerical"]):
                if has_any(s, ["latex", "\\boxed", "equation", "符号", "推导"]):
                    return "symbolic", 0.85
                return "numerical", 0.75

            if has_any(s, ["qa", "question_answering", "extractive", "span"]):
                if has_any(s, ["long", "context", "长文本", "longbench"]):
                    return "long_context_qa", 0.7
                return "qa_extractive", 0.7

            if td == "math":
                return "numerical", 0.55
            if td in {"text", "nlp"} and has_any(s, ["qa", "question", "answer", "问答"]):
                return "qa_extractive", 0.55

            return None, 0.0

        return None, 0.0

    def _normalize_key(self, key: str) -> str:
        clean_key = re.sub(r'[^a-z0-9]', '_', key.lower())
        clean_key = re.sub(r'_+', '_', clean_key).strip('_')
        return f"_{clean_key}_"

    def get_metrics(
        self,
        dataset_name: str,
        bench_meta: Optional[Dict[str, Any]] = None,
        eval_type: Optional[str] = None,
        task_domain: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        获取数据集的指标配置。
        """
        
        # 1. 预处理
        raw_name = dataset_name.lower().strip()
        normalized_name = self._normalize_key(raw_name)
        
        current_eval_type = eval_type
        if not current_eval_type:
             current_eval_type = self.infer_eval_type(bench_meta)
        
        # 2. Name Match (Layer 2)
        # 查找配置表中是否有匹配
        matched_template_name = None
        best_match_len = 0
        
        for key, template in self._dataset_map.items():
            normalized_key = self._normalize_key(key)
            if normalized_key in normalized_name:
                if len(key) > best_match_len:
                    best_match_len = len(key)
                    matched_template_name = template
        
        # 如果命中配置，直接返回
        if matched_template_name:
             # Conflict check could be added here if needed
             if matched_template_name in self._templates:
                 return self._templates[matched_template_name]
             else:
                 log.warning(f"Dataset {dataset_name} mapped to template {matched_template_name}, but template not found.")

        # 3. Type + Domain Inference (Layer 3)
        if current_eval_type:
            family, confidence = self.infer_task_family(
                current_eval_type,
                bench_meta=bench_meta,
                task_domain=task_domain,
            )
            
            if family and family in self._templates:
                return self._templates[family]
                
            # Fallback based on eval_type only
            if current_eval_type in {MetricCategory.CHOICE_SINGLE, MetricCategory.CHOICE_MULTI}:
                return self._templates.get("choice", [])
            
            if current_eval_type == MetricCategory.PAIRWISE:
                return self._templates.get("win_rate", [])

        return None
    
metric_dispatcher = MetricDispatcher()