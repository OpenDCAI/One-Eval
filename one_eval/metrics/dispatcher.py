# one_eval/metrics/dispatcher.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import re
from one_eval.logger import get_logger 
from one_eval.core.metric_registry import (
    get_registered_metrics_meta, 
    load_metric_implementations
)
from one_eval.metrics.config import DATASET_METRICS
from one_eval.metrics.prompt_generator import MetricPromptGenerator

log = get_logger(__name__)

class MetricDispatcher:
    """
    指标调度器：负责根据数据集信息推荐合适的评测指标。
    替代了原有的 MetricRegistry 类。
    """

    def __init__(self):
        # 确保指标实现已加载
        load_metric_implementations()
        
        # 加载数据集映射配置 (直接使用扁平化配置)
        self._dataset_map = DATASET_METRICS

    def get_decision_logic_doc(self) -> str:
        return MetricPromptGenerator.get_decision_logic_doc()

    def get_metric_library_doc(self) -> str:
        return MetricPromptGenerator.get_metric_library_doc(get_registered_metrics_meta())

    def register_dataset(self, dataset_name: str, metrics: List[str]):
        """动态注册数据集映射 (List[str])"""
        self._dataset_map[dataset_name.lower()] = metrics
        log.info(f"[MetricDispatcher] 已注册数据集映射 '{dataset_name}'")

    def _normalize_key(self, key: str) -> str:
        clean_key = re.sub(r'[^a-z0-9]', '_', key.lower())
        clean_key = re.sub(r'_+', '_', clean_key).strip('_')
        return f"_{clean_key}_"

    def get_default_priority(self, metric_name: str) -> str:
        diagnostic = {
            "extraction_rate",
            "format_compliance",
            "case_study_analyst",
            "metric_summary_analyst",
        }
        return "diagnostic" if metric_name in diagnostic else "secondary"

    def _inflate_metrics(self, metric_names: List[str]) -> List[Dict[str, Any]]:
        """
        将简单的指标名称列表膨胀为带有默认优先级的配置字典。
        策略：
        1. 第一个指标 -> primary
        2. 特殊诊断指标 -> diagnostic
        3. 其他 -> secondary
        """
        result = []
        for idx, name in enumerate(metric_names):
            priority = "primary" if idx == 0 else self.get_default_priority(name)
            result.append({
                "name": name,
                "priority": priority
            })
        return result

    def get_metrics(
        self,
        dataset_name: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        获取数据集的指标配置 (仅查表)。
        """
        
        # 1. 预处理
        raw_name = dataset_name.lower().strip()
        normalized_name = self._normalize_key(raw_name)
        
        # 2. Name Match (Layer 2)
        # 查找配置表中是否有匹配
        matched_metric_names = None
        best_match_len = 0
        
        for key, names in self._dataset_map.items():
            normalized_key = self._normalize_key(key)
            if normalized_key in normalized_name:
                if len(key) > best_match_len:
                    best_match_len = len(key)
                    matched_metric_names = names
        
        # 如果命中配置，膨胀并返回
        if matched_metric_names:
             return self._inflate_metrics(matched_metric_names)

        return None

    
metric_dispatcher = MetricDispatcher()