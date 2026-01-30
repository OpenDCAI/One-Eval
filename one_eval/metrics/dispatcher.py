# one_eval/metrics/dispatcher.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import re
from one_eval.logger import get_logger 
from one_eval.core.metric_registry import (
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

    def _normalize_key(self, key: str) -> str:
        clean_key = re.sub(r'[^a-z0-9]', '_', key.lower())
        clean_key = re.sub(r'_+', '_', clean_key).strip('_')
        return f"_{clean_key}_"

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
             if matched_template_name in self._templates:
                 return self._templates[matched_template_name]
             else:
                 log.warning(f"Dataset {dataset_name} mapped to template {matched_template_name}, but template not found.")

        return None

    
metric_dispatcher = MetricDispatcher()