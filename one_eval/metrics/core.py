# one_eval/metrics/core.py
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
import importlib
import pkgutil
from one_eval.logger import get_logger

log = get_logger(__name__)

@dataclass
class MetricMeta:
    name: str
    func: Callable
    desc: str
    usage: str
    categories: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    groups: Dict[str, str] = field(default_factory=dict)  # Key: template_group_name, Value: priority
    priority: str = "secondary" # default priority if not specified in group

class MetricCategory:
    """Metric Categories Constants (Eval Types)"""
    TEXT_SCORE = "key1_text_score"
    QA_SINGLE = "key2_qa" 
    QA_MULTI = "key2_q_ma"
    CHOICE_SINGLE = "key3_q_choices_a"
    CHOICE_MULTI = "key3_q_choices_as"
    PAIRWISE = "key3_q_a_rejected"

# 全局注册表缓存
# key: metric_name, value: MetricMeta
_REGISTRY_CACHE: Dict[str, MetricMeta] = {}

# 别名映射
# key: alias_name, value: real_metric_name
_ALIAS_MAP: Dict[str, str] = {}

def register_metric(
    name: Optional[str] = None, 
    desc: str = "", 
    usage: str = "", 
    categories: Optional[List[str]] = None,
    aliases: Optional[List[str]] = None,
    groups: Optional[Dict[str, str]] = None,
    priority: str = "secondary"
):
    """
    装饰器：注册 Metric 实现及其元数据。
    
    Args:
        name: 指标名称
        desc: 描述 (用于 Agent 理解)
        usage: 适用场景 (用于 Agent 推荐)
        categories: 归属的大类列表 (使用 MetricCategory 常量)
        aliases: 别名列表 (如 'em', 'acc')
        groups: 推荐模板组定义，格式为 { "group_name": "priority" }。
                例如: {"qa_extractive": "primary", "long_context_qa": "secondary"}
        priority: 默认优先级
    """
    def decorator(func):
        nonlocal name
        if name is None:
            # 自动推断名称：compute_exact_match -> exact_match
            fn_name = func.__name__
            if fn_name.startswith("compute_"):
                name = fn_name[8:]
            else:
                name = fn_name
        
        meta = MetricMeta(
            name=name,
            func=func,
            desc=desc,
            usage=usage,
            categories=categories or [],
            aliases=aliases or [],
            groups=groups or {},
            priority=priority
        )
        
        # 注册主名称
        _REGISTRY_CACHE[name] = meta
        
        # 注册别名
        for alias in meta.aliases:
            _ALIAS_MAP[alias] = name
            
        return func
    return decorator

def load_metric_implementations():
    """
    自动扫描并加载 one_eval.metrics.common 下的所有模块，触发装饰器注册。
    """
    lib_package_name = "one_eval.metrics.common"
    try:
        lib_module = importlib.import_module(lib_package_name)
    except ImportError:
        log.warning(f"无法导入 {lib_package_name}，跳过自动发现。")
        return

    if not hasattr(lib_module, "__path__"):
        return

    for _, mod_name, _ in pkgutil.walk_packages(lib_module.__path__, lib_package_name + "."):
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            log.error(f"加载 Metric 模块 {mod_name} 失败: {e}")

def get_metric_fn(name: str) -> Optional[Callable]:
    """获取 Metric 计算函数"""
    # 1. 确保已加载
    if not _REGISTRY_CACHE:
        load_metric_implementations()
        
    # 2. 查找别名
    target_name = _ALIAS_MAP.get(name, name)
    
    # 3. 查找注册表
    if target_name in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[target_name].func
        
    return None

def get_registered_metrics_meta() -> List[MetricMeta]:
    """获取所有已注册的 Metric 元数据 (用于 Agent 注册表构建)"""
    if not _REGISTRY_CACHE:
        load_metric_implementations()
    return list(_REGISTRY_CACHE.values())