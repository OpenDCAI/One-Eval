"""
One-Eval Skill 共享层。

所有 scripts 通过本模块复用 one_eval 主包，并统一：
- 把仓库根加入 sys.path（保证 `import one_eval` 可用）
- ModelConfig / BenchInfo 的构造（从 evalspec dict）
- 路径约定（输出目录、缓存目录、本地状态文件）
- .local_state.json 的读写（记录已测通 bench 的 READY 状态）

设计：直接 import one_eval，不拷贝评测内核。评测本身就依赖 one_eval + dataflow 环境。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- 仓库根定位：one-eval-skill/scripts/_common.py -> 仓库根是上上级 ---
SKILL_DIR = Path(__file__).resolve().parent.parent      # one-eval-skill/
REPO_ROOT = SKILL_DIR.parent                            # One-Eval/（含 one_eval 包）

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- 路径约定 ---
DEFAULT_OUTPUT_DIR = SKILL_DIR / "eval_outputs"         # 评测产出根（结果/图/报告）
DEFAULT_CACHE_DIR = SKILL_DIR / "cache"                 # 数据集下载缓存
LOCAL_STATE_PATH = SKILL_DIR / ".local_state.json"      # 已测通 bench 的 READY 记录
CUSTOM_METRICS_DIR = SKILL_DIR / "custom_metrics"       # 用户自定义 metric 落地处
RUNS_SUBDIR = "runs"                                    # 每次评测独立子目录的父目录名
LATEST_RUN_FILE = "latest_run.txt"                      # 指向最新 run 目录的指针文件


def new_run_id() -> str:
    """生成本次评测的唯一 run_id（本地时间戳，秒级）。

    用于把产物隔离到 eval_outputs/runs/<run_id>/，避免多次评测互相覆盖。
    """
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(output_root: Path, run_id: str) -> Path:
    """在 output_root/runs/<run_id>/ 下建本次评测目录，并把 latest 指针指向它。"""
    run_dir = Path(output_root) / RUNS_SUBDIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        (Path(output_root) / LATEST_RUN_FILE).write_text(str(run_dir.resolve()),
                                                         encoding="utf-8")
    except Exception:
        pass  # 指针只是便利，写失败不影响评测
    return run_dir


def sanitize_model_config(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """落盘用的模型配置快照：保留可复现关键参数，api_key 脱敏（绝不落真值）。"""
    keep = ["model_name_or_path", "is_api", "api_provider", "api_url",
            "temperature", "top_p", "top_k", "repetition_penalty",
            "max_tokens", "seed", "tensor_parallel_size", "max_model_len"]
    snap = {k: model_dict.get(k) for k in keep if model_dict.get(k) is not None}
    if model_dict.get("api_key"):
        snap["api_key"] = "***"  # 标记用过 key，但不落真值
    return snap

# bench 种类：dataflow（默认，走确定性内核）/ external_repo（自带仓库、需特殊环境，
# 不走内核，按 meta.repo_eval 在外部执行后回填分数）。详见 references/external_bench.md。
BENCH_KIND_DATAFLOW = "dataflow"
BENCH_KIND_EXTERNAL = "external_repo"
VALID_BENCH_KINDS = {BENCH_KIND_DATAFLOW, BENCH_KIND_EXTERNAL}


def get_bench_kind(bench_dict: Dict[str, Any]) -> str:
    """读 bench 的 kind，缺省为 dataflow（现有条目不带该字段，行为不变）。"""
    kind = (bench_dict.get("bench_kind") or BENCH_KIND_DATAFLOW)
    if kind not in VALID_BENCH_KINDS:
        raise ValueError(
            f"bench {bench_dict.get('bench_name')!r} 的 bench_kind 非法: {kind!r}，"
            f"只能是 {sorted(VALID_BENCH_KINDS)}"
        )
    return kind


# 6 种合法 eval 类型（硬契约，与 one_eval/nodes/dataflow_eval_node.py 一致）
VALID_EVAL_TYPES = {
    "key1_text_score",
    "key2_qa",
    "key2_q_ma",
    "key3_q_choices_a",
    "key3_q_choices_as",
    "key3_q_a_rejected",
}

# 每种 eval 类型必填的 key_mapping 字段
REQUIRED_KEYS: Dict[str, List[str]] = {
    "key1_text_score": ["input_text_key"],
    "key2_qa": ["input_question_key", "input_target_key"],
    "key2_q_ma": ["input_question_key", "input_targets_key"],
    "key3_q_choices_a": ["input_question_key", "input_choices_key", "input_label_key"],
    "key3_q_choices_as": ["input_question_key", "input_choices_key", "input_labels_key"],
    "key3_q_a_rejected": ["input_better_key", "input_rejected_key"],
}

def build_model_config(model_dict: Dict[str, Any]):
    """从 evalspec 的 model 段构造 one_eval 的 ModelConfig。"""
    from one_eval.core.state import ModelConfig

    if not model_dict or not model_dict.get("model_name_or_path"):
        raise ValueError("model.model_name_or_path 必填")

    allowed = {
        "model_name_or_path", "is_api", "api_url", "api_key", "api_provider",
        "api_extra_body", "api_max_workers", "api_connect_timeout", "api_read_timeout",
        "temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "seed",
        "tensor_parallel_size", "max_model_len", "gpu_memory_utilization",
    }
    kwargs = {k: v for k, v in model_dict.items() if k in allowed and v is not None}
    return ModelConfig(**kwargs)


def build_bench_info(bench_dict: Dict[str, Any], dataset_cache: Optional[str] = None):
    """从 evalspec 的单个 benchmark 段构造 one_eval 的 BenchInfo。

    key_mapping / download_config 放进 meta，供 DataFlowEvalTool.run_eval 读取。
    """
    from one_eval.core.state import BenchInfo

    # external_repo bench 不走内核，eval_type/key_mapping 不适用：直接带出 repo_eval 信息。
    # 正常情况下 run_eval.py 会在更上层就短路，这里是防御性兜底（避免误调时崩在硬校验上）。
    if get_bench_kind(bench_dict) == BENCH_KIND_EXTERNAL:
        bench = BenchInfo(
            bench_name=bench_dict.get("bench_name"),
            bench_source_url=bench_dict.get("bench_source_url"),
            bench_dataflow_eval_type=bench_dict.get("bench_dataflow_eval_type"),
            dataset_cache=dataset_cache,
        )
        repo_eval = (bench_dict.get("meta") or {}).get("repo_eval", {})
        bench.meta["bench_kind"] = BENCH_KIND_EXTERNAL
        bench.meta["repo_eval"] = repo_eval
        return bench

    eval_type = bench_dict.get("bench_dataflow_eval_type")
    if eval_type not in VALID_EVAL_TYPES:
        raise ValueError(
            f"bench_dataflow_eval_type 非法: {eval_type!r}，"
            f"只能是 6 种之一: {sorted(VALID_EVAL_TYPES)}"
        )

    key_mapping = bench_dict.get("key_mapping", {}) or {}
    missing = [k for k in REQUIRED_KEYS[eval_type] if not key_mapping.get(k)]
    if missing:
        raise ValueError(
            f"bench {bench_dict.get('bench_name')!r} 的 eval_type={eval_type} "
            f"缺少必填 key_mapping 字段: {missing}"
        )

    bench = BenchInfo(
        bench_name=bench_dict.get("bench_name"),
        bench_source_url=bench_dict.get("bench_source_url"),
        bench_dataflow_eval_type=eval_type,
        dataset_cache=dataset_cache,
    )
    bench.meta["key_mapping"] = key_mapping
    if bench_dict.get("download_config"):
        bench.meta["download_config"] = bench_dict["download_config"]
    return bench


# --- .local_state.json：已测通 bench 的 READY 记录 ---
def load_local_state() -> Dict[str, Any]:
    if LOCAL_STATE_PATH.exists():
        try:
            return json.loads(LOCAL_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_local_state(state: Dict[str, Any]) -> None:
    LOCAL_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def mark_bench_ready(bench_name: str, dataset_path: str, eval_type: str,
                     key_mapping: Dict[str, Any]) -> None:
    """标记某 bench 已测通：记录本地数据路径 + 验证过的 eval_type/key_mapping。"""
    state = load_local_state()
    state.setdefault("ready_benches", {})[bench_name] = {
        "dataset_path": str(dataset_path),
        "bench_dataflow_eval_type": eval_type,
        "key_mapping": key_mapping,
    }
    save_local_state(state)


def get_ready_bench(bench_name: str) -> Optional[Dict[str, Any]]:
    """查某 bench 是否已测通；返回其记录（含本地路径），否则 None。"""
    return load_local_state().get("ready_benches", {}).get(bench_name)


def load_evalspec(path: str) -> Dict[str, Any]:
    """读取 evalspec.yaml。"""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"evalspec 解析结果不是 dict: {path}")
    return spec


# --- metric 注册表加载：内置 + 用户自定义 ---
_METRICS_LOADED = False


def ensure_metrics_loaded() -> List[str]:
    """加载内置 metric，并动态 import custom_metrics/*.py 触发其 @register_metric。

    内核的 load_metric_implementations() 只扫描 one_eval.metrics.common，不会扫到
    skill 的 custom_metrics/。这里补上：把 custom_metrics/ 加进 sys.path 后逐个
    import，使自定义 metric 用注册名即可被引擎/CLI 引用。幂等。

    返回成功加载的自定义模块名列表（供调用方打印/调试）。
    """
    global _METRICS_LOADED
    loaded_custom: List[str] = []

    from one_eval.core.metric_registry import load_metric_implementations
    if not _METRICS_LOADED:
        load_metric_implementations()

    if CUSTOM_METRICS_DIR.is_dir():
        import importlib
        if str(CUSTOM_METRICS_DIR) not in sys.path:
            sys.path.insert(0, str(CUSTOM_METRICS_DIR))
        for py in sorted(CUSTOM_METRICS_DIR.glob("*.py")):
            if py.name.startswith("_"):
                continue
            mod_name = py.stem
            try:
                if mod_name in sys.modules:
                    importlib.reload(sys.modules[mod_name])
                else:
                    importlib.import_module(mod_name)
                loaded_custom.append(mod_name)
            except Exception as e:
                print(f"⚠ 加载自定义 metric {py.name} 失败: {e}", file=sys.stderr)

    _METRICS_LOADED = True
    return loaded_custom

