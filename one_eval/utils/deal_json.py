import json
from pathlib import Path
from dataclasses import asdict, is_dataclass, fields
from one_eval.core.state import NodeState, BenchInfo

def _json_safe(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def _save_state_json(data, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, ensure_ascii=False, indent=2)

def _restore_state_from_snap(values: dict) -> NodeState:
    # 1) 处理 benches：dict -> BenchInfo
    benches = []
    for b in values.get("benches", []) or []:
        if isinstance(b, BenchInfo):
            benches.append(b)
        elif isinstance(b, dict):
            # 只取 BenchInfo 支持的字段，避免多余字段报错
            allowed = {f.name for f in fields(BenchInfo)}
            benches.append(BenchInfo(**{k: v for k, v in b.items() if k in allowed}))
        else:
            benches.append(b)

    # 2) NodeState 还原（同样只取 NodeState 支持字段）
    allowed_s = {f.name for f in fields(NodeState)}
    base = {k: v for k, v in values.items() if k in allowed_s}
    base["benches"] = benches
    return NodeState(**base)
