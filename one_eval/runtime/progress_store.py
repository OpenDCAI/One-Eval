from __future__ import annotations

from typing import Any, Dict, Optional
from threading import Lock

_LOCK = Lock()
_PROGRESS: Dict[str, Dict[str, Any]] = {}


def set_progress(thread_id: str, payload: Dict[str, Any]) -> None:
    if not thread_id:
        return
    with _LOCK:
        _PROGRESS[thread_id] = dict(payload or {})


def get_progress(thread_id: str) -> Optional[Dict[str, Any]]:
    if not thread_id:
        return None
    with _LOCK:
        val = _PROGRESS.get(thread_id)
        return dict(val) if isinstance(val, dict) else None


def clear_progress(thread_id: str) -> None:
    if not thread_id:
        return
    with _LOCK:
        _PROGRESS.pop(thread_id, None)
