
import asyncio
import uuid
import json
import os
import re
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from one_eval.logger import get_logger
from one_eval.toolkits.hf_download_tool import HFDownloadTool
from one_eval.runtime.progress_store import get_progress, clear_progress

log = get_logger("OneEval-Server")

# === Early Environment Setup ===
# Must be done before importing langgraph/transformers/etc. to ensure env vars take effect
SERVER_DIR = Path(__file__).resolve().parent
DATA_DIR = SERVER_DIR / "_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = DATA_DIR / "config.json"
MODELS_FILE = DATA_DIR / "models.json"
THREAD_META_FILE = DATA_DIR / "thread_meta.json"
# SERVER_DIR is .../one_eval/server
# parents[0]=one_eval, parents[1]=One-Eval (Repo Root)
REPO_ROOT = SERVER_DIR.parents[1]
ENV_FILE = REPO_ROOT / "env.sh"

# Original DB location was parents[2] (scy/checkpoints)
# We keep it there or move it? 
# If previous code used parents[2], we should respect it to find existing DB.
DB_PATH = (SERVER_DIR.parents[2] / "checkpoints" / "eval.db").resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 内存缓存：记录每个 thread 是否处于 interrupted 状态
# ainvoke 返回时立即写入，避免 get_status 依赖 checkpoint 竞态
_thread_interrupt_cache: Dict[str, bool] = {}

def _load_env_file():
    """Parse env.sh and set os.environ if not already set."""
    if not ENV_FILE.exists():
        return
    
    log.info(f"Loading env from {ENV_FILE}")
    content = ENV_FILE.read_text()
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Support 'export KEY=VALUE' or 'KEY=VALUE'
        if line.startswith("export "):
            line = line[7:].strip()
        
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Only set if not already set (allow shell override) or force?
            # User wants to avoid export, so we should set it if missing.
            # But if config.json exists, it might override later.
            if key not in os.environ and val:
                os.environ[key] = val
                log.info(f"Set {key} from env.sh")

_load_env_file()

def _load_json_file(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text())
    except Exception:
        log.error(f"Error loading {path}: ", exc_info=True)
        return default

def _write_json_file(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        log.error(f"Error writing {path}: ", exc_info=True)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _load_thread_meta() -> Dict[str, Any]:
    data = _load_json_file(THREAD_META_FILE, default={})
    return data if isinstance(data, dict) else {}

def _set_thread_created_at(thread_id: str, created_at: Optional[str] = None) -> str:
    ts = created_at or _now_iso()
    meta = _load_thread_meta()
    item = meta.get(thread_id)
    if not isinstance(item, dict):
        item = {}
    item["created_at"] = ts
    item["updated_at"] = ts
    meta[thread_id] = item
    _write_json_file(THREAD_META_FILE, meta)
    return ts

def _touch_thread_updated_at(thread_id: str, updated_at: Optional[str] = None) -> None:
    meta = _load_thread_meta()
    item = meta.get(thread_id)
    if not isinstance(item, dict):
        item = {}
    if "created_at" not in item or not item.get("created_at"):
        item["created_at"] = updated_at or _now_iso()
    item["updated_at"] = updated_at or _now_iso()
    meta[thread_id] = item
    _write_json_file(THREAD_META_FILE, meta)

def _normalize_model_path_for_host(raw: str) -> str:
    p = (raw or "").strip()
    if not p:
        return p
    if os.name == "nt":
        m = re.match(r"^/mnt/([a-zA-Z])/(.+)$", p)
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace("/", "\\")
            return f"{drive}:\\{rest}"
        return p
    m = re.match(r"^([a-zA-Z]):\\(.+)$", p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p

def load_server_config() -> Dict[str, Any]:
    cfg = _load_json_file(CONFIG_FILE, default={})
    if not isinstance(cfg, dict):
        cfg = {}
        
    # Merge env.sh defaults if config is empty
    # (Optional, but good for first run)
    
    hf = cfg.get("hf")
    if not isinstance(hf, dict):
        hf = {}
    endpoint = hf.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        # Fallback to env or default
        endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    token = hf.get("token")
    if token is not None and (not isinstance(token, str) or not token.strip()):
        token = None
    # If token missing in config, maybe check env?
    if not token and os.environ.get("HF_TOKEN"):
         token = os.environ.get("HF_TOKEN")
         
    cfg["hf"] = {"endpoint": endpoint, "token": token}

    agent = cfg.get("agent")
    if not isinstance(agent, dict):
        agent = {}
    provider = agent.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        provider = "openai_compatible"
    base_url = agent.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        base_url = os.environ.get("DF_API_BASE_URL", "http://123.129.219.111:3000/v1")
    model = agent.get("model")
    if not isinstance(model, str) or not model.strip():
        model = os.environ.get("DF_MODEL_NAME", "gpt-4o")
    api_key = agent.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        api_key = None
    if not api_key and os.environ.get("OE_API_KEY"):
        api_key = os.environ.get("OE_API_KEY")

    agent_timeout_s = agent.get("timeout_s")
    if not isinstance(agent_timeout_s, int) or agent_timeout_s <= 0:
        agent_timeout_s = 15
    cfg["agent"] = {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "timeout_s": agent_timeout_s,
    }
    return cfg

def save_server_config(cfg: Dict[str, Any]) -> None:
    _write_json_file(CONFIG_FILE, cfg)

def apply_hf_env_from_config(cfg: Dict[str, Any]) -> None:
    hf = cfg.get("hf") or {}
    endpoint = hf.get("endpoint")
    token = hf.get("token")
    if isinstance(endpoint, str) and endpoint.strip():
        os.environ["HF_ENDPOINT"] = endpoint.strip()
    if isinstance(token, str) and token.strip():
        os.environ["HF_TOKEN"] = token.strip()
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token.strip()

def _normalize_openai_base_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    if u.endswith("/v1/chat/completions"):
        u = u[: -len("/v1/chat/completions")] + "/v1"
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]
    if u.endswith("/v1/"):
        u = u[:-1]
    return u

def apply_agent_env_from_config(cfg: Dict[str, Any]) -> None:
    agent = cfg.get("agent") or {}
    base_url = agent.get("base_url")
    api_key = agent.get("api_key")
    model = agent.get("model")
    timeout_s = agent.get("timeout_s")
    if isinstance(base_url, str) and base_url.strip():
        os.environ["OE_API_BASE"] = _normalize_openai_base_url(base_url.strip())
        os.environ["DF_API_BASE_URL"] = _normalize_openai_base_url(base_url.strip())
    if isinstance(api_key, str) and api_key.strip():
        os.environ["OE_API_KEY"] = api_key.strip()
        os.environ["DF_API_KEY"] = api_key.strip()
    if isinstance(model, str) and model.strip():
        os.environ["DF_MODEL_NAME"] = model.strip()
        os.environ["OE_MODEL_NAME"] = model.strip()
    if isinstance(timeout_s, int) and timeout_s > 0:
        os.environ["OE_TIMEOUT_S"] = str(timeout_s)
        os.environ["DF_TIMEOUT_S"] = str(timeout_s)

# Initialize Env ASAP
_cfg0 = load_server_config()
log.info(f"Loaded server config: {_cfg0}")
if not CONFIG_FILE.exists():
    save_server_config(_cfg0)
apply_hf_env_from_config(_cfg0)
apply_agent_env_from_config(_cfg0)

from one_eval.graph.workflow_all import build_complete_workflow
from one_eval.utils.checkpoint import get_checkpointer
from one_eval.core.state import NodeState, ModelConfig, BenchInfo, MainRequest
from one_eval.utils.deal_json import _save_state_json
from langgraph.types import Command
from one_eval.utils.bench_registry import BenchRegistry
from one_eval.core.metric_registry import get_registered_metrics_meta, MetricMeta

# Bench Registry - 使用 bench_gallery.json 作为数据源
BENCH_GALLERY_PATH = REPO_ROOT / "one_eval" / "utils" / "bench_table" / "bench_gallery.json"
bench_registry = BenchRegistry(str(BENCH_GALLERY_PATH))

# Models
class HFConfigResponse(BaseModel):
    endpoint: str
    token_set: bool

class HFConfigUpdateRequest(BaseModel):
    endpoint: Optional[str] = None
    token: Optional[str] = None
    clear_token: bool = False

class AgentConfigResponse(BaseModel):
    provider: str
    base_url: str
    model: str
    api_key_set: bool
    timeout_s: int

class AgentConfigUpdateRequest(BaseModel):
    provider: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    clear_api_key: bool = False
    timeout_s: Optional[int] = None

class AgentTestResponse(BaseModel):
    ok: bool
    status_code: Optional[int] = None
    detail: str
    mode: str

app = FastAPI(title="One Eval API")
RUNNING_WORKFLOW_TASKS: Dict[str, asyncio.Task] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/config/hf", response_model=HFConfigResponse)
def get_hf_config():
    cfg = load_server_config()
    hf = cfg.get("hf") or {}
    endpoint = hf.get("endpoint") or "https://hf-mirror.com"
    token = hf.get("token")
    return {"endpoint": endpoint, "token_set": isinstance(token, str) and bool(token.strip())}

@app.post("/api/config/hf", response_model=HFConfigResponse)
def update_hf_config(req: HFConfigUpdateRequest):
    cfg = load_server_config()
    hf = cfg.get("hf") or {}
    endpoint = hf.get("endpoint") or "https://hf-mirror.com"
    token = hf.get("token")

    if req.endpoint is not None:
        ep = req.endpoint.strip()
        endpoint = ep if ep else "https://hf-mirror.com"

    if req.clear_token:
        token = None
    elif req.token is not None:
        tk = req.token.strip()
        if tk:
            token = tk

    cfg["hf"] = {"endpoint": endpoint, "token": token}
    save_server_config(cfg)
    apply_hf_env_from_config(cfg)
    return {"endpoint": endpoint, "token_set": isinstance(token, str) and bool(token.strip())}

@app.get("/api/config/agent", response_model=AgentConfigResponse)
def get_agent_config():
    cfg = load_server_config()
    agent = cfg.get("agent") or {}
    base_url = _normalize_openai_base_url(agent.get("base_url") or "")
    model = agent.get("model") or "gpt-4o"
    provider = agent.get("provider") or "openai_compatible"
    timeout_s = agent.get("timeout_s") or 15
    api_key = agent.get("api_key")
    return {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key_set": isinstance(api_key, str) and bool(api_key.strip()),
        "timeout_s": int(timeout_s),
    }

@app.post("/api/config/agent", response_model=AgentConfigResponse)
def update_agent_config(req: AgentConfigUpdateRequest):
    cfg = load_server_config()
    agent = cfg.get("agent") or {}

    provider = agent.get("provider") or "openai_compatible"
    base_url = agent.get("base_url") or "http://123.129.219.111:3000/v1"
    model = agent.get("model") or "gpt-4o"
    api_key = agent.get("api_key")
    timeout_s = agent.get("timeout_s") or 15

    if req.provider is not None and req.provider.strip():
        provider = req.provider.strip()
    if req.base_url is not None and req.base_url.strip():
        base_url = _normalize_openai_base_url(req.base_url.strip())
    if req.model is not None and req.model.strip():
        model = req.model.strip()

    if req.clear_api_key:
        api_key = None
    elif req.api_key is not None:
        k = req.api_key.strip()
        if k:
            api_key = k

    if req.timeout_s is not None:
        if req.timeout_s > 0:
            timeout_s = int(req.timeout_s)

    cfg["agent"] = {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "timeout_s": timeout_s,
    }
    save_server_config(cfg)
    apply_agent_env_from_config(cfg)
    return {
        "provider": provider,
        "base_url": _normalize_openai_base_url(base_url),
        "model": model,
        "api_key_set": isinstance(api_key, str) and bool(api_key.strip()),
        "timeout_s": timeout_s,
    }

class AgentTestRequest(BaseModel):
    provider: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    timeout_s: Optional[int] = None

import httpx

@app.post("/api/config/agent/test", response_model=AgentTestResponse)
async def test_agent_config(req: Optional[AgentTestRequest] = None):
    cfg = load_server_config()
    agent = cfg.get("agent") or {}
    
    base_url = agent.get("base_url") or ""
    if req and req.base_url and req.base_url.strip():
        base_url = req.base_url.strip()
    base_url = _normalize_openai_base_url(base_url)

    api_key = agent.get("api_key")
    if req and req.api_key is not None:
        api_key = req.api_key.strip()
    
    model = agent.get("model") or "gpt-4o"
    if req and req.model and req.model.strip():
        model = req.model.strip()

    timeout_s = int(agent.get("timeout_s") or 15)
    if req and req.timeout_s and req.timeout_s > 0:
        timeout_s = req.timeout_s

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if isinstance(api_key, str) and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    models_ok = False
    models_status: Optional[int] = None
    models_detail = ""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            r = await client.get(f"{base_url}/models", headers=headers)
            if r.status_code == 200:
                models_ok = True
                models_status = r.status_code
                models_detail = "GET /models ok"
        except Exception:
            pass

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
        }
        
        try:
            r = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            if 200 <= r.status_code < 300:
                detail = "POST /chat/completions ok"
                if models_ok:
                    detail = f"{models_detail}; {detail}"
                return {"ok": True, "status_code": r.status_code, "detail": detail, "mode": "chat"}
            
            try:
                err_detail = r.json()
            except:
                err_detail = r.text[:200]
                
            if r.status_code in (401, 403):
                return {"ok": False, "status_code": r.status_code, "detail": f"Unauthorized: {err_detail}", "mode": "chat"}
            
            return {"ok": False, "status_code": r.status_code, "detail": f"Request failed: {err_detail}", "mode": "chat"}
        except Exception as e:
            if models_ok:
                return {"ok": False, "status_code": models_status, "detail": f"{models_detail}; chat failed: {e}", "mode": "chat"}
            return {"ok": False, "status_code": None, "detail": f"Connection error: {e}", "mode": "chat"}

class StartWorkflowRequest(BaseModel):
    user_query: str
    target_model_name: str
    target_model_path: str
    language: str = "zh"
    tensor_parallel_size: int = 1
    max_tokens: int = 2048
    use_rag: bool = True
    local_count: int = 3
    hf_count: int = 2
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9

class ResumeWorkflowRequest(BaseModel):
    thread_id: str
    action: str = "approved"  # or "rejected", etc.
    feedback: Optional[str] = None
    selected_benches: Optional[List[str]] = None
    state_updates: Optional[Dict[str, Any]] = None # For manual config modifications

class RedownloadBenchRequest(BaseModel):
    bench_name: str
    repo_id: Optional[str] = None
    config: Optional[str] = None
    split: Optional[str] = None
    force: bool = False

class RerunExecutionRequest(BaseModel):
    bench_name: Optional[str] = None
    state_updates: Optional[Dict[str, Any]] = None
    goto_confirm: bool = True

class ManualBenchRequest(BaseModel):
    bench_name: str
    dataset_cache: str
    bench_dataflow_eval_type: str
    meta: Optional[Dict[str, Any]] = None

class ManualStartRequest(BaseModel):
    user_query: str = "manual eval"
    target_model_name: Optional[str] = None
    language: str = "zh"
    target_model: Dict[str, Any]
    benches: List[ManualBenchRequest]

class WorkflowStatusResponse(BaseModel):
    thread_id: str
    status: str # "running", "interrupted", "completed", "failed", "idle"
    current_node: Optional[str] = None
    state_values: Optional[Dict[str, Any]] = None
    next_node: Optional[str] = None

class HistoryItem(BaseModel):
    thread_id: str
    updated_at: str
    user_query: Optional[str] = None
    status: str

# ... (Previous imports)

@app.post("/api/workflow/start")
async def start_workflow(req: StartWorkflowRequest):
    thread_id = str(uuid.uuid4())
    _set_thread_created_at(thread_id)
    log.info(f"Starting workflow for thread_id={thread_id}")

    # Initialize State
    initial_state = NodeState(
        user_query=req.user_query,
        target_model_name=req.target_model_name,
        request=MainRequest(language=req.language),
        use_rag=req.use_rag,
        local_count=req.local_count,
        hf_count=req.hf_count,
        target_model=ModelConfig(
            model_name_or_path=req.target_model_path,
            tensor_parallel_size=req.tensor_parallel_size,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_model_len=req.max_model_len,
            gpu_memory_utilization=req.gpu_memory_utilization,
        )
    )
    
    _launch_graph_task(thread_id, initial_state)
    
    return {"thread_id": thread_id, "status": "started"}

async def run_graph_background(thread_id: str, input_state: Any, resume_command: Optional[Command] = None):
    # Ensure env is fresh (though we set it at top level, dynamic updates might need this)
    apply_hf_env_from_config(load_server_config())
    apply_agent_env_from_config(load_server_config())
    
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            _touch_thread_updated_at(thread_id)
            log.info(f"Invoking graph for {thread_id}")
            if resume_command:
                # If resume_command is passed, we assume state updates were handled before calling this if needed
                result = await graph.ainvoke(resume_command, config=config)
            else:
                result = await graph.ainvoke(input_state, config=config)

            # Check if workflow was interrupted
            if result and "__interrupt__" in result:
                log.info(f"Graph interrupted for {thread_id}, interrupts: {result['__interrupt__']}")
                _thread_interrupt_cache[thread_id] = True
                _touch_thread_updated_at(thread_id)
            else:
                log.info(f"Graph execution finished for {thread_id}")
                _thread_interrupt_cache.pop(thread_id, None)
                _touch_thread_updated_at(thread_id)
        except asyncio.CancelledError:
            log.warning(f"Graph execution cancelled by user for {thread_id}")
            _touch_thread_updated_at(thread_id)
            raise
        except Exception as e:
            log.error(f"Error executing graph for {thread_id}: {e}")
            _touch_thread_updated_at(thread_id)
        finally:
            clear_progress(thread_id)
            task = RUNNING_WORKFLOW_TASKS.get(thread_id)
            if task is asyncio.current_task():
                RUNNING_WORKFLOW_TASKS.pop(thread_id, None)

def _launch_graph_task(thread_id: str, input_state: Any = None, resume_command: Optional[Command] = None):
    old = RUNNING_WORKFLOW_TASKS.get(thread_id)
    if old and not old.done():
        log.warning(f"Cancelling existing task for {thread_id} because a new task is being launched. New State: {bool(input_state)}, Resume: {resume_command}")
        old.cancel()
    task = asyncio.create_task(run_graph_background(thread_id, input_state, resume_command=resume_command))
    RUNNING_WORKFLOW_TASKS[thread_id] = task
    return task

@app.post("/api/workflow/stop/{thread_id}")
async def stop_workflow(thread_id: str):
    task = RUNNING_WORKFLOW_TASKS.get(thread_id)
    if not task:
        log.info(f"Stop request for {thread_id}, but no running task found.")
        return {"thread_id": thread_id, "status": "idle", "detail": "no running workflow"}
    if task.done():
        RUNNING_WORKFLOW_TASKS.pop(thread_id, None)
        log.info(f"Stop request for {thread_id}, task already finished.")
        return {"thread_id": thread_id, "status": "idle", "detail": "workflow already finished"}
    
    log.warning(f"Stop request received for {thread_id}. Cancelling task...")
    task.cancel()
    return {"thread_id": thread_id, "status": "stopping"}

@app.get("/api/workflow/status/{thread_id}")
async def get_status(thread_id: str):
    """
    获取工作流状态。

    解决 interrupt() 执行期间的竞态条件：
    当 next=() 且 interrupts=[] 但有 benches 数据时，
    可能是 interrupt() 正在执行中，需要短暂等待并重试。
    """
    import asyncio

    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # 竞态重试：interrupt() 写入 checkpoint 需要一点时间
        # 最多重试 8 次，间隔从 100ms 线性增加到 300ms
        max_retries = 8
        retry_delays = [0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3]

        for attempt in range(max_retries):
            try:
                snap = await graph.aget_state(config)
            except Exception as e:
                log.error(f"Failed to get state for {thread_id}: {e}")
                return {"thread_id": thread_id, "status": "not_found"}

            if not snap or (not snap.values and not snap.next):
                return {"thread_id": thread_id, "status": "idle"}

            next_nodes = snap.next
            current_values = snap.values or {}
            interrupts = snap.interrupts

            INTERRUPT_NODES = ("HumanReviewNode", "PreEvalReviewNode", "MetricReviewNode")

            # 检测中断
            is_interrupted = bool(interrupts and len(interrupts) > 0)
            if not is_interrupted and next_nodes:
                is_interrupted = any(node in next_nodes for node in INTERRUPT_NODES)

            # 判断状态
            if not next_nodes and not is_interrupted:
                # 优先查内存缓存：ainvoke 返回时已确认有 interrupt
                if _thread_interrupt_cache.get(thread_id):
                    status = "interrupted"
                else:
                    # next=() 且没检测到中断——可能是竞态窗口：
                    # background task 仍在运行（ainvoke 尚未返回），或
                    # interrupt() 已触发但尚未持久化进 checkpoint
                    benches = current_values.get("benches", [])
                    has_phase2_data = bool(current_values.get("eval_results"))
                    task = RUNNING_WORKFLOW_TASKS.get(thread_id)
                    task_still_running = task is not None and not task.done()

                    if (task_still_running or (benches and not has_phase2_data)) and attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        log.info(f"[get_status] Race condition detected (attempt {attempt+1}, task_running={task_still_running}), retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    status = "completed"
            elif is_interrupted:
                status = "interrupted"
            else:
                status = "running"

            log.info(f"[get_status] thread_id={thread_id}, status={status}, next={next_nodes}, interrupts={len(interrupts) if interrupts else 0}")

            return {
                "thread_id": thread_id,
                "status": status,
                "next_node": next_nodes,
                "state_values": current_values,
                "interrupts": [{"value": i.value} for i in interrupts] if interrupts else [],
                "eval_progress": get_progress(thread_id),
            }

        return {"thread_id": thread_id, "status": "completed"}

@app.post("/api/workflow/resume/{thread_id}")
async def resume_workflow(thread_id: str, req: ResumeWorkflowRequest):
    req.thread_id = thread_id
    # Apply state updates if provided
    if req.state_updates:
        if "target_model" in req.state_updates and isinstance(req.state_updates["target_model"], dict):
            try:
                tm = req.state_updates["target_model"]
                model_name_or_path = (
                    tm.get("model_name_or_path")
                    or tm.get("path")
                    or tm.get("model_path")
                    or tm.get("hf_model_name_or_path")
                )
                if not model_name_or_path:
                    raise ValueError("target_model missing model_name_or_path/path")

                req.state_updates["target_model"] = ModelConfig(
                    model_name_or_path=str(model_name_or_path),
                    is_api=bool(tm.get("is_api", False)),
                    api_url=tm.get("api_url"),
                    api_key=tm.get("api_key"),
                    temperature=float(tm.get("temperature", 0.0) or 0.0),
                    top_p=float(tm.get("top_p", 1.0) or 1.0),
                    top_k=int(tm.get("top_k", -1) if tm.get("top_k", -1) is not None else -1),
                    repetition_penalty=float(tm.get("repetition_penalty", 1.0) or 1.0),
                    max_tokens=int(tm.get("max_tokens", 2048) or 2048),
                    seed=(int(tm["seed"]) if tm.get("seed") is not None else None),
                    tensor_parallel_size=int(tm.get("tensor_parallel_size", 1) or 1),
                    max_model_len=tm.get("max_model_len"),
                    gpu_memory_utilization=float(tm.get("gpu_memory_utilization", 0.9) or 0.9),
                )
            except Exception as e:
                log.error(f"Failed to parse target_model update: {e}")
                del req.state_updates["target_model"]

        # Deserialize nested objects if needed
        if "benches" in req.state_updates and isinstance(req.state_updates["benches"], list):
            # Convert dicts back to BenchInfo objects
            benches_data = req.state_updates["benches"]
            incoming_benches = [
                _coerce_bench_info(b) if isinstance(b, dict) else b 
                for b in benches_data
            ]
            async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
                graph = build_complete_workflow(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": req.thread_id}}
                try:
                    snap = await graph.aget_state(config)
                except Exception:
                    snap = None
                current_values = snap.values if snap and getattr(snap, "values", None) else {}
                current_any = (current_values or {}).get("benches") or []
                current_benches = []
                for b in current_any:
                    try:
                        current_benches.append(_coerce_bench_info(b) if isinstance(b, dict) else b)
                    except Exception:
                        continue
                req.state_updates["benches"] = _merge_benches_preserve_runtime(incoming_benches, current_benches)

        async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
            graph = build_complete_workflow(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": req.thread_id}}
            log.info(f"Applying state updates for {req.thread_id}: {req.state_updates.keys()}")
            await graph.aupdate_state(config, req.state_updates)

    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": req.thread_id}}
        try:
            snap = await graph.aget_state(config)
        except Exception:
            raise HTTPException(status_code=404, detail="thread not found")
        next_nodes = snap.next or []
        values = snap.values or {}
        if req.action == "approved" and "PreEvalReviewNode" in next_nodes:
            benches_any = values.get("benches") or []
            missing = []
            invalid = []
            for b in benches_any:
                bench_name = None
                eval_type = None
                if isinstance(b, dict):
                    bench_name = b.get("bench_name")
                    eval_type = b.get("bench_dataflow_eval_type")
                    if not eval_type and isinstance(b.get("meta"), dict):
                        eval_type = b["meta"].get("bench_dataflow_eval_type")
                else:
                    bench_name = getattr(b, "bench_name", None)
                    eval_type = getattr(b, "bench_dataflow_eval_type", None)
                    meta = getattr(b, "meta", None)
                    if not eval_type and isinstance(meta, dict):
                        eval_type = meta.get("bench_dataflow_eval_type")
                if not eval_type:
                    missing.append(str(bench_name or "unknown"))
                elif str(eval_type).strip() not in _VALID_EVAL_TYPES:
                    invalid.append(f"{str(bench_name or 'unknown')}({str(eval_type).strip()})")
            if missing:
                raise HTTPException(status_code=400, detail=f"missing eval_type for benches: {', '.join(missing)}")
            if invalid:
                raise HTTPException(status_code=400, detail=f"invalid eval_type for benches: {', '.join(invalid)}")

    command = Command(resume=req.action)

    _thread_interrupt_cache.pop(req.thread_id, None)
    _launch_graph_task(req.thread_id, None, resume_command=command)
    return {"status": "resuming"}

@app.post("/api/workflow/rerun_execution/{thread_id}")
async def rerun_execution(thread_id: str, req: RerunExecutionRequest):
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        try:
            snap = await graph.aget_state(config)
        except Exception:
            raise HTTPException(status_code=404, detail="thread not found")

        if not snap or not snap.values:
            raise HTTPException(status_code=404, detail="thread not found")

        if req.state_updates:
            updates = dict(req.state_updates)

            if "target_model" in updates and isinstance(updates["target_model"], dict):
                try:
                    tm = updates["target_model"]
                    model_name_or_path = (
                        tm.get("model_name_or_path")
                        or tm.get("path")
                        or tm.get("model_path")
                        or tm.get("hf_model_name_or_path")
                    )
                    if not model_name_or_path:
                        raise ValueError("target_model missing model_name_or_path/path")

                    updates["target_model"] = ModelConfig(
                        model_name_or_path=str(model_name_or_path),
                        is_api=bool(tm.get("is_api", False)),
                        api_url=tm.get("api_url"),
                        api_key=tm.get("api_key"),
                        temperature=float(tm.get("temperature", 0.0) or 0.0),
                        top_p=float(tm.get("top_p", 1.0) or 1.0),
                        top_k=int(tm.get("top_k", -1) if tm.get("top_k", -1) is not None else -1),
                        repetition_penalty=float(tm.get("repetition_penalty", 1.0) or 1.0),
                        max_tokens=int(tm.get("max_tokens", 2048) or 2048),
                        seed=(int(tm["seed"]) if tm.get("seed") is not None else None),
                        tensor_parallel_size=int(tm.get("tensor_parallel_size", 1) or 1),
                        max_model_len=tm.get("max_model_len"),
                        gpu_memory_utilization=float(tm.get("gpu_memory_utilization", 0.9) or 0.9),
                    )
                except Exception as e:
                    log.error(f"Failed to parse target_model update: {e}")
                    updates.pop("target_model", None)

            if "benches" in updates and isinstance(updates["benches"], list):
                benches_data = updates["benches"]
                updates["benches"] = [
                    _coerce_bench_info(b) if isinstance(b, dict) else b
                    for b in benches_data
                ]

            log.info(f"Applying rerun state updates for {thread_id}: {list(updates.keys())}")
            await graph.aupdate_state(config, updates)

        snap = await graph.aget_state(config)
        values = snap.values or {}
        benches_any = values.get("benches") or []
        if not isinstance(benches_any, list):
            raise HTTPException(status_code=400, detail="invalid state benches")

        benches_list: List[BenchInfo] = []
        for b in benches_any:
            if isinstance(b, BenchInfo):
                benches_list.append(b)
            elif isinstance(b, dict):
                benches_list.append(_coerce_bench_info(b))

        for b in benches_list:
            if req.bench_name and b.bench_name != req.bench_name:
                continue
            b.eval_status = "pending"
            if b.meta is None:
                b.meta = {}
            if isinstance(b.meta, dict):
                for k in ("eval_result", "eval_detail_path", "eval_error", "eval_abnormality"):
                    b.meta.pop(k, None)

        await graph.aupdate_state(config, {"benches": benches_list, "eval_cursor": 0})

    goto_node = "PreEvalReviewNode" if req.goto_confirm else "DataFlowEvalNode"
    _launch_graph_task(thread_id, None, resume_command=Command(goto=goto_node))
    return {"ok": True, "status": "queued", "goto": goto_node}

@app.post("/api/workflow/manual_start")
async def manual_start(req: ManualStartRequest):
    thread_id = str(uuid.uuid4())
    _set_thread_created_at(thread_id)

    tm = req.target_model or {}
    model_name_or_path = (
        tm.get("model_name_or_path")
        or tm.get("path")
        or tm.get("model_path")
        or tm.get("hf_model_name_or_path")
    )
    if not model_name_or_path:
        raise HTTPException(status_code=400, detail="target_model missing model_name_or_path/path")

    model_cfg = ModelConfig(
        model_name_or_path=str(model_name_or_path),
        is_api=bool(tm.get("is_api", False)),
        api_url=tm.get("api_url"),
        api_key=tm.get("api_key"),
        temperature=float(tm.get("temperature", 0.0) or 0.0),
        top_p=float(tm.get("top_p", 1.0) or 1.0),
        top_k=int(tm.get("top_k", -1) if tm.get("top_k", -1) is not None else -1),
        repetition_penalty=float(tm.get("repetition_penalty", 1.0) or 1.0),
        max_tokens=int(tm.get("max_tokens", 2048) or 2048),
        seed=(int(tm["seed"]) if tm.get("seed") is not None else None),
        tensor_parallel_size=int(tm.get("tensor_parallel_size", 1) or 1),
        max_model_len=tm.get("max_model_len"),
        gpu_memory_utilization=float(tm.get("gpu_memory_utilization", 0.9) or 0.9),
    )

    benches: List[BenchInfo] = []
    for b in req.benches:
        meta = b.meta or {}
        benches.append(
            BenchInfo(
                bench_name=b.bench_name,
                bench_dataflow_eval_type=b.bench_dataflow_eval_type,
                meta=meta,
                dataset_cache=b.dataset_cache,
                download_status="success" if b.dataset_cache else None,
                eval_status="pending",
            )
        )

    initial_state = NodeState(
        user_query=req.user_query,
        request=MainRequest(language=req.language),
        target_model_name=req.target_model_name or str(model_name_or_path),
        target_model=model_cfg,
        benches=benches,
        eval_cursor=0,
    )

    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        await graph.aupdate_state(
            config,
            {
                "user_query": initial_state.user_query,
                "target_model_name": initial_state.target_model_name,
                "target_model": initial_state.target_model,
                "benches": initial_state.benches,
                "eval_cursor": 0,
            },
        )

    _launch_graph_task(thread_id, None, resume_command=Command(goto="DataFlowEvalNode"))
    return {"thread_id": thread_id, "status": "started"}

def _bench_download_sync(bench: Dict[str, Any], *, repo_root: Path, overrides: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    def _pick_best_split(splits: List[str], preferred: str) -> str:
        if not splits:
            return preferred
        if preferred in splits:
            return preferred
        for cand in ("test", "validation", "dev", "val", "train"):
            if cand in splits:
                return cand
        fuzzy = [s for s in splits if "test" in s.lower()]
        if fuzzy:
            return fuzzy[0]
        fuzzy = [s for s in splits if "valid" in s.lower() or "dev" in s.lower()]
        if fuzzy:
            return fuzzy[0]
        return splits[0]

    meta = bench.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    bench["meta"] = meta

    if overrides.get("repo_id"):
        hf_meta = meta.get("hf_meta") or {}
        if not isinstance(hf_meta, dict):
            hf_meta = {}
        hf_meta["hf_repo"] = overrides["repo_id"]
        meta["hf_meta"] = hf_meta

    dl_config = meta.get("download_config") or {}
    if not isinstance(dl_config, dict):
        dl_config = {}
    if overrides.get("config"):
        dl_config["config"] = overrides["config"]
    if overrides.get("split"):
        dl_config["split"] = overrides["split"]
    if dl_config:
        meta["download_config"] = dl_config

    hf_repo = None
    if isinstance(meta.get("hf_meta"), dict):
        hf_repo = meta["hf_meta"].get("hf_repo")
    if not hf_repo:
        hf_repo = bench.get("bench_name") or bench.get("name") or ""

    if not dl_config:
        dl_config = {"config": "default", "split": "test"}
        meta["download_config"] = dl_config

    config_name = dl_config.get("config", "default")
    split_name = dl_config.get("split", "test")

    structure = meta.get("structure") or {}
    if isinstance(structure, dict) and structure.get("ok"):
        subsets = structure.get("subsets", [])
        if isinstance(subsets, list):
            available_configs = [s.get("subset") for s in subsets if isinstance(s, dict) and s.get("subset")]
            if available_configs and config_name not in available_configs:
                if "main" in available_configs:
                    config_name = "main"
                else:
                    config_name = available_configs[0]
            matched_subset = next((s for s in subsets if isinstance(s, dict) and s.get("subset") == config_name), None)
            raw_splits = (matched_subset or {}).get("splits", []) if isinstance(matched_subset, dict) else []
            available_splits: List[str] = []
            for sp in raw_splits:
                if isinstance(sp, dict) and sp.get("name"):
                    available_splits.append(str(sp.get("name")))
                elif isinstance(sp, str):
                    available_splits.append(sp)
            split_name = _pick_best_split(available_splits, split_name) if available_splits else split_name

    meta["download_config"] = {
        "config": config_name,
        "split": split_name,
        "reason": "auto-corrected by server",
    }

    cache_root = repo_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    safe_repo = str(hf_repo).replace("/", "__")
    filename = f"{safe_repo}__{config_name}__{split_name}.jsonl"
    output_path = cache_root / filename

    safe_bench = str(bench.get("bench_name") or "").replace("/", "__")
    if safe_bench:
        exact = list(cache_root.glob(f"*__{safe_bench}__{config_name}__{split_name}.jsonl"))
        candidates = exact or list(cache_root.glob(f"*__{safe_bench}__{config_name}__*.jsonl")) or list(cache_root.glob(f"*__{safe_bench}__*.jsonl"))
        candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
        if candidates:
            chosen = max(candidates, key=lambda p: p.stat().st_mtime)
            bench["dataset_cache"] = str(chosen)
            bench["download_status"] = "success"
            meta.pop("download_error", None)
            return bench

    if overrides.get("force") and output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass

    if output_path.exists() and output_path.stat().st_size > 0:
        bench["dataset_cache"] = str(output_path)
        bench["download_status"] = "success"
        meta.pop("download_error", None)
        return bench

    tool = HFDownloadTool(cache_dir=str(cache_root))
    last_err = ""
    ok = False
    for i in range(max_retries):
        res = tool.download_and_convert(
            repo_id=str(hf_repo),
            config_name=str(config_name),
            split=str(split_name),
            output_path=output_path,
        )
        if res.get("ok"):
            ok = True
            break
        last_err = res.get("error") or ""

    if ok and output_path.exists() and output_path.stat().st_size > 0:
        bench["dataset_cache"] = str(output_path)
        bench["download_status"] = "success"
        meta.pop("download_error", None)
    else:
        bench["download_status"] = "failed"
        meta["download_error"] = last_err or "download failed"
    return bench

def _bench_from_dict(b: Any) -> BenchInfo:
    """从前端传来的 dict 安全构建 BenchInfo，过滤掉 BenchInfo 不认识的字段"""
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(BenchInfo)}
    filtered = {k: v for k, v in b.items() if k in valid_fields}
    return BenchInfo(**filtered)

def _bench_to_dict(b: Any) -> Optional[Dict[str, Any]]:
    if b is None:
        return None
    if isinstance(b, dict):
        return b
    if hasattr(b, "__dict__"):
        d = dict(getattr(b, "__dict__", {}) or {})
        return d if isinstance(d, dict) else None
    return None

_BENCHINFO_FIELDS = {f.name for f in fields(BenchInfo)}
_VALID_EVAL_TYPES = {
    "key1_text_score",
    "key2_qa",
    "key2_q_ma",
    "key3_q_choices_a",
    "key3_q_choices_as",
    "key3_q_a_rejected",
}
_RUNTIME_META_KEYS = {
    "eval_result",
    "eval_detail_path",
    "eval_step3_path",
    "eval_progress",
    "eval_error",
    "eval_abnormality",
    "pred_key",
    "ref_key",
    "artifact_paths",
}

def _coerce_bench_info(value: Any) -> BenchInfo:
    if isinstance(value, BenchInfo):
        return value
    if not isinstance(value, dict):
        raise ValueError("invalid bench payload")
    b = dict(value)
    if not b.get("bench_dataflow_eval_type") and isinstance(b.get("eval_type"), str):
        b["bench_dataflow_eval_type"] = b.get("eval_type")
    if isinstance(b.get("bench_dataflow_eval_type"), str):
        et = b.get("bench_dataflow_eval_type").strip()
        if not et or et == "unknown":
            b["bench_dataflow_eval_type"] = None
    b.pop("eval_type", None)
    filtered = {k: v for k, v in b.items() if k in _BENCHINFO_FIELDS}
    return BenchInfo(**filtered)

def _is_empty_like(v: Any) -> bool:
    return v is None or v == "" or v == {} or v == []

def _merge_benches_preserve_runtime(incoming: List[BenchInfo], current: List[BenchInfo]) -> List[BenchInfo]:
    current_map = {b.bench_name: b for b in (current or []) if isinstance(b, BenchInfo) and b.bench_name}
    merged: List[BenchInfo] = []

    for b in incoming or []:
        if not isinstance(b, BenchInfo):
            merged.append(b)
            continue
        cur = current_map.get(b.bench_name)
        if not cur:
            merged.append(b)
            continue

        if _is_empty_like(getattr(b, "eval_status", None)) and not _is_empty_like(getattr(cur, "eval_status", None)):
            b.eval_status = cur.eval_status
        if _is_empty_like(getattr(b, "dataset_cache", None)) and not _is_empty_like(getattr(cur, "dataset_cache", None)):
            b.dataset_cache = cur.dataset_cache
        if _is_empty_like(getattr(b, "download_status", None)) and not _is_empty_like(getattr(cur, "download_status", None)):
            b.download_status = cur.download_status

        meta_new = b.meta if isinstance(b.meta, dict) else {}
        meta_cur = cur.meta if isinstance(cur.meta, dict) else {}
        for k in _RUNTIME_META_KEYS:
            if _is_empty_like(meta_new.get(k)) and not _is_empty_like(meta_cur.get(k)):
                meta_new[k] = meta_cur.get(k)
        b.meta = meta_new
        merged.append(b)

    return merged

async def _redownload_bench_background(thread_id: str, bench_name: str, overrides: Dict[str, Any]):
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        snap = await graph.aget_state(config)
        if not snap or not snap.values:
            return
        values = snap.values
        benches_any = values.get("benches") or []
        if not isinstance(benches_any, list):
            return
        benches = [x for x in (_bench_to_dict(b) for b in benches_any) if isinstance(x, dict)]

        idx = None
        for i, b in enumerate(benches):
            if b.get("bench_name") == bench_name:
                idx = i
                break
        if idx is None:
            return

        bench = benches[idx]
        updated = await asyncio.to_thread(_bench_download_sync, bench, repo_root=REPO_ROOT, overrides=overrides)
        benches[idx] = updated
        await graph.aupdate_state(config, {"benches": [_coerce_bench_info(b) for b in benches]})

@app.post("/api/workflow/redownload/{thread_id}")
async def redownload_bench(thread_id: str, req: RedownloadBenchRequest):
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snap = await graph.aget_state(config)
        except Exception:
            raise HTTPException(status_code=404, detail="thread not found")

        if not snap or not snap.values:
            raise HTTPException(status_code=404, detail="thread not found")

        values = snap.values
        benches_any = values.get("benches") or []
        if not isinstance(benches_any, list):
            raise HTTPException(status_code=400, detail="invalid state benches")
        benches = [x for x in (_bench_to_dict(b) for b in benches_any) if isinstance(x, dict)]

        idx = None
        for i, b in enumerate(benches):
            if b.get("bench_name") == req.bench_name:
                idx = i
                break
        if idx is None:
            raise HTTPException(status_code=404, detail="bench not found")

        bench = benches[idx]

        bench["download_status"] = "pending"
        meta = bench.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        meta.pop("download_error", None)
        bench["meta"] = meta
        benches[idx] = bench
        await graph.aupdate_state(config, {"benches": [_coerce_bench_info(b) for b in benches]})

    overrides = {"repo_id": req.repo_id, "config": req.config, "split": req.split, "force": req.force}
    asyncio.create_task(_redownload_bench_background(thread_id, req.bench_name, overrides))
    return {"ok": True, "status": "queued"}

@app.get("/api/workflow/history", response_model=List[HistoryItem])
async def get_history():
    if not DB_PATH.exists():
        return []
        
    items = []
    thread_meta = _load_thread_meta()
    meta_dirty = False
    try:
        # Optimize: Reuse single connection/checkpointer for all lookups
        async with get_checkpointer(DB_PATH, mode="run") as cp:
            # 1. Get thread_ids using the same connection
            async with cp.conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC LIMIT 50") as cursor:
                rows = await cursor.fetchall()

            # 2. Build graph once
            graph = build_complete_workflow(checkpointer=cp)
            
            for (tid,) in rows:
                cfg = {"configurable": {"thread_id": tid}}
                try:
                    # aget_state uses the passed checkpointer (cp)
                    snap = await graph.aget_state(cfg)
                    if snap and snap.values:
                        q = snap.values.get("user_query", "Unknown Query")
                        # Determine status
                        status = "completed"
                        if snap.next:
                            status = "interrupted" if ("HumanReviewNode" in snap.next or "PreEvalReviewNode" in snap.next or "MetricReviewNode" in snap.next) else "running"
                        # If no next and no error -> completed
                        
                        ts = snap.metadata.get("created_at") if snap.metadata else None
                        meta_item = thread_meta.get(tid) if isinstance(thread_meta.get(tid), dict) else {}
                        created_ts = meta_item.get("created_at")
                        if not created_ts and isinstance(ts, str) and ts.strip():
                            created_ts = ts.strip()
                        if not created_ts:
                            legacy_ts = meta_item.get("updated_at")
                            if isinstance(legacy_ts, str) and legacy_ts.strip():
                                created_ts = legacy_ts.strip()
                        if not created_ts:
                            created_ts = "1970-01-01T00:00:00+00:00"
                        if not meta_item.get("created_at"):
                            next_meta = dict(meta_item)
                            next_meta["created_at"] = created_ts
                            if not next_meta.get("updated_at"):
                                next_meta["updated_at"] = created_ts
                            thread_meta[tid] = next_meta
                            meta_dirty = True
                        date_str = str(created_ts)
                        
                        items.append(HistoryItem(
                            thread_id=tid,
                            updated_at=str(date_str),
                            user_query=str(q),
                            status=status
                        ))
                except Exception:
                    pass
    except Exception as e:
        log.error(f"Error fetching history: {e}")
        return []
    if meta_dirty:
        _write_json_file(THREAD_META_FILE, thread_meta)

    def _parse_dt(v: str):
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except Exception:
            return datetime(1970, 1, 1)

    items.sort(key=lambda x: _parse_dt(x.updated_at), reverse=True)
    return items


@app.get("/api/models")
def get_models():
    models = _load_json_file(MODELS_FILE, default=[])
    return models if isinstance(models, list) else []

@app.post("/api/models")
def add_model(model: Dict[str, Any]):
    models = _load_json_file(MODELS_FILE, default=[])
    if not isinstance(models, list):
        models = []
    models.append(model)
    _write_json_file(MODELS_FILE, models)
    return {"status": "success"}

class ModelLoadTestRequest(BaseModel):
    model_path: str
    tensor_parallel_size: int = 1
    max_tokens: int = 32

@app.post("/api/models/test_load")
def test_model_load(req: ModelLoadTestRequest):
    raw = (req.model_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="model_path is required")
    resolved = _normalize_model_path_for_host(raw)
    exists_local = Path(resolved).exists()
    if not exists_local and (":" in raw or raw.startswith("/mnt/")):
        raise HTTPException(status_code=400, detail=f"Model path not found on current host: {resolved}")
    try:
        from dataflow.serving.local_model_llm_serving import LocalModelLLMServing_vllm
        serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path=resolved,
            vllm_tensor_parallel_size=max(1, int(req.tensor_parallel_size or 1)),
            vllm_max_tokens=max(1, int(req.max_tokens or 32)),
            vllm_temperature=0.0,
        )
        serving.start_serving()
        has_tokenizer = hasattr(serving, "tokenizer")
        serving.cleanup()
        if not has_tokenizer:
            raise RuntimeError("serving started but tokenizer is missing")
        return {"ok": True, "detail": f"Model load test passed: {resolved}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model load test failed: {e}")

@app.get("/api/benches/gallery")
def get_bench_gallery():
    return bench_registry.get_all_benches()


@app.get("/api/metrics/registry")
def get_metrics_registry():
    """获取所有注册的 Metric 元数据"""
    return get_registered_metrics_meta()


class AddBenchRequest(BaseModel):
    bench_name: str
    type: str  # 如 "language & reasoning", "safety", "code" 等
    description: str
    dataset_url: Optional[str] = None


@app.post("/api/benches/gallery")
def add_bench_to_gallery(req: AddBenchRequest):
    """添加新的 benchmark 到 gallery"""
    # 构建完整的 bench 数据结构
    bench_data = {
        "bench_name": req.bench_name,
        "bench_table_exist": False,  # 用户添加的默认为 False
        "bench_source_url": req.dataset_url or f"https://huggingface.co/datasets/{req.bench_name}",
        "bench_dataflow_eval_type": None,
        "bench_prompt_template": None,
        "bench_keys": [],
        "meta": {
            "bench_name": req.bench_name,
            "source": "user_added",
            "aliases": [req.bench_name],
            "category": "General",
            "tags": [req.type],
            "description": req.description,
            "hf_meta": {
                "bench_name": req.bench_name,
                "hf_repo": req.bench_name,
                "card_text": "",
                "tags": [req.type],
                "exists_on_hf": True
            }
        }
    }

    success = bench_registry.add_bench(bench_data, str(BENCH_GALLERY_PATH))
    if success:
        return {"status": "success", "bench": bench_data}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to add bench. It may already exist.")


if __name__ == "__main__":
    import uvicorn
    # Disable uvloop to allow nest_asyncio patching in synchronous nodes
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
