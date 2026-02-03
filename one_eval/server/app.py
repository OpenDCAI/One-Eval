
import asyncio
import uuid
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from one_eval.logger import get_logger

log = get_logger("OneEval-Server")

# === Early Environment Setup ===
# Must be done before importing langgraph/transformers/etc. to ensure env vars take effect
SERVER_DIR = Path(__file__).resolve().parent
DATA_DIR = SERVER_DIR / "_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = DATA_DIR / "config.json"
MODELS_FILE = DATA_DIR / "models.json"
# SERVER_DIR is .../one_eval/server
# parents[0]=one_eval, parents[1]=One-Eval (Repo Root)
REPO_ROOT = SERVER_DIR.parents[1]
ENV_FILE = REPO_ROOT / "env.sh"

# Original DB location was parents[2] (scy/checkpoints)
# We keep it there or move it? 
# If previous code used parents[2], we should respect it to find existing DB.
DB_PATH = (SERVER_DIR.parents[2] / "checkpoints" / "eval.db").resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    if isinstance(base_url, str) and base_url.strip():
        os.environ["OE_API_BASE"] = _normalize_openai_base_url(base_url.strip())
        os.environ["DF_API_BASE_URL"] = _normalize_openai_base_url(base_url.strip())
    if isinstance(api_key, str) and api_key.strip():
        os.environ["OE_API_KEY"] = api_key.strip()
        os.environ["DF_API_KEY"] = api_key.strip()
    if isinstance(model, str) and model.strip():
        os.environ["DF_MODEL_NAME"] = model.strip()

# Initialize Env ASAP
_cfg0 = load_server_config()
log.info(f"Loaded server config: {_cfg0}")
if not CONFIG_FILE.exists():
    save_server_config(_cfg0)
apply_hf_env_from_config(_cfg0)
apply_agent_env_from_config(_cfg0)

from one_eval.graph.workflow_all import build_complete_workflow
from one_eval.utils.checkpoint import get_checkpointer
from one_eval.core.state import NodeState, ModelConfig, BenchInfo
from one_eval.utils.deal_json import _save_state_json
from langgraph.types import Command
from one_eval.utils.bench_registry import BenchRegistry

# Bench Registry - 使用 bench_gallery.json 作为数据源
BENCH_GALLERY_PATH = REPO_ROOT / "one_eval" / "bench_gallery.json"
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

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            r = await client.get(f"{base_url}/models", headers=headers)
            if r.status_code == 200:
                return {"ok": True, "status_code": 200, "detail": "GET /models ok", "mode": "models"}
        except Exception:
            pass

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
        }
        
        try:
            r = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            if 200 <= r.status_code < 300:
                return {"ok": True, "status_code": r.status_code, "detail": "POST /chat/completions ok", "mode": "chat"}
            
            try:
                err_detail = r.json()
            except:
                err_detail = r.text[:200]
                
            if r.status_code in (401, 403):
                return {"ok": False, "status_code": r.status_code, "detail": f"Unauthorized: {err_detail}", "mode": "chat"}
            
            return {"ok": False, "status_code": r.status_code, "detail": f"Request failed: {err_detail}", "mode": "chat"}
        except Exception as e:
            return {"ok": False, "status_code": None, "detail": f"Connection error: {e}", "mode": "chat"}

class StartWorkflowRequest(BaseModel):
    user_query: str
    target_model_name: str
    target_model_path: str
    tensor_parallel_size: int = 1
    max_tokens: int = 2048
    use_rag: bool = True

class ResumeWorkflowRequest(BaseModel):
    thread_id: str
    action: str = "approved"  # or "rejected", etc.
    feedback: Optional[str] = None
    selected_benches: Optional[List[str]] = None
    state_updates: Optional[Dict[str, Any]] = None # For manual config modifications

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
    log.info(f"Starting workflow for thread_id={thread_id}")

    # Initialize State
    initial_state = NodeState(
        user_query=req.user_query,
        target_model_name=req.target_model_name,
        use_rag=req.use_rag,
        target_model=ModelConfig(
            model_name_or_path=req.target_model_path,
            tensor_parallel_size=req.tensor_parallel_size,
            max_tokens=req.max_tokens
        )
    )
    
    asyncio.create_task(run_graph_background(thread_id, initial_state))
    
    return {"thread_id": thread_id, "status": "started"}

async def run_graph_background(thread_id: str, input_state: Any, resume_command: Optional[Command] = None):
    # Ensure env is fresh (though we set it at top level, dynamic updates might need this)
    apply_hf_env_from_config(load_server_config())
    apply_agent_env_from_config(load_server_config())
    
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            log.info(f"Invoking graph for {thread_id}")
            if resume_command:
                # If resume_command is passed, we assume state updates were handled before calling this if needed
                result = await graph.ainvoke(resume_command, config=config)
            else:
                result = await graph.ainvoke(input_state, config=config)

            # Check if workflow was interrupted
            if result and "__interrupt__" in result:
                log.info(f"Graph interrupted for {thread_id}, interrupts: {result['__interrupt__']}")
            else:
                log.info(f"Graph execution finished for {thread_id}")
        except Exception as e:
            log.error(f"Error executing graph for {thread_id}: {e}")

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

        # 最多重试3次，每次间隔100ms
        max_retries = 3
        retry_delay = 0.1

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

            INTERRUPT_NODES = ("HumanReviewNode", "PreEvalReviewNode")

            # 检测中断
            is_interrupted = bool(interrupts and len(interrupts) > 0)
            if not is_interrupted and next_nodes:
                is_interrupted = any(node in next_nodes for node in INTERRUPT_NODES)

            # 判断状态
            if not next_nodes and not is_interrupted:
                # next=() 且没检测到中断
                # 检查是否在竞态窗口：有 benches 但没进入 Phase 2
                benches = current_values.get("benches", [])
                has_phase2_data = bool(current_values.get("eval_results"))

                if benches and not has_phase2_data and attempt < max_retries - 1:
                    log.info(f"[get_status] Race condition detected (attempt {attempt+1}), retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
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
                "interrupts": [{"value": i.value} for i in interrupts] if interrupts else []
            }

        return {"thread_id": thread_id, "status": "completed"}

@app.post("/api/workflow/resume/{thread_id}")
async def resume_workflow(req: ResumeWorkflowRequest):
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
                    max_tokens=int(tm.get("max_tokens", 2048) or 2048),
                    tensor_parallel_size=int(tm.get("tensor_parallel_size", 1) or 1),
                    max_model_len=tm.get("max_model_len"),
                )
            except Exception as e:
                log.error(f"Failed to parse target_model update: {e}")
                del req.state_updates["target_model"]

        # Deserialize nested objects if needed
        if "benches" in req.state_updates and isinstance(req.state_updates["benches"], list):
            # Convert dicts back to BenchInfo objects
            benches_data = req.state_updates["benches"]
            req.state_updates["benches"] = [
                BenchInfo(**b) if isinstance(b, dict) else b 
                for b in benches_data
            ]

        async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
            graph = build_complete_workflow(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": req.thread_id}}
            log.info(f"Applying state updates for {req.thread_id}: {req.state_updates.keys()}")
            await graph.aupdate_state(config, req.state_updates)

    command = Command(resume=req.action)
    
    asyncio.create_task(run_graph_background(req.thread_id, None, resume_command=command))
    return {"status": "resuming"}

@app.get("/api/workflow/history", response_model=List[HistoryItem])
async def get_history():
    import aiosqlite
    import datetime
    
    if not DB_PATH.exists():
        return []
        
    items = []
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # Query distinct thread_ids from checkpoints
            # Schema usually: thread_id, checkpoint_id, checkpoint, metadata...
            # We want the latest checkpoint for each thread.
            # But LangGraph 0.2 schema might differ.
            # Usually 'checkpoints' table has 'thread_id'.
            
            # Simple query to get unique threads
            cursor = await db.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC LIMIT 50")
            rows = await cursor.fetchall()
            
            for (tid,) in rows:
                # Get latest state for this thread to find query/status
                async with get_checkpointer(DB_PATH, mode="run") as cp:
                    graph = build_complete_workflow(checkpointer=cp)
                    cfg = {"configurable": {"thread_id": tid}}
                    try:
                        snap = await graph.aget_state(cfg)
                        if snap and snap.values:
                            q = snap.values.get("user_query", "Unknown Query")
                            # Determine status
                            status = "completed"
                            if snap.next:
                                status = "interrupted" if ("HumanReviewNode" in snap.next or "PreEvalReviewNode" in snap.next) else "running"
                            # If no next and no error -> completed
                            
                            ts = snap.metadata.get("created_at") if snap.metadata else None
                            # If not in metadata, use current time or skip
                            date_str = ts or datetime.datetime.now().isoformat()
                            
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

@app.get("/api/benches/gallery")
def get_bench_gallery():
    return bench_registry.get_all_benches()


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
    uvicorn.run(app, host="0.0.0.0", port=8000)
