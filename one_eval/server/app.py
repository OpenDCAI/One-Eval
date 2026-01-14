
import asyncio
import uuid
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from one_eval.graph.workflow_all import build_complete_workflow
from one_eval.utils.checkpoint import get_checkpointer
from one_eval.core.state import NodeState, ModelConfig
from one_eval.utils.deal_json import _save_state_json
from langgraph.types import Command

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OneEval-Server")

SERVER_DIR = Path(__file__).resolve().parent
DATA_DIR = SERVER_DIR / "_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = DATA_DIR / "config.json"
MODELS_FILE = DATA_DIR / "models.json"
DB_PATH = (SERVER_DIR.parents[2] / "checkpoints" / "eval.db").resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_json_file(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text())
    except Exception:
        return default

def _write_json_file(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def load_server_config() -> Dict[str, Any]:
    cfg = _load_json_file(CONFIG_FILE, default={})
    if not isinstance(cfg, dict):
        cfg = {}
    hf = cfg.get("hf")
    if not isinstance(hf, dict):
        hf = {}
    endpoint = hf.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        endpoint = "https://hf-mirror.com"
    token = hf.get("token")
    if token is not None and (not isinstance(token, str) or not token.strip()):
        token = None
    cfg["hf"] = {"endpoint": endpoint, "token": token}

    agent = cfg.get("agent")
    if not isinstance(agent, dict):
        agent = {}
    provider = agent.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        provider = "openai_compatible"
    base_url = agent.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        base_url = "http://123.129.219.111:3000/v1"
    model = agent.get("model")
    if not isinstance(model, str) or not model.strip():
        model = "gpt-4o"
    api_key = agent.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        api_key = None
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

_cfg0 = load_server_config()
if not CONFIG_FILE.exists():
    save_server_config(_cfg0)
apply_hf_env_from_config(_cfg0)
apply_agent_env_from_config(_cfg0)

# Models
class StartWorkflowRequest(BaseModel):
    user_query: str
    target_model_name: str
    target_model_path: str
    tensor_parallel_size: int = 1
    max_tokens: int = 2048

class ResumeWorkflowRequest(BaseModel):
    thread_id: str
    action: str = "approved"  # or "rejected", etc.
    feedback: Optional[str] = None
    selected_benches: Optional[List[str]] = None

class WorkflowStatusResponse(BaseModel):
    thread_id: str
    status: str # "running", "interrupted", "completed", "failed", "idle"
    current_node: Optional[str] = None
    state_values: Optional[Dict[str, Any]] = None
    next_node: Optional[str] = None

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

@app.post("/api/config/agent/test", response_model=AgentTestResponse)
def test_agent_config():
    cfg = load_server_config()
    agent = cfg.get("agent") or {}
    base_url = _normalize_openai_base_url(agent.get("base_url") or "")
    api_key = agent.get("api_key")
    model = agent.get("model") or "gpt-4o"
    timeout_s = int(agent.get("timeout_s") or 15)

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if isinstance(api_key, str) and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    try:
        r = requests.get(f"{base_url}/models", headers=headers, timeout=timeout_s)
        if r.status_code == 200:
            return {"ok": True, "status_code": 200, "detail": "GET /models ok", "mode": "models"}
    except Exception as e:
        pass

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout_s)
        if 200 <= r.status_code < 300:
            return {"ok": True, "status_code": r.status_code, "detail": "POST /chat/completions ok", "mode": "chat"}
        if r.status_code in (401, 403):
            return {"ok": False, "status_code": r.status_code, "detail": "reachable but unauthorized (check API key)", "mode": "chat"}
        return {"ok": False, "status_code": r.status_code, "detail": f"request failed: {r.text[:200]}", "mode": "chat"}
    except Exception as e:
        return {"ok": False, "status_code": None, "detail": f"connection error: {e}", "mode": "chat"}

@app.post("/api/workflow/start")
async def start_workflow(req: StartWorkflowRequest):
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting workflow for thread_id={thread_id}")

    # Initialize State
    initial_state = NodeState(
        user_query=req.user_query,
        target_model=req.target_model_name,
        target_model_config=ModelConfig(
            model_name_or_path=req.target_model_path,
            tensor_parallel_size=req.tensor_parallel_size,
            max_tokens=req.max_tokens
        )
    )

    # We launch the workflow in background? 
    # LangGraph is async. We can start it and let it run until first interrupt or end.
    # But since we need to persist state, we should run it.
    
    # Issue: ainvoke waits until end. We need non-blocking execution or just start it.
    # For a web server, we usually want to trigger the run and return immediately.
    # We'll use a background task wrapper.
    
    asyncio.create_task(run_graph_background(thread_id, initial_state))
    
    return {"thread_id": thread_id, "status": "started"}

async def run_graph_background(thread_id: str, input_state: Any, resume_command: Optional[Command] = None):
    apply_hf_env_from_config(load_server_config())
    apply_agent_env_from_config(load_server_config())
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            logger.info(f"Invoking graph for {thread_id}")
            if resume_command:
                await graph.ainvoke(resume_command, config=config)
            else:
                await graph.ainvoke(input_state, config=config)
            logger.info(f"Graph execution finished for {thread_id}")
        except Exception as e:
            logger.error(f"Error executing graph for {thread_id}: {e}")

@app.get("/api/workflow/status/{thread_id}")
async def get_status(thread_id: str):
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            snap = await graph.aget_state(config)
        except Exception:
            return {"thread_id": thread_id, "status": "not_found"}

        if not snap or (not snap.values and not snap.next):
             return {"thread_id": thread_id, "status": "idle"}
        
        # Check if interrupted
        next_nodes = snap.next
        current_values = snap.values
        
        # LangGraph state:
        # If next is empty and we have values, it might be done.
        # If next has value, it is waiting (e.g. interrupt) or ready to run next step?
        # With interrupts, graph stops.
        
        status = "running"
        if not next_nodes:
            status = "completed"
        elif "HumanReviewNode" in next_nodes: # Or whatever the interrupt node name is logic-wise
            # Our HumanReviewNode is an InterruptNode. 
            # If the graph stopped *before* it? Or *at* it?
            # InterruptNode raises NodeInterrupt. LangGraph catches it and stops.
            # The next node to run would be the one *after* interrupt? 
            # Actually, standard LangGraph interrupt logic:
            # If we use `interrupt_before=["HumanReviewNode"]`, next would be HumanReviewNode.
            # But our InterruptNode implementation inside `one_eval` likely returns a Command or uses checkpointer to pause?
            # Let's assume if we are stopped, we check tasks.
            status = "interrupted"

        return {
            "thread_id": thread_id,
            "status": status,
            "next_node": next_nodes,
            "state_values": current_values # Be careful with size
        }

@app.post("/api/workflow/resume/{thread_id}")
async def resume_workflow(req: ResumeWorkflowRequest):
    # Depending on how InterruptNode is implemented.
    # If it uses standard LangGraph interrupts, we resume with Command or update state.
    # Based on workflow_all.py:
    # out = await graph.ainvoke(Command(resume="approved"), config=config)
    
    command = Command(resume=req.action) # Assuming value is what InterruptNode expects
    
    # If user modified selected benches, we might need to update state *before* resuming?
    # Or pass it in resume value?
    # For now, let's assume we just approve.
    
    asyncio.create_task(run_graph_background(req.thread_id, None, resume_command=command))
    return {"status": "resuming"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
