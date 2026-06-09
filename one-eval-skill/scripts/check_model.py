#!/usr/bin/env python3
"""
check_model.py — 模型连通性测试（接入任何待评测模型前的必做门槛）。

为什么需要：评测数据量大、耗时长。先用一条极简 prompt 做端到端真实推理，
确认模型可达、凭证正确、返回正常，避免跑到一半才发现模型连不上。

支持两种模型：
  - API 模型（is_api=true）：openai_compatible / deepseek
  - 本地 vLLM 模型（is_api=false）：需要 GPU + vllm 环境

用法：
  # API 模型
  python check_model.py --api \
      --model gpt-4o-mini \
      --api-url http://host:3000/v1/chat/completions \
      --api-key sk-xxx

  # 本地 vLLM 模型
  python check_model.py --model /path/to/model --tensor-parallel-size 1

  # 或从 evalspec.yaml 读 model 段
  python check_model.py --spec evalspec.yaml

退出码：0 = 连通成功；非 0 = 失败（stderr 打印可读原因）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402

PROBE_PROMPT = "Reply with exactly: OK"

def _probe_api_lightweight(model_dict: dict) -> dict:
    """轻量 API 探测：纯 requests 直打 OpenAI 兼容端点。

    不依赖 one_eval / dataflow，因此在未装评测引擎的环境也能验证 API 连通性。
    """
    import requests

    api_url = common_normalize_api_url(model_dict)
    api_key = model_dict.get("api_key", "")
    model_name = model_dict["model_name_or_path"]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(
            api_url, headers=headers, json=payload,
            timeout=(float(model_dict.get("api_connect_timeout", 10.0)),
                     float(model_dict.get("api_read_timeout", 30.0))),
        )
    except requests.exceptions.ConnectTimeout:
        return {"ok": False, "reason": f"连接超时：无法在限定时间内连上 {api_url}（检查网络/URL/防火墙）"}
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "reason": f"连接失败：{api_url} 不可达（{e}）"}
    except Exception as e:
        return {"ok": False, "reason": f"请求异常：{type(e).__name__}: {e}"}

    if resp.status_code == 401:
        return {"ok": False, "reason": "鉴权失败(401)：api_key 无效或缺失"}
    if resp.status_code == 404:
        return {"ok": False, "reason": f"端点或模型不存在(404)：检查 api_url 与 model 名 {model_name!r}"}
    if resp.status_code == 429:
        return {"ok": False, "reason": "限流(429)：请求过于频繁，稍后重试"}
    if resp.status_code >= 400:
        return {"ok": False, "reason": f"HTTP {resp.status_code}：{resp.text[:200]}"}

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return {"ok": False, "reason": f"响应格式非 OpenAI 兼容：{resp.text[:200]}"}

    return {"ok": True, "reply": content, "via": "lightweight-requests"}


def common_normalize_api_url(model_dict: dict) -> str:
    """复用 one_eval 的 URL 规范化逻辑；不可用时退回本地最小实现。"""
    try:
        from one_eval.toolkits.dataflow_eval_tool import DataFlowEvalTool
        return DataFlowEvalTool._normalize_api_url(
            model_dict.get("api_url"),
            model_dict.get("api_provider", "openai_compatible"),
        )
    except Exception:
        raw = (model_dict.get("api_url") or "").strip()
        provider = (model_dict.get("api_provider") or "openai_compatible").lower()
        if not raw:
            return ("https://api.deepseek.com/chat/completions" if provider == "deepseek"
                    else "https://api.openai.com/v1/chat/completions")
        low = raw.lower().rstrip("/")
        if low.endswith("/chat/completions"):
            return raw.rstrip("/")
        if low.endswith("/v1"):
            return f"{raw.rstrip('/')}/chat/completions"
        return raw


def _probe_local_vllm(model_dict: dict) -> dict:
    """本地 vLLM 探测：通过 one_eval 的 DataFlowEvalTool 起 serving 并真实推理一条。

    需要 GPU + vllm 环境（本机 Mac 无法验证，留给 GPU 机）。
    """
    try:
        from one_eval.toolkits.dataflow_eval_tool import DataFlowEvalTool
    except Exception as e:
        return {"ok": False, "reason": f"无法导入评测引擎(one_eval/dataflow 未装好): {e}"}

    try:
        mc = common.build_model_config(model_dict)
        tool = DataFlowEvalTool(output_root=str(common.DEFAULT_OUTPUT_DIR / "_check"))
        tool._init_llm_serving(mc)
        serving = tool.llm_serving
        outputs = serving.generate_from_input(
            user_inputs=[PROBE_PROMPT], system_prompt="",
        )
        reply = outputs[0] if outputs else ""
        DataFlowEvalTool.release_serving()
        if not reply:
            return {"ok": False, "reason": "vLLM 起服务成功但推理返回空"}
        return {"ok": True, "reply": reply, "via": "vllm-serving"}
    except Exception as e:
        return {"ok": False, "reason": f"本地 vLLM 启动/推理失败：{type(e).__name__}: {e}"}


def check_model(model_dict: dict) -> dict:
    """统一入口：根据 is_api 选择探测方式。"""
    if not model_dict.get("model_name_or_path"):
        return {"ok": False, "reason": "缺少 model_name_or_path"}
    if model_dict.get("is_api"):
        return _probe_api_lightweight(model_dict)
    return _probe_local_vllm(model_dict)


def _parse_args(argv):
    p = argparse.ArgumentParser(description="模型连通性测试")
    p.add_argument("--spec", help="从 evalspec.yaml 读取 model 段")
    p.add_argument("--api", action="store_true", help="标记为 API 模型")
    p.add_argument("--model", help="model_name_or_path（API 模型名或本地路径）")
    p.add_argument("--api-url", dest="api_url")
    p.add_argument("--api-key", dest="api_key")
    p.add_argument("--api-provider", dest="api_provider", default="openai_compatible")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])
    if args.spec:
        spec = common.load_evalspec(args.spec)
        model_dict = spec.get("model", {})
    else:
        if not args.model:
            print("错误：需要 --model 或 --spec", file=sys.stderr)
            return 2
        model_dict = {
            "model_name_or_path": args.model,
            "is_api": bool(args.api),
            "api_url": args.api_url,
            "api_key": args.api_key,
            "api_provider": args.api_provider,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }

    result = check_model(model_dict)
    if result["ok"]:
        print(f"✓ 模型连通成功 [{result.get('via')}]")
        print(f"  模型: {model_dict['model_name_or_path']}")
        print(f"  回复: {str(result.get('reply'))[:120]}")
        return 0
    else:
        print(f"✗ 模型连通失败：{result['reason']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

