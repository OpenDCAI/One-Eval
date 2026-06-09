# Model Setup — 模型接入、凭证与 HF 下载配置

评测前**必须**先用 `scripts/check_model.py` 测连通（数据量大、跑错浪费时间）。
连通失败不要进入评测，先按 stderr 的可读原因排查。

## API 模型（is_api: true）

支持 OpenAI 兼容端点：`openai_compatible` / `deepseek`。

```bash
python scripts/check_model.py --api \
    --model gpt-4o-mini \
    --api-url http://HOST:3000/v1/chat/completions \
    --api-key sk-xxxx \
    --api-provider openai_compatible
```

evalspec.yaml 对应字段：
```yaml
model:
  model_name_or_path: "gpt-4o-mini"
  is_api: true
  api_url: "http://HOST:3000/v1/chat/completions"
  api_key: "sk-xxxx"            # 或留空，改用环境变量 DF_API_KEY
  api_provider: "openai_compatible"
  api_max_workers: 8            # 并发
  api_connect_timeout: 10.0
  api_read_timeout: 60.0
```

URL 规范化规则（`check_model.py` 自动处理）：
- 以 `/chat/completions` 结尾 → 原样用
- 以 `/v1` 结尾 → 自动补 `/chat/completions`
- 空 → 按 provider 用默认（deepseek / openai 官方端点）

常见失败与含义（stderr 直接给中文）：
- 401 → api_key 无效/缺失
- 404 → 端点或模型名不存在（核对 api_url 与 model）
- 429 → 限流，稍后重试
- 连接超时/不可达 → 网络/URL/防火墙

凭证安全：api_key 只写进本地 `evalspec.yaml`（已 gitignore），**不要回显到对话或入库**。

## 本地 vLLM 模型（is_api: false）

需 GPU + vllm 环境。**本机 Mac 不验证此路径**，代码完整但交由 GPU 机测试。

```bash
python scripts/check_model.py \
    --model /path/to/model \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

evalspec.yaml 对应字段：
```yaml
model:
  model_name_or_path: "/path/to/model"
  is_api: false
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 4096
```

vLLM 探测会真正起一次 serving 并推理一条极简 prompt；起服务失败/推理空都会报可读原因。

## 采样参数（两类模型通用）

```yaml
  temperature: 0.0     # 评测建议确定性
  top_p: 1.0
  max_tokens: 2048
  seed: 42
```

## HuggingFace 下载配置（接入新 bench 时）

`prepare_bench.py` / `run_eval.py` 下载数据走 HFDownloadTool。若下载失败，按提示配置：

- **gated/private 数据集或需要 token**：
  ```bash
  export HF_TOKEN=<你的 token>   # https://huggingface.co/settings/tokens 申请
  ```
- **国内网络访问不到 huggingface.co**：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- **仓库/配置 404**：核对 repo_id；很多数据集需指定 `--config <子集>` 和 `--split`。

这些是**失败时的引导**，不在脚本里硬编码。配好环境变量后重试即可。

## 评测引擎依赖

评测内核依赖 `open-dataflow`（pypi 名，import 名 `dataflow`）+ `datasets` 等。
裸环境缺这些包时 `import one_eval` 会失败 —— 需要 Python ≥3.10 的环境装好依赖。
`check_model.py` 的 API 探测是**纯 requests 实现**，不依赖 dataflow，所以未装引擎也能验 API 连通。
