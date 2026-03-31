from __future__ import annotations

import os
import time
import traceback
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import re

import pandas as pd
from dataflow.operators.core_text import BenchAnswerGenerator, UnifiedBenchDatasetEvaluator
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, APILLMServing_request
from dataflow.core import LLMServingABC

from one_eval.core.state import BenchInfo, ModelConfig
from one_eval.logger import get_logger
import random

log = get_logger("DataFlowEvalTool")


class RobustAPILLMServing(APILLMServing_request):
    """
    A robust wrapper around DataFlow's APILLMServing_request that intercepts 
    and handles HTTP 429 Too Many Requests errors with exponential backoff and jitter,
    without modifying the underlying DataFlow library.
    """
    def _api_chat_with_id(self, id: int, payload, model: str, is_embedding: bool = False, json_schema: dict = None):
        start = time.time()
        # Call the original method
        try:
            # We need to temporarily mock the session's post method to catch 429s before the original method suppresses them
            original_post = self.session.post
            
            def custom_post(*args, **kwargs):
                resp = original_post(*args, **kwargs)
                if resp.status_code == 429:
                    class RateLimitException(Exception): pass
                    raise RateLimitException("429 Too Many Requests")
                return resp
                
            self.session.post = custom_post
            try:
                return super()._api_chat_with_id(id, payload, model, is_embedding, json_schema)
            finally:
                # Always restore the original method
                self.session.post = original_post
        except Exception as e:
            if e.__class__.__name__ == "RateLimitException":
                raise e
            # Re-raise for the retry loop to handle
            raise e

    def _api_chat_id_retry(self, id, payload, model, is_embedding: bool = False, json_schema: dict = None):
        for i in range(self.max_retries):
            try:
                result = self._api_chat_with_id(id, payload, model, is_embedding, json_schema)
                if result[1] is not None:
                    return result
                
                # If None is returned (non-429 error), use standard backoff
                sleep_time = (2 ** i) + random.uniform(0, 1)
                self.logger.info(f"Retrying API request (id={id}) after {sleep_time:.2f}s (Attempt {i+1}/{self.max_retries})")
                time.sleep(sleep_time)
            except Exception as e:
                if e.__class__.__name__ == "RateLimitException":
                    # Specific backoff for rate limits, longer and with more jitter
                    sleep_time = (2 ** i) * 1.5 + random.uniform(0, 2)
                    self.logger.warning(f"Rate limit hit. Retrying API request (id={id}) after {sleep_time:.2f}s (Attempt {i+1}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    sleep_time = (2 ** i) + random.uniform(0, 1)
                    self.logger.warning(f"Error hit. Retrying API request (id={id}) after {sleep_time:.2f}s (Attempt {i+1}/{self.max_retries})")
                    time.sleep(sleep_time)
                    
        self.logger.error(f"Failed to get response for id={id} after {self.max_retries} retries.")
        return id, None


class DataFlowEvalTool:
    """
    封装 DataFlow 的评测 Pipeline
    - BenchAnswerGenerator
    - UnifiedBenchDatasetEvaluator
    """
    
    # Class-level cache to prevent reloading vLLM on every request
    _cached_llm_serving: Optional[LLMServingABC] = None
    _cached_model_config: Optional[ModelConfig] = None

    def __init__(self, output_root: str = "cache/eval_results"):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)
        # Initialize instance members from cache if available, otherwise None
        self.llm_serving: Optional[LLMServingABC] = DataFlowEvalTool._cached_llm_serving
        self._current_model_config: Optional[ModelConfig] = DataFlowEvalTool._cached_model_config

    def _init_llm_serving(self, config: ModelConfig):
        """初始化或更新 LLM Serving"""
        def _is_broken_local(serving: Any) -> bool:
            if serving is None:
                return False
            if isinstance(serving, LocalModelLLMServing_vllm):
                if getattr(serving, "backend_initialized", False) and not hasattr(serving, "tokenizer"):
                    return True
            return False

        # Check global cache first
        if DataFlowEvalTool._cached_llm_serving and DataFlowEvalTool._cached_model_config == config:
            if _is_broken_local(DataFlowEvalTool._cached_llm_serving):
                log.warning("Detected broken cached local serving (missing tokenizer), rebuilding...")
                try:
                    if hasattr(DataFlowEvalTool._cached_llm_serving, "cleanup"):
                        DataFlowEvalTool._cached_llm_serving.cleanup()
                except Exception:
                    pass
                DataFlowEvalTool._cached_llm_serving = None
                DataFlowEvalTool._cached_model_config = None
            else:
                self.llm_serving = DataFlowEvalTool._cached_llm_serving
                self._current_model_config = config
                return

        # If cache exists but config differs, cleanup old one
        if DataFlowEvalTool._cached_llm_serving:
            try:
                log.info("Cleaning up old LLM serving instance...")
                if hasattr(DataFlowEvalTool._cached_llm_serving, "cleanup"):
                    DataFlowEvalTool._cached_llm_serving.cleanup()
            except Exception as e:
                log.warning(f"Failed to cleanup old serving: {e}")
            DataFlowEvalTool._cached_llm_serving = None
            DataFlowEvalTool._cached_model_config = None

        # 如果配置相同且 serving 已存在 (instance level check, just in case)
        if self.llm_serving and self._current_model_config == config:
            return

        model_name_or_path = config.model_name_or_path
        if isinstance(model_name_or_path, str) and model_name_or_path:
            p = model_name_or_path.strip()
            if os.name == "nt":
                m = re.match(r"^/mnt/([a-zA-Z])/(.+)$", p)
                if m:
                    drive = m.group(1).upper()
                    rest = m.group(2).replace("/", "\\")
                    p = f"{drive}:\\{rest}"
            else:
                m = re.match(r"^([a-zA-Z]):\\(.+)$", p)
                if m:
                    drive = m.group(1).lower()
                    rest = m.group(2).replace("\\", "/")
                    p = f"/mnt/{drive}/{rest}"
            model_name_or_path = p

        log.info(f"Initializing LLM Serving: {model_name_or_path} (is_api={config.is_api})")
        
        if config.is_api:
            # DataFlow's APILLMServing_request strictly reads API key from environment variables.
            # We temporarily set it here before initialization to avoid modifying DataFlow library.
            if config.api_key:
                os.environ["DF_API_KEY"] = config.api_key
                
            self.llm_serving = RobustAPILLMServing(
                api_url=config.api_url or "https://api.openai.com/v1/chat/completions",
                model_name=model_name_or_path,
                key_name_of_api_key="DF_API_KEY",
                max_workers=16, # 默认并发
                # API 模式下的 generation 参数通常在调用时传递，或者由 Serving 类处理
            )
        else:
            self.llm_serving = LocalModelLLMServing_vllm(
                hf_model_name_or_path=model_name_or_path,
                vllm_tensor_parallel_size=config.tensor_parallel_size,
                vllm_max_tokens=config.max_tokens,
                vllm_temperature=config.temperature,
                vllm_top_p=config.top_p,
                vllm_top_k=getattr(config, "top_k", -1),
                vllm_repetition_penalty=getattr(config, "repetition_penalty", 1.0),
                vllm_seed=getattr(config, "seed", None),
                vllm_max_model_len=getattr(config, "max_model_len", None),
                vllm_gpu_memory_utilization=getattr(config, "gpu_memory_utilization", 0.9),
                # trust_remote_code=True, # 默认信任，State 中已移除该配置
            )
            try:
                self.llm_serving.start_serving()
                if not hasattr(self.llm_serving, "tokenizer"):
                    raise RuntimeError("vLLM serving initialized without tokenizer")
            except Exception as e:
                try:
                    if hasattr(self.llm_serving, "backend_initialized"):
                        self.llm_serving.backend_initialized = False
                except Exception:
                    pass
                DataFlowEvalTool._cached_llm_serving = None
                DataFlowEvalTool._cached_model_config = None
                raise RuntimeError(f"Local vLLM serving init failed: {e}") from e
        
        self._current_model_config = config
        
        # Update class-level cache
        DataFlowEvalTool._cached_llm_serving = self.llm_serving
        DataFlowEvalTool._cached_model_config = config

    def _preprocess_dataframe(self, df, bench_name, key_mapping, cache_path="", eval_type=""):
        """Ad-hoc 数据预处理"""
        
        # 1. 自动合并 choices
        choices_key = key_mapping.get("input_choices_key")
        if isinstance(choices_key, list):
            # 检查这些列是否都在 df 中
            missing_cols = [c for c in choices_key if c not in df.columns]
            if not missing_cols:
                # 合并列
                df["merged_choices"] = df.apply(lambda row: [str(row[c]) for c in choices_key], axis=1)
                key_mapping["input_choices_key"] = "merged_choices"
                log.info(f"[{bench_name}] Auto-merged columns {choices_key} into 'merged_choices'")
            else:
                log.warning(f"[{bench_name}] Cannot merge choices, missing columns: {missing_cols}")

        # 2. 自动注入 choices (针对 key3_q_choices_a)
        if eval_type == "key3_q_choices_a":
            # 如果 input_choices_key 缺失，或者对应的列不存在
            current_choices_key = key_mapping.get("input_choices_key")
            if not current_choices_key or (isinstance(current_choices_key, str) and current_choices_key not in df.columns):
                # 尝试推断是否为 Bool/Binary 任务
                # 简单启发式：检查 label 列是否存在，且值域是否类似 0/1 或 False/True
                # 为了安全，我们只对明确缺失 choices 的情况注入 ["False", "True"]
                # 这是一个合理的默认值，即便对于 Yes/No 任务，通常也是映射到 False/True 的
                if "choices" not in df.columns:
                    df["choices"] = [["False", "True"]] * len(df)
                    key_mapping["input_choices_key"] = "choices"
                    log.info(f"[{bench_name}] Auto-injected default choices ['False', 'True'] for key3_q_choices_a")
        
        return df, key_mapping

    def _extract_path_value(self, obj: Any, path: str) -> Any:
        if not path or not isinstance(path, str):
            return None
        cur = obj
        for p in path.split("."):
            if isinstance(cur, dict):
                if p not in cur:
                    return None
                cur = cur[p]
                continue
            if isinstance(cur, list):
                if not p.isdigit():
                    return None
                idx = int(p)
                if idx < 0 or idx >= len(cur):
                    return None
                cur = cur[idx]
                continue
            return None
        return cur

    def _materialize_nested_keys(self, source_path: str, key_paths: List[str], target_path: str) -> str:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(source_path, "r", encoding="utf-8") as rf, open(target_path, "w", encoding="utf-8") as wf:
            for line in rf:
                s = line.strip()
                if not s:
                    continue
                row = json.loads(s)
                if isinstance(row, dict):
                    for kp in key_paths:
                        if kp and "." in kp and kp not in row:
                            row[kp] = self._extract_path_value(row, kp)
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        return target_path

    def _count_jsonl_rows(self, path: str) -> int:
        if not path or not os.path.exists(path):
            return 0
        cnt = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    cnt += 1
        return cnt

    def run_eval(self, bench: BenchInfo, model_config: ModelConfig, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        执行单个 Bench 的评测
        Returns:
            {
                "stats": dict,  # 评测统计结果
                "detail_path": str  # step2 结果文件路径
            }
        """
        if not bench.dataset_cache or not os.path.exists(bench.dataset_cache):
            raise FileNotFoundError(f"Bench {bench.bench_name} data not found at {bench.dataset_cache}")

        if not bench.bench_dataflow_eval_type:
            raise ValueError(f"Bench {bench.bench_name} missing bench_dataflow_eval_type")

        # 1. 准备 Serving
        self._init_llm_serving(model_config)

        # 2. 准备路径
        timestamp = int(time.time())
        safe_name = bench.bench_name.replace("/", "__")
        
        # 中间结果目录
        step_cache_dir = os.path.join(self.output_root, f"{safe_name}_{timestamp}_steps")
        os.makedirs(step_cache_dir, exist_ok=True)
        
        # 最终结果文件
        eval_result_path = os.path.join(self.output_root, f"{safe_name}_{timestamp}_result.jsonl")
        nested_stage_path = os.path.join(step_cache_dir, "step_input_nested.jsonl")

        def _emit(stage: str, generated: int = 0, total: int = 0, percent: float = 0.0):
            if progress_callback:
                progress_callback({
                    "bench_name": bench.bench_name,
                    "stage": stage,
                    "generated": int(generated),
                    "total": int(total),
                    "percent": float(percent),
                })

        # 3. 准备参数映射
        key_mapping = bench.meta.get("key_mapping", {})
        log.info(f"[{bench.bench_name}] Initial Key Mapping: {key_mapping}")

        all_key_paths = [v for v in key_mapping.values() if isinstance(v, str) and v.strip()]
        nested_paths = [p for p in all_key_paths if "." in p]
        input_dataset_path = bench.dataset_cache
        if nested_paths:
            try:
                input_dataset_path = self._materialize_nested_keys(bench.dataset_cache, nested_paths, nested_stage_path)
                log.info(f"[{bench.bench_name}] Materialized nested keys: {nested_paths}")
            except Exception as e:
                log.warning(f"[{bench.bench_name}] Materialize nested keys failed, fallback to raw dataset: {e}")
                input_dataset_path = bench.dataset_cache

        # 4. 初始化 Storage
        # cache_type="jsonl" 对应 .jsonl 文件
        storage = FileStorage(
            first_entry_file_name=input_dataset_path,
            cache_path=step_cache_dir,
            file_name_prefix="step",
            cache_type="jsonl",
        )
        
        # === Ad-hoc 预处理 ===
        # 读取初始数据，进行必要的列注入，然后写回
        try:
            # 直接读取原始文件，而不是通过 storage.read (因为它要求先 step)
            # 假设 dataset_cache 是 jsonl
            df = pd.read_json(input_dataset_path, lines=True)
            df, key_mapping = self._preprocess_dataframe(
                df, 
                bench.bench_name, 
                key_mapping, 
                cache_path=input_dataset_path,
                eval_type=bench.bench_dataflow_eval_type
            )
            # 写回作为 step_0 (这将推进 storage 的 step 计数)
            storage.write(df)
        except Exception as e:
            log.error(f"[{bench.bench_name}] 预处理失败: {e}")
            log.error(traceback.format_exc())
            # 如果预处理失败，我们继续尝试，也许不需要预处理也能跑
        
        # 提取关键字段名
        q_key = key_mapping.get("input_question_key")
        ctx_key = key_mapping.get("input_context_key")
        
        # Target keys 处理
        target_key = key_mapping.get("input_target_key")
        targets_key = key_mapping.get("input_targets_key")
        choices_key = key_mapping.get("input_choices_key")
        
        # 强制 choices_key 为 string（如果它是 list）
        if isinstance(choices_key, list):
            # 如果预处理中的合并失败了（比如列不存在），我们只能取第一个作为最后的挣扎，或者直接报错
            # 这里选择保留之前的防御逻辑，但加上警告，表明这是不正常的状态
            log.warning(f"[{bench.bench_name}] input_choices_key is still list {choices_key} after preprocessing. Using first element.")
            choices_key = choices_key[0]

        label_key = key_mapping.get("input_label_key")
        labels_key = key_mapping.get("input_labels_key")
        better_key = key_mapping.get("input_better_key")
        rejected_key = key_mapping.get("input_rejected_key")
        text_key = key_mapping.get("input_text_key")

        # 5. Step 1: Generator
        # 对于不需要生成的任务（如 text_score, choices_a_ll），Generator 可能只是透传或计算
        # BenchAnswerGenerator 内部会根据 eval_type 判断是否需要 generate
        
        # 构造 Prompt Template (简单通用版)
        # 注意：对于 chat 模型，通常建议使用 apply_chat_template，这里简化为 FormatStrPrompt
        # 如果是 base 模型，这个 template 很重要
        prompt_tmpl = FormatStrPrompt(f_str_template="{{question}}\nAnswer:")
        
        generator = BenchAnswerGenerator(
            llm_serving=self.llm_serving,
            eval_type=bench.bench_dataflow_eval_type,
            prompt_template=prompt_tmpl,
            allow_overwrite=False,
            force_generate=False, # 让算子自己决定
        )

        log.info(f"[{bench.bench_name}] Running Step 1: Generator ({bench.bench_dataflow_eval_type})")
        total_rows = self._count_jsonl_rows(input_dataset_path)
        _emit("generator", generated=0, total=total_rows, percent=0.0)
        step1_output_path = os.path.join(step_cache_dir, "step_step1.jsonl")
        try:
            step1_result: Dict[str, Any] = {"err": None}
            def _run_step1():
                try:
                    generator.run(
                        storage=storage.step(),
                        input_question_key=q_key,
                        input_context_key=ctx_key,
                        input_text_key=text_key,
                        input_choices_key=choices_key,
                        output_key="generated_ans",
                    )
                except Exception as ex:
                    step1_result["err"] = ex
            th = threading.Thread(target=_run_step1, daemon=True)
            th.start()
            last_generated = -1
            while th.is_alive():
                generated = self._count_jsonl_rows(step1_output_path)
                if generated != last_generated:
                    pct = (float(generated) / float(total_rows) * 100.0) if total_rows > 0 else 0.0
                    if pct > 99.0:
                        pct = 99.0
                    _emit("generator", generated=generated, total=total_rows, percent=pct)
                    last_generated = generated
                time.sleep(0.5)
            th.join()
            if step1_result["err"] is not None:
                raise step1_result["err"]
            generated_done = self._count_jsonl_rows(step1_output_path)
            final_pct = 100.0 if total_rows > 0 else 0.0
            _emit("generator", generated=generated_done, total=total_rows, percent=final_pct)
        except Exception as e:
            log.error(f"[{bench.bench_name}] Generator failed: {e}")
            log.error(traceback.format_exc())
            # 强制重置 serving，防止脏状态
            self.llm_serving = None
            raise e

        # 6. Step 2: Evaluator
        evaluator = UnifiedBenchDatasetEvaluator(
            eval_result_path=eval_result_path, # 这里的 path 其实是统计结果落盘 path？
            # UnifiedBenchDatasetEvaluator 的 eval_result_path 是存 stats json 的
            # 但是它也会把 per-sample 结果写回 dataframe (storage)
            llm_serving=self.llm_serving,
            eval_type=bench.bench_dataflow_eval_type,
            prompt_template=None,
            use_semantic_judge=False, # 暂不启用 semantic judge，除非显式指定
            metric_type=None,
        )

        log.info(f"[{bench.bench_name}] Running Step 2: Evaluator")
        _emit("evaluator", generated=total_rows, total=total_rows, percent=100.0)
        
        # 收集所有可能的 input keys
        eval_kwargs = {
            "storage": storage.step(),
            "input_question_key": q_key,
            "input_context_key": ctx_key,
            "input_pred_key": "generated_ans",
            "input_text_key": text_key,
            "input_target_key": target_key,
            "input_targets_key": targets_key,
            "input_choices_key": choices_key,
            "input_label_key": label_key,
            "input_labels_key": labels_key,
            "input_better_key": better_key,
            "input_rejected_key": rejected_key,
        }
        # 过滤 None 和 空字符串
        eval_kwargs = {k: v for k, v in eval_kwargs.items() if v}
        
        try:
            evaluator.run(**eval_kwargs)
        except Exception as e:
            log.error(f"[{bench.bench_name}] Evaluator failed: {e}")
            log.error(traceback.format_exc())
            # Evaluator 失败通常不涉及 serving 状态，但为了保险起见
            self.llm_serving = None
            raise e

        # 7. 获取结果
        # step2 产生的文件是包含完整数据的
        # storage.step() 调用了两次，现在 index 是 2 (0->1->2)
        # 实际上 evaluator 跑完后，结果在 storage 当前指向的文件里
        # FileStorage 的 step() 会移动指针，所以我们需要获取“上一步”的文件名，或者当前最新的文件
        # FileStorage 没有直接暴露 current file path，但我们可以推断
        # file_name_prefix="step" -> step_0.jsonl (input), step_1.jsonl (gen output), step_2.jsonl (eval output)
        
        # 简单起见，我们列出 step_cache_dir 下最新的 jsonl
        files = sorted([f for f in os.listdir(step_cache_dir) if f.endswith(".jsonl") and f.startswith("step_")])
        if not files:
            raise RuntimeError("No step files generated")
        last_step_file = os.path.join(step_cache_dir, files[-1])

        # 读取统计结果
        # Evaluator 会把 stats 写入 eval_result_path (这是一个 json 文件，不是 jsonl)
        # 注意 UnifiedBenchDatasetEvaluator 代码里：df.to_json(..., orient="records")
        stats = {}
        if os.path.exists(eval_result_path):
            try:
                stats_df = pd.read_json(eval_result_path)
                if not stats_df.empty:
                    stats = stats_df.iloc[0].to_dict()
            except Exception as e:
                log.error(f"Failed to read stats from {eval_result_path}: {e}")

        return {
            "stats": stats,
            "detail_path": str(Path(last_step_file).absolute()),
            "key_mapping": key_mapping,
        }
