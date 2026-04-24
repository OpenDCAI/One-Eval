from __future__ import annotations

import os
import time
import traceback
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import re
from urllib.parse import urlparse

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

    @classmethod
    def release_serving(cls):
        """释放 vLLM 显存，评测完成后调用"""
        if cls._cached_llm_serving is not None:
            try:
                if hasattr(cls._cached_llm_serving, "cleanup"):
                    cls._cached_llm_serving.cleanup()
            except Exception:
                pass
            cls._cached_llm_serving = None
            cls._cached_model_config = None

    def __init__(self, output_root: str = "cache/eval_results"):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)
        # Initialize instance members from cache if available, otherwise None
        self.llm_serving: Optional[LLMServingABC] = DataFlowEvalTool._cached_llm_serving
        self._current_model_config: Optional[ModelConfig] = DataFlowEvalTool._cached_model_config
        self.judge_serving: Optional[LLMServingABC] = None

    @staticmethod
    def _normalize_api_url(api_url: Optional[str], provider: str) -> str:
        provider_name = str(provider or "openai_compatible").strip().lower()
        raw = (api_url or "").strip()
        if not raw:
            if provider_name == "deepseek":
                return "https://api.deepseek.com/chat/completions"
            return "https://api.openai.com/v1/chat/completions"

        lowered = raw.lower().rstrip("/")
        if lowered.endswith("/chat/completions"):
            return raw.rstrip("/")
        if lowered.endswith("/v1"):
            return f"{raw.rstrip('/')}/chat/completions"

        parsed = urlparse(raw)
        if provider_name == "deepseek" and parsed.netloc.lower() == "api.deepseek.com":
            return f"{raw.rstrip('/')}/chat/completions"
        return raw

    @staticmethod
    def _build_api_serving_kwargs(config: ModelConfig) -> Dict[str, Any]:
        provider_name = str(getattr(config, "api_provider", "openai_compatible") or "openai_compatible").strip().lower()
        payload_kwargs: Dict[str, Any] = {}

        if config.temperature is not None:
            payload_kwargs["temperature"] = float(config.temperature)
        if config.top_p is not None:
            payload_kwargs["top_p"] = float(config.top_p)
        if config.max_tokens is not None:
            payload_kwargs["max_tokens"] = int(config.max_tokens)
        if getattr(config, "seed", None) is not None:
            payload_kwargs["seed"] = int(config.seed)

        if getattr(config, "top_k", -1) not in (None, -1):
            log.info(
                "Skip top_k for API target model '%s' (provider=%s); keep UI non-blocking for unsupported API params.",
                config.model_name_or_path,
                provider_name,
            )
        if getattr(config, "repetition_penalty", None) not in (None, 1.0):
            log.info(
                "Skip repetition_penalty for API target model '%s' (provider=%s); keep UI non-blocking for unsupported API params.",
                config.model_name_or_path,
                provider_name,
            )

        extra_body = getattr(config, "api_extra_body", None) or {}
        if not isinstance(extra_body, dict):
            raise ValueError("api_extra_body must be a dict")

        serving_kwargs = {
            "api_url": DataFlowEvalTool._normalize_api_url(getattr(config, "api_url", None), provider_name),
            "model_name": config.model_name_or_path,
            "key_name_of_api_key": "DF_API_KEY",
            "max_workers": max(1, int(getattr(config, "api_max_workers", 16) or 16)),
            "connect_timeout": float(getattr(config, "api_connect_timeout", 10.0) or 10.0),
            "read_timeout": float(getattr(config, "api_read_timeout", 120.0) or 120.0),
        }
        payload_kwargs.update(extra_body)
        serving_kwargs.update(payload_kwargs)
        return serving_kwargs

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

            api_serving_kwargs = self._build_api_serving_kwargs(config)
            api_serving_kwargs["model_name"] = model_name_or_path
            self.llm_serving = RobustAPILLMServing(**api_serving_kwargs)
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
        elif isinstance(choices_key, str) and choices_key in df.columns:
            df["normalized_choices"] = df[choices_key].apply(self._normalize_choices_value)
            if df["normalized_choices"].map(len).gt(0).any():
                key_mapping["input_choices_key"] = "normalized_choices"
                log.info(f"[{bench_name}] Normalized choices column '{choices_key}' into 'normalized_choices'")

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

            current_label_key = key_mapping.get("input_label_key")
            label_candidates = ["answer_idx", "answer_index", "answerKey", "label", "target"]
            if (not current_label_key or current_label_key not in df.columns) and any(c in df.columns for c in label_candidates):
                for cand in label_candidates:
                    if cand in df.columns:
                        key_mapping["input_label_key"] = cand
                        log.info(f"[{bench_name}] Auto-selected label column '{cand}' for key3_q_choices_a")
                        break
        
        return df, key_mapping

    @staticmethod
    def _is_empty_value(value: Any) -> bool:
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, (list, tuple, dict)) and len(value) == 0:
            return True
        return False

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    @staticmethod
    def _sort_choice_items(items: List[Any]) -> List[Any]:
        def _key(item: Any):
            raw = str(item[0]).strip()
            if len(raw) == 1 and raw.isalpha():
                return (0, ord(raw.upper()))
            if raw.isdigit():
                return (1, int(raw))
            m = re.match(r"^[A-Za-z]+[_-]?(\d+)$", raw)
            if m:
                return (2, int(m.group(1)))
            return (3, raw)
        return sorted(items, key=_key)

    def _normalize_choices_value(self, value: Any) -> List[str]:
        if self._is_empty_value(value):
            return []
        if isinstance(value, (list, tuple)):
            return [self._stringify_value(v).strip() for v in value if not self._is_empty_value(v)]
        if isinstance(value, dict):
            ordered = self._sort_choice_items(list(value.items()))
            return [self._stringify_value(v).strip() for _, v in ordered if not self._is_empty_value(v)]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("{") and raw.endswith("}")):
                try:
                    parsed = json.loads(raw)
                    return self._normalize_choices_value(parsed)
                except Exception:
                    pass
            for sep in ("||", "|", ";"):
                if sep in raw:
                    parts = [p.strip() for p in raw.split(sep)]
                    parts = [p for p in parts if p]
                    if len(parts) > 1:
                        return parts
        return [self._stringify_value(value).strip()]

    def _normalize_target_list(self, value: Any) -> List[str]:
        if self._is_empty_value(value):
            return []
        if isinstance(value, (list, tuple)):
            return [self._stringify_value(v).strip() for v in value if not self._is_empty_value(v)]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("{") and raw.endswith("}")):
                try:
                    parsed = json.loads(raw)
                    return self._normalize_target_list(parsed)
                except Exception:
                    pass
            for sep in ("||", "|", ";"):
                if sep in raw:
                    parts = [p.strip() for p in raw.split(sep)]
                    parts = [p for p in parts if p]
                    if len(parts) > 1:
                        return parts
        return [self._stringify_value(value).strip()]

    def _normalize_label_to_index(self, label: Any, choices: List[str]) -> Optional[int]:
        if self._is_empty_value(label):
            return None
        n = len(choices)
        if isinstance(label, int):
            if 0 <= int(label) < n:
                return int(label)
            if 1 <= int(label) <= n:
                return int(label) - 1
            return None
        raw = str(label).strip()
        if not raw:
            return None
        if len(raw) == 1 and raw.isalpha():
            idx = ord(raw.upper()) - ord("A")
            return idx if 0 <= idx < n else None
        if raw.isdigit():
            val = int(raw)
            if 0 <= val < n:
                return val
            if 1 <= val <= n:
                return val - 1
        normalized_choices = [self._stringify_value(c).strip().casefold() for c in choices]
        try:
            return normalized_choices.index(raw.casefold())
        except ValueError:
            return None

    def _normalize_multilabel_to_indices(self, labels: Any, choices: List[str]) -> List[int]:
        values = self._normalize_target_list(labels)
        out: List[int] = []
        for item in values:
            idx = self._normalize_label_to_index(item, choices)
            if idx is not None and idx not in out:
                out.append(idx)
        return out

    def _build_serving_instance(self, config: ModelConfig) -> LLMServingABC:
        model_name_or_path = config.model_name_or_path
        if config.is_api:
            if config.api_key:
                os.environ["DF_API_KEY"] = config.api_key
            api_serving_kwargs = self._build_api_serving_kwargs(config)
            api_serving_kwargs["model_name"] = model_name_or_path
            return RobustAPILLMServing(**api_serving_kwargs)
        serving = LocalModelLLMServing_vllm(
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
        )
        serving.start_serving()
        return serving

    def _format_judge_prompt(self, judge_config: Dict[str, Any], payload: Dict[str, str]) -> str:
        prompt_template = str(judge_config.get("prompt_template") or "").strip()
        if prompt_template:
            class _SafeDict(dict):
                def __missing__(self, key):
                    return ""
            try:
                return prompt_template.format_map(_SafeDict(payload)).strip()
            except Exception as e:
                log.warning("Custom judge prompt_template format failed, fallback to default: %s", e)
        sections = [
            f"Evaluation Type:\n{payload.get('eval_type', '')}",
            f"Question:\n{payload.get('question', '')}",
            f"Context:\n{payload.get('context', '')}",
            f"Choices:\n{payload.get('choices', '')}",
            f"Prediction:\n{payload.get('prediction', '')}",
            f"Reference Answer:\n{payload.get('reference_answer', '')}",
            f"Reference Answers:\n{payload.get('reference_answers', '')}",
            f"Correct Choice:\n{payload.get('correct_answer', '')}",
            f"Correct Choices:\n{payload.get('correct_answers', '')}",
            f"Preferred Answer:\n{payload.get('better_answer', '')}",
            f"Rejected Answer:\n{payload.get('rejected_answer', '')}",
            f"Rule:\n{payload.get('rule', '')}",
        ]
        body = "\n\n".join([s for s in sections if not s.endswith("\n")])
        return (
            "Judge whether the model prediction should be considered correct for this sample.\n"
            "Use every provided field that is relevant, especially the rule if present.\n"
            "Return JSON only: {\"judgement_result\": true} or {\"judgement_result\": false}.\n\n"
            f"{body}"
        ).strip()

    def _resolve_judge_response(self, response: Any) -> Optional[bool]:
        if isinstance(response, bool):
            return response
        if isinstance(response, dict):
            for key in ("judgement_result", "judgment_result", "correct", "result"):
                if key in response:
                    value = response[key]
                    if isinstance(value, bool):
                        return value
                    if isinstance(value, str):
                        lowered = value.strip().lower()
                        if lowered in ("true", "yes", "correct", "1"):
                            return True
                        if lowered in ("false", "no", "incorrect", "0"):
                            return False
        text = self._stringify_value(response).strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return self._resolve_judge_response(parsed)
        except Exception:
            pass
        lowered = text.lower()
        if '"judgement_result": true' in lowered or '"judgment_result": true' in lowered:
            return True
        if '"judgement_result": false' in lowered or '"judgment_result": false' in lowered:
            return False
        if re.search(r"\btrue\b", lowered):
            return True
        if re.search(r"\bfalse\b", lowered):
            return False
        return None

    def _run_llm_judge(
        self,
        *,
        bench: BenchInfo,
        judge_config: Dict[str, Any],
        judge_model_config: ModelConfig,
        step1_output_path: str,
        step2_output_path: str,
        eval_result_path: str,
        key_mapping: Dict[str, Any],
        eval_type: str,
    ) -> Dict[str, Any]:
        judge_serving = self._build_serving_instance(judge_model_config)
        judge_cleanup_needed = hasattr(judge_serving, "cleanup")
        system_prompt = str(judge_config.get("system_prompt") or "").strip() or (
            "You are a strict answer judge. Consider the question, context, references, choices and any provided rule. "
            "Return JSON only with a boolean field named judgement_result."
        )
        try:
            df = pd.read_json(step1_output_path, lines=True)
            prompts: List[str] = []
            row_indices: List[int] = []
            raw_payloads: Dict[int, Dict[str, str]] = {}
            q_key = key_mapping.get("input_question_key")
            ctx_key = key_mapping.get("input_context_key")
            text_key = key_mapping.get("input_text_key")
            target_key = key_mapping.get("input_target_key")
            targets_key = key_mapping.get("input_targets_key")
            choices_key = key_mapping.get("input_choices_key")
            label_key = key_mapping.get("input_label_key")
            labels_key = key_mapping.get("input_labels_key")
            better_key = key_mapping.get("input_better_key")
            rejected_key = key_mapping.get("input_rejected_key")
            rule_key = str(judge_config.get("rule_key") or "").strip()
            pred_key = "generated_ans"

            df["eval_valid"] = False
            df["eval_error"] = ""
            df["eval_score"] = None
            df["eval_pred"] = None
            df["judge_response"] = ""

            for idx, row in df.iterrows():
                choices_list = self._normalize_choices_value(row.get(choices_key)) if choices_key else []
                single_targets = self._normalize_target_list(row.get(target_key)) if target_key else []
                multi_targets = self._normalize_target_list(row.get(targets_key)) if targets_key else []
                better_targets = self._normalize_target_list(row.get(better_key)) if better_key else []
                rejected_targets = self._normalize_target_list(row.get(rejected_key)) if rejected_key else []
                prediction = row.get(pred_key)
                if self._is_empty_value(prediction) and text_key:
                    prediction = row.get(text_key)
                if self._is_empty_value(prediction):
                    df.at[idx, "eval_error"] = "empty_prediction"
                    continue

                correct_answer = ""
                correct_answers = ""
                if label_key and choices_list:
                    choice_idx = self._normalize_label_to_index(row.get(label_key), choices_list)
                    if choice_idx is not None:
                        correct_answer = choices_list[choice_idx]
                if labels_key and choices_list:
                    choice_indices = self._normalize_multilabel_to_indices(row.get(labels_key), choices_list)
                    correct_answers = "\n".join([choices_list[i] for i in choice_indices if 0 <= i < len(choices_list)])

                payload = {
                    "eval_type": eval_type,
                    "question": self._stringify_value(row.get(q_key)).strip() if q_key else "",
                    "context": self._stringify_value(row.get(ctx_key)).strip() if ctx_key else "",
                    "choices": "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices_list)]),
                    "prediction": self._stringify_value(prediction).strip(),
                    "reference_answer": "\n".join(single_targets).strip(),
                    "reference_answers": "\n".join(multi_targets).strip(),
                    "correct_answer": correct_answer,
                    "correct_answers": correct_answers,
                    "better_answer": "\n".join(better_targets).strip(),
                    "rejected_answer": "\n".join(rejected_targets).strip(),
                    "rule": self._stringify_value(row.get(rule_key)).strip() if rule_key else "",
                }
                prompts.append(self._format_judge_prompt(judge_config, payload))
                row_indices.append(idx)
                raw_payloads[idx] = payload

            if prompts:
                responses = judge_serving.generate_from_input(user_inputs=prompts, system_prompt=system_prompt)
                for idx, resp in zip(row_indices, responses):
                    ok = self._resolve_judge_response(resp)
                    df.at[idx, "judge_response"] = self._stringify_value(resp)
                    if ok is None:
                        df.at[idx, "eval_valid"] = False
                        df.at[idx, "eval_error"] = "judge_parse_failed"
                    else:
                        df.at[idx, "eval_valid"] = True
                        df.at[idx, "eval_error"] = ""
                        df.at[idx, "eval_score"] = 1.0 if ok else 0.0
                        df.at[idx, "eval_pred"] = 1 if ok else 0
            df.to_json(step2_output_path, orient="records", lines=True, force_ascii=False)
            total_samples = int(len(df))
            valid_samples = int((df["eval_valid"] == True).sum())
            score_series = pd.to_numeric(df.loc[df["eval_valid"] == True, "eval_score"], errors="coerce")
            accuracy = float(score_series.mean()) if valid_samples > 0 and not score_series.empty else 0.0
            stats = {
                "total_samples": total_samples,
                "valid_samples": valid_samples,
                "accuracy": accuracy,
                "score": accuracy,
                "bench_name_or_prefix": "step",
                "type": eval_type,
                "metric": "llm_as_judge",
                "judge_model": judge_model_config.model_name_or_path,
            }
            Path(eval_result_path).write_text(json.dumps([stats], ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "stats": stats,
                "detail_path": str(Path(step2_output_path).absolute()),
                "key_mapping": key_mapping,
            }
        finally:
            if judge_cleanup_needed:
                try:
                    judge_serving.cleanup()
                except Exception:
                    pass

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

    def run_eval(
        self,
        bench: BenchInfo,
        model_config: ModelConfig,
        judge_model_config: Optional[ModelConfig] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
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

        judge_config = bench.meta.get("judge_config", {}) if isinstance(bench.meta, dict) else {}
        use_llm_as_judge = bool(judge_config.get("enabled") or judge_config.get("use_llm_as_judge"))

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
            force_generate=use_llm_as_judge, # judge 依赖 generated_ans，必须先生成
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

        step2_output_path = os.path.join(step_cache_dir, "step_step2.jsonl")

        if use_llm_as_judge:
            if judge_model_config is None:
                raise RuntimeError(f"[{bench.bench_name}] llm as judge is enabled but no judge model is configured")
            log.info(f"[{bench.bench_name}] Running Step 2: One-Eval LLM Judge")
            _emit("judge", generated=total_rows, total=total_rows, percent=100.0)
            return self._run_llm_judge(
                bench=bench,
                judge_config=judge_config if isinstance(judge_config, dict) else {},
                judge_model_config=judge_model_config,
                step1_output_path=step1_output_path,
                step2_output_path=step2_output_path,
                eval_result_path=eval_result_path,
                key_mapping=key_mapping,
                eval_type=bench.bench_dataflow_eval_type,
            )

        metric_type = None
        if bool(getattr(model_config, "is_api", False)) and bench.bench_dataflow_eval_type == "key3_q_choices_a":
            # API serving does not expose loglikelihood hooks; force parse-based fallback instead of ll_choice_acc.
            metric_type = "parse_choice_acc"
            log.info(
                "[%s] API target model detected for key3_q_choices_a; use parse-based choice evaluation instead of ll_choice_acc.",
                bench.bench_name,
            )

        # 6. Step 2: Evaluator
        evaluator = UnifiedBenchDatasetEvaluator(
            eval_result_path=eval_result_path, # 这里的 path 其实是统计结果落盘 path？
            # UnifiedBenchDatasetEvaluator 的 eval_result_path 是存 stats json 的
            # 但是它也会把 per-sample 结果写回 dataframe (storage)
            llm_serving=self.llm_serving,
            eval_type=bench.bench_dataflow_eval_type,
            prompt_template=None,
            use_semantic_judge=False,
            metric_type=metric_type,
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
