from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import re

import pandas as pd
from dataflow.operators.core_text import BenchAnswerGenerator, UnifiedBenchDatasetEvaluator
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, APILLMServing_request
from dataflow.core import LLMServingABC

from one_eval.core.state import BenchInfo, ModelConfig
from one_eval.logger import get_logger

log = get_logger("DataFlowEvalTool")


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
        # Check global cache first
        if DataFlowEvalTool._cached_llm_serving and DataFlowEvalTool._cached_model_config == config:
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
                m = re.match(r"^([a-zA-Z]):\\\\(.+)$", p)
                if m:
                    drive = m.group(1).lower()
                    rest = m.group(2).replace("\\", "/")
                    p = f"/mnt/{drive}/{rest}"
            model_name_or_path = p

        log.info(f"Initializing LLM Serving: {model_name_or_path} (is_api={config.is_api})")
        
        if config.is_api:
            self.llm_serving = APILLMServing_request(
                api_url=config.api_url,
                model_name=model_name_or_path,
                api_key=config.api_key,
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

    def run_eval(self, bench: BenchInfo, model_config: ModelConfig) -> Dict[str, Any]:
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

        # 3. 初始化 Storage
        # cache_type="jsonl" 对应 .jsonl 文件
        storage = FileStorage(
            first_entry_file_name=bench.dataset_cache,
            cache_path=step_cache_dir,
            file_name_prefix="step",
            cache_type="jsonl",
        )

        # 4. 准备参数映射
        key_mapping = bench.meta.get("key_mapping", {})
        log.info(f"[{bench.bench_name}] Initial Key Mapping: {key_mapping}")
        
        # === Ad-hoc 预处理 ===
        # 读取初始数据，进行必要的列注入，然后写回
        try:
            # 直接读取原始文件，而不是通过 storage.read (因为它要求先 step)
            # 假设 dataset_cache 是 jsonl
            df = pd.read_json(bench.dataset_cache, lines=True)
            df, key_mapping = self._preprocess_dataframe(
                df, 
                bench.bench_name, 
                key_mapping, 
                cache_path=bench.dataset_cache,
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
        try:
            generator.run(
                storage=storage.step(),
                input_question_key=q_key,
                input_context_key=ctx_key,
                input_text_key=text_key, # text_score 用
                input_choices_key=choices_key, # 有些生成任务可能需要选项进 prompt
                output_key="generated_ans",
            )
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
