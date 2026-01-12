from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

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

    def __init__(self, output_root: str = "cache/eval_results"):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)
        self.llm_serving: Optional[LLMServingABC] = None
        self._current_model_config: Optional[ModelConfig] = None

    def _init_llm_serving(self, config: ModelConfig):
        """初始化或更新 LLM Serving"""
        # 如果配置相同且 serving 已存在，复用
        if self.llm_serving and self._current_model_config == config:
            return

        log.info(f"Initializing LLM Serving: {config.model_name_or_path} (is_api={config.is_api})")
        
        if config.is_api:
            self.llm_serving = APILLMServing_request(
                api_url=config.api_url,
                model_name=config.model_name_or_path,
                api_key=config.api_key,
                max_workers=16, # 默认并发
                # API 模式下的 generation 参数通常在调用时传递，或者由 Serving 类处理
            )
        else:
            self.llm_serving = LocalModelLLMServing_vllm(
                hf_model_name_or_path=config.model_name_or_path,
                vllm_tensor_parallel_size=config.tensor_parallel_size,
                vllm_max_tokens=config.max_tokens,
                vllm_temperature=config.temperature,
                vllm_top_p=config.top_p,
                vllm_max_model_len=getattr(config, "max_model_len", None),
                # trust_remote_code=True, # 默认信任，State 中已移除该配置
            )
        
        self._current_model_config = config

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
        
        # 提取关键字段名
        q_key = key_mapping.get("input_question_key")
        ctx_key = key_mapping.get("input_context_key")
        
        # Target keys 处理
        target_key = key_mapping.get("input_target_key")
        targets_key = key_mapping.get("input_targets_key")
        choices_key = key_mapping.get("input_choices_key")
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
        generator.run(
            storage=storage.step(),
            input_question_key=q_key,
            input_context_key=ctx_key,
            input_text_key=text_key, # text_score 用
            input_choices_key=choices_key, # 有些生成任务可能需要选项进 prompt
            output_key="generated_ans",
        )

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
        # 过滤 None
        eval_kwargs = {k: v for k, v in eval_kwargs.items() if v is not None}
        
        evaluator.run(**eval_kwargs)

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
                import pandas as pd
                stats_df = pd.read_json(eval_result_path)
                if not stats_df.empty:
                    stats = stats_df.iloc[0].to_dict()
            except Exception as e:
                log.error(f"Failed to read stats from {eval_result_path}: {e}")

        return {
            "stats": stats,
            "detail_path": str(Path(last_step_file).absolute())
        }
