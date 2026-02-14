from __future__ import annotations

import os
from pathlib import Path

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState, ModelConfig
from one_eval.toolkits.dataflow_eval_tool import DataFlowEvalTool
from one_eval.logger import get_logger
from langgraph.types import Command

log = get_logger("DataFlowEvalNode")


class DataFlowEvalNode(BaseNode):
    """
    Step4: DataFlowEvalNode
    - 遍历 benches
    - 检查 eval_status
    - 准备 ModelConfig
    - 调用 DataFlowEvalTool
    - 更新状态
    """

    def __init__(self):
        self.name = "DataFlowEvalNode"
        self.logger = log
        # 默认输出目录
        self.output_root = os.path.join(os.getcwd(), "cache", "eval_results")
        self._tool = DataFlowEvalTool(output_root=self.output_root)

    async def run(self, state: NodeState) -> NodeState:
        state.current_node = self.name

        benches = getattr(state, "benches", None)
        if not benches:
            self.logger.warning("[DataFlowEvalNode] state.benches 为空")
            return state

        # 1. 准备模型配置
        # 优先使用 state.target_model，如果没有则尝试从 target_model 推断
        model_config = state.target_model
        
        if not model_config:
            # 简单的自动配置兜底
            if state.target_model:
                self.logger.info(f"Using default config for model: {state.target_model}")
                model_config = ModelConfig(
                    model_name_or_path=state.target_model,
                    is_api=False, # 默认为本地模型
                    tensor_parallel_size=1,
                    max_tokens=2048
                )
            else:
                self.logger.error("No target_model or target_model found!")
                return state

        tool = self._tool
        cursor = int(getattr(state, "eval_cursor", 0) or 0)
        if cursor < 0:
            cursor = 0
        if cursor >= len(benches):
            return state

        bench = benches[cursor]

        if bench.eval_status == "success" and bench.meta and bench.meta.get("eval_result"):
            self.logger.info(f"[{bench.bench_name}] 已评测成功，跳过")
            state.eval_cursor = cursor + 1
            return state

        if not bench.dataset_cache:
            self.logger.warning(f"[{bench.bench_name}] 缺少 dataset_cache，跳过")
            bench.eval_status = "failed"
            state.eval_cursor = cursor + 1
            return state

        if not bench.bench_dataflow_eval_type:
            self.logger.warning(f"[{bench.bench_name}] 缺少 eval_type，跳过")
            bench.eval_status = "failed"
            state.eval_cursor = cursor + 1
            return state

        try:
            self.logger.info(f"[{bench.bench_name}] 开始评测... Type={bench.bench_dataflow_eval_type}")
            bench.eval_status = "running"

            result = tool.run_eval(bench, model_config)

            if not bench.meta:
                bench.meta = {}

            stats = result["stats"]
            bench.meta["eval_result"] = stats
            bench.meta["eval_detail_path"] = result["detail_path"]

            # === Pass Key Mapping to MetricRunner ===
            final_key_mapping = result.get("key_mapping", {})
            eval_type = bench.bench_dataflow_eval_type

            # 1. 确定 Pred Key (预测列)
            # 逻辑：优先查看映射中是否指定了 'input_pred_key'，如果没有，则默认为 'generated_ans'
            # 这兼容了 Pipeline 中生成的结果，也兼容了用户手动指定列名的情况
            default_pred_key = "generated_ans"
            mapped_pred_key = final_key_mapping.get("input_pred_key")
            pred_key = mapped_pred_key if mapped_pred_key else default_pred_key

            # 2. 确定 Ref Key (参考答案列)
            ref_key = None

            if eval_type == "key2_qa":
                # 问答（单参考）：取 target
                ref_key = final_key_mapping.get("input_target_key")

            elif eval_type == "key2_q_ma":
                # 问答（多参考）：取 targets
                ref_key = final_key_mapping.get("input_targets_key")

            elif eval_type == "key3_q_choices_a":
                # 单选：取 label (A/B/C)
                ref_key = final_key_mapping.get("input_label_key")
                # 选择题优先用 ll-choice 的预测列
                pred_key = "eval_pred"

            elif eval_type == "key3_q_choices_as":
                # 多选：取 labels 集合
                ref_key = final_key_mapping.get("input_labels_key")
                pred_key = "eval_pred"

            elif eval_type == "key3_q_a_rejected":
                # 偏好对比：通常将 "better" 视为正例 (Positive Reference)
                # 注意：MetricRunner 可能还需要 rejected_key，但通常 ref_key 指向 ground truth
                ref_key = final_key_mapping.get("input_better_key")

            elif eval_type == "key1_text_score":
                # 文本评分 (PPL)：属于无监督评估，没有 Ref
                # 特殊情况：此时我们要评估的对象(pred)其实就是输入的 text 列
                ref_key = None
                # 如果映射里指定了 input_text_key，它就是我们要评估的内容
                if final_key_mapping.get("input_text_key"):
                    pred_key = final_key_mapping.get("input_text_key")

            # 3. 写入 Meta
            if ref_key:
                bench.meta["ref_key"] = ref_key
                self.logger.info(f"[{bench.bench_name}] Set ref_key='{ref_key}' based on type '{eval_type}'")

            bench.meta["pred_key"] = pred_key
            self.logger.info(f"[{bench.bench_name}] Set pred_key='{pred_key}'")

            bench.eval_status = "success"

            total_samples = stats.get("total_samples", 0)
            accuracy = stats.get("accuracy", 0)
            score = stats.get("score", 0)
            valid_samples = stats.get("valid_samples", 0)

            if total_samples > 0 and (accuracy == 0 and score == 0):
                reason = "Score is 0. Possibly a hidden test set without public labels."
                if valid_samples == 0:
                    reason += " (No valid samples found for evaluation)"

                bench.meta["eval_abnormality"] = {
                    "is_abnormal": True,
                    "reason": reason,
                    "type": "zero_score"
                }
                self.logger.warning(f"[{bench.bench_name}] Detected abnormality: {reason}")

            self.logger.info(f"[{bench.bench_name}] 评测完成。Stats: {result['stats']}")

        except Exception as e:
            self.logger.error(f"[{bench.bench_name}] 评测失败: {e}")
            bench.eval_status = "failed"
            if not bench.meta:
                bench.meta = {}
            bench.meta["eval_error"] = str(e)

        state.eval_cursor = cursor + 1
        return state
