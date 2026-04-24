from one_eval.core.state import ModelConfig
from one_eval.toolkits.dataflow_eval_tool import DataFlowEvalTool
import pandas as pd


def test_deepseek_api_url_and_extra_body_are_preserved():
    cfg = ModelConfig(
        model_name_or_path="deepseek-v4-pro",
        is_api=True,
        api_provider="deepseek",
        api_url="https://api.deepseek.com",
        api_extra_body={
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "stream": False,
        },
        temperature=0.3,
        top_p=0.95,
        max_tokens=4096,
        api_max_workers=8,
        api_connect_timeout=5.0,
        api_read_timeout=90.0,
    )

    kwargs = DataFlowEvalTool._build_api_serving_kwargs(cfg)

    assert kwargs["api_url"] == "https://api.deepseek.com/chat/completions"
    assert kwargs["model_name"] == "deepseek-v4-pro"
    assert kwargs["thinking"] == {"type": "enabled"}
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["stream"] is False
    assert kwargs["temperature"] == 0.3
    assert kwargs["top_p"] == 0.95
    assert kwargs["max_tokens"] == 4096
    assert kwargs["max_workers"] == 8
    assert kwargs["connect_timeout"] == 5.0
    assert kwargs["read_timeout"] == 90.0


def test_local_model_config_does_not_mix_api_fields():
    cfg = ModelConfig(
        model_name_or_path="/models/qwen",
        is_api=False,
        tensor_parallel_size=2,
        max_model_len=16384,
        gpu_memory_utilization=0.8,
    )

    assert cfg.is_api is False
    assert cfg.tensor_parallel_size == 2
    assert cfg.max_model_len == 16384
    assert cfg.gpu_memory_utilization == 0.8
    assert cfg.api_extra_body == {}


def test_preprocess_dataframe_normalizes_dict_choices_and_label_key():
    tool = DataFlowEvalTool(output_root="cache/test_eval_results")
    df = pd.DataFrame(
        [
            {
                "question": "Q1",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma"},
                "answer_idx": "B",
                "answer": "Beta",
            }
        ]
    )
    key_mapping = {
        "input_question_key": "question",
        "input_choices_key": "options",
    }

    out_df, out_mapping = tool._preprocess_dataframe(
        df,
        bench_name="medqa_like",
        key_mapping=key_mapping,
        eval_type="key3_q_choices_a",
    )

    assert out_mapping["input_choices_key"] == "normalized_choices"
    assert out_mapping["input_label_key"] == "answer_idx"
    assert out_df.loc[0, "normalized_choices"] == ["Alpha", "Beta", "Gamma"]


def test_normalize_label_supports_answer_text_matching():
    tool = DataFlowEvalTool(output_root="cache/test_eval_results")
    choices = ["Alpha", "Beta", "Gamma"]

    assert tool._normalize_label_to_index("B", choices) == 1
    assert tool._normalize_label_to_index("Beta", choices) == 1
