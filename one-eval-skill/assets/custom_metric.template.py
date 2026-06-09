"""
custom_metric.template.py — 用户自定义 metric 模板（与 agent 聊天即可生成新评分维度）。

怎么用：
  1. 复制本文件到 one-eval-skill/custom_metrics/ 下，改个有意义的文件名
     （如 my_keyword_hit.py）。custom_metrics/*.py 不入库，是你的本地资产。
  2. 按下面的契约实现一个 compute_xxx 函数，用 @register_metric 注册。
  3. 评测引擎启动时会自动扫描并加载（与内置 metric 同等地位）。
  4. 在 evalspec.yaml 的 metrics 段用注册名引用，或 run_metrics.py --metrics <名> 调用。

契约（务必遵守，否则 MetricRunner 无法正确聚合）：
  函数签名：compute_xxx(preds, refs, **kwargs) -> Dict
    - preds: List[Any]  模型预测（已由引擎抽取，通常是字符串）
    - refs:  List[Any]  参考答案（与 preds 等长、一一对应）
    - **kwargs: 运行时附加参数；引擎会注入 all_metric_results（已算出的其他 metric），
                以及 evalspec metrics[].args 里你自定义的参数
  返回值：必须含两个键
    - "score":   float        语料级总分（聚合后的单一数值）
    - "details": List[float]  每条样本的分数，长度需与 preds 一致（便于并行分块合并）
  失败时也尽量返回 {"score": 0.0, "details": [...]}，不要抛异常吞掉全表。
"""
from typing import List, Any, Dict

from one_eval.core.metric_registry import register_metric, MetricCategory


@register_metric(
    name="keyword_hit",                          # 注册名；evalspec/CLI 用它引用
    desc="关键词命中率：预测中是否包含参考答案的关键词",   # 给 agent 理解用
    usage="适合答案为短语/实体、只需判断是否命中的开放问答",  # 给 agent 推荐用
    categories=[MetricCategory.QA_SINGLE],       # 归属的 eval_type（可多个）
    aliases=["kw_hit"],                          # 别名（可选）
)
def compute_keyword_hit(preds: List[Any], refs: List[Any], **kwargs) -> Dict[str, Any]:
    """逐条判断预测是否包含参考答案（大小写不敏感），返回命中率。

    这是一个最小可用示例：把它替换成你真正需要的打分逻辑即可。
    需要 LLM 裁判的复杂 metric 同理 —— 在函数内调用你的判定逻辑，
    最终仍返回 {"score", "details"}。
    """
    case_sensitive = kwargs.get("case_sensitive", False)

    if not preds or not refs:
        return {"score": 0.0, "details": []}

    details: List[float] = []
    for p, r in zip(preds, refs):
        pred_s = "" if p is None else str(p)
        ref_s = "" if r is None else str(r)
        if not case_sensitive:
            pred_s, ref_s = pred_s.lower(), ref_s.lower()
        hit = 1.0 if ref_s.strip() and ref_s.strip() in pred_s else 0.0
        details.append(hit)

    score = sum(details) / len(details) if details else 0.0
    return {"score": score, "details": details}
