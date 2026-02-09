from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from one_eval.logger import get_logger

log = get_logger("HFDownloadTool")


class HFDownloadTool:
    """
    负责从 HuggingFace 下载指定 Dataset 的特定 Config 和 Split，
    并将其转换为统一的 JSONL 格式保存。
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        offline: Optional[bool] = None,
    ):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.cache_dir = cache_dir
        if offline is None:
            offline = os.getenv("HF_HUB_OFFLINE", "0") in ("1", "true", "True")
        self.offline = offline

    def _ensure_dir(self, p: Union[str, Path]) -> Path:
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _import_datasets(self):
        try:
            from datasets import load_dataset
            return load_dataset
        except Exception as e:
            raise RuntimeError(
                "缺少 datasets 依赖或导入失败 请先 pip install datasets"
            ) from e

    def download_and_convert(
        self,
        repo_id: str,
        config_name: str,
        split: str,
        output_path: Union[str, Path],
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        下载指定 config/split 并转换为 jsonl
        """
        load_dataset = self._import_datasets()
        out_path = Path(output_path)
        self._ensure_dir(out_path.parent)

        log.info(f"开始下载并转换: repo={repo_id} config={config_name} split={split} -> {out_path}")

        try:
            kwargs: Dict[str, Any] = {
                "split": split,
                "revision": revision,
                "token": self.hf_token,
                "cache_dir": self.cache_dir,
            }
            try:
                ds = load_dataset(repo_id, config_name, trust_remote_code=True, **kwargs)
            except TypeError:
                ds = load_dataset(repo_id, config_name, **kwargs)
            except Exception as e:
                msg = str(e)
                if "trust_remote_code" in msg:
                    ds = load_dataset(repo_id, config_name, **kwargs)
                else:
                    raise
        except Exception as e:
            error_msg = f"load_dataset 失败: {e}"
            log.error(error_msg)
            return {"ok": False, "error": error_msg}

        try:
            count = 0
            with out_path.open("w", encoding="utf-8") as f:
                for item in ds:
                    # 转换为 dict
                    # 为了防止包含非 JSON 序列化的对象（如 PIL.Image），这里做一个简单的清洗
                    clean_item = {}
                    for k, v in item.items():
                        try:
                            # 尝试序列化
                            json.dumps({k: v})
                            clean_item[k] = v
                        except (TypeError, OverflowError):
                            # 如果失败，转换为字符串
                            # TODO: 如果是 Image/Audio，可能需要更复杂的处理（如保存文件路径）
                            # 但对于目前的评测任务，通常是文本/JSON 兼容的
                            clean_item[k] = str(v)
                    
                    f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
                    count += 1
            
            log.info(f"转换完成，共 {count} 条数据")
            return {
                "ok": True,
                "output_path": str(out_path),
                "num_rows": count,
                "columns": list(ds.column_names) if hasattr(ds, "column_names") else [],
            }

        except Exception as e:
            error_msg = f"写入 jsonl 失败: {e}"
            log.error(error_msg)
            return {"ok": False, "error": error_msg}
