import logging
import os
import re
import json
import ast
from logging.handlers import RotatingFileHandler

# === 默认配置 ===
LOG_FILE = os.getenv("ONE_EVAL_LOG_FILE", "one_eval.log")
LOG_LEVEL = os.getenv("ONE_EVAL_LOG_LEVEL", "INFO").upper()
MAX_SIZE = 10 * 1024 * 1024
BACKUP_COUNT = 3
PRETTY_JSON = os.getenv("ONE_EVAL_LOG_PRETTY_JSON", "1").lower() in {"1", "true", "yes"}

# === 颜色映射 ===
COLOR = {
    "DEBUG": "\033[36m",     # 青色
    "INFO": "\033[32m",      # 绿色
    "WARNING": "\033[33m",   # 黄色
    "ERROR": "\033[31m",     # 红色
    "CRITICAL": "\033[41m\033[37m",  # 红底白字
    "RESET": "\033[0m",
}
FIELD = {
    "time": "\033[90m",
    "name": "\033[35m",
    "loc": "\033[94m",
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        reset = COLOR["RESET"]
        msg = record.getMessage()
        if PRETTY_JSON:
            msg = _prettify_message(msg)
        return (
            f"{FIELD['time']}{self.formatTime(record)}{reset} | "
            f"{COLOR.get(record.levelname, '')}{record.levelname:<8}{reset} | "
            f"{FIELD['name']}{record.name}{reset} | "
            f"{FIELD['loc']}{record.filename}:{record.lineno}{reset} | "
            f"{COLOR.get(record.levelname, '')}{msg}{reset}"
        )


def _prettify_message(msg: str) -> str:
    s = msg
    try:
        pattern = r"```json\s*([\s\S]*?)```"
        def repl(m):
            text = m.group(1)
            obj = None
            try:
                obj = json.loads(text)
            except Exception:
                try:
                    obj = ast.literal_eval(text)
                except Exception:
                    return m.group(0)
            return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"
        new_msg = re.sub(pattern, repl, s)
        if new_msg != s:
            return new_msg
    except Exception:
        pass

    def find_json_end(text: str, start: int) -> int | None:
        stack = []
        in_str = False
        esc = False
        for j in range(start, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            else:
                if c == '"':
                    in_str = True
                    continue
                if c in '{[':
                    stack.append(c)
                elif c in '}]':
                    if not stack:
                        return None
                    open_c = stack.pop()
                    if (open_c == '{' and c != '}') or (open_c == '[' and c != ']'):
                        return None
                    if not stack:
                        return j
        return None

    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in '{[':
            end = find_json_end(s, i)
            if end is not None:
                segment = s[i:end+1]
                obj = None
                try:
                    obj = json.loads(segment)
                except Exception:
                    try:
                        obj = ast.literal_eval(segment)
                    except Exception:
                        obj = None
                if obj is not None and isinstance(obj, (dict, list, str, int, float, bool, type(None))):
                    pretty = json.dumps(obj, ensure_ascii=False, indent=2)
                    out.append(pretty)
                    i = end + 1
                    continue
        out.append(ch)
        i += 1
    return ''.join(out)


class PlainFormatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        if PRETTY_JSON:
            msg = _prettify_message(msg)
        record.msg = msg
        record.args = None
        return super().format(record)

def _make_handlers():
    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)
    console.setFormatter(ColorFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))

    file = RotatingFileHandler(LOG_FILE, maxBytes=MAX_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8")
    file.setLevel(LOG_LEVEL)
    file.setFormatter(PlainFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    return [console, file]

def get_logger(name: str = "one_eval") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        for h in _make_handlers():
            logger.addHandler(h)
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
    return logger

log = get_logger()

if __name__ == "__main__":
    log.info("Logger ready.")
    log.warning("Warning example.")
    log.error("Error example.")
