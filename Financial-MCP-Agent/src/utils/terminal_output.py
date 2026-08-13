"""终端输出控制工具。"""
import re
from typing import Any


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "\uFE0F"
    "]+"
)


def strip_terminal_emoji(value: Any) -> str:
    """移除终端输出中的表情符号，保留正文内容。"""
    return EMOJI_PATTERN.sub("", str(value))


def print_analysis_output(label: str, content: Any, enabled: bool = False) -> None:
    """按开关打印 Agent 分析结果，默认不向终端输出大段分析。"""
    if not enabled:
        return

    print(f"{label}: {strip_terminal_emoji(content)}")
