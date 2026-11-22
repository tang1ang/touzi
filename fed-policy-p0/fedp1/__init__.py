"""P1：在 P0 市场先验上叠加宏观反应函数与 FOMC 沟通文本信号。"""

from .combine import build_combination, decompose, pool
from .macro import MacroConfig, build_macro_path, build_macro_signal
from .pipeline import build_p1
from .text import build_text_signal, score_text, score_statement

__all__ = [
    "MacroConfig",
    "build_p1",
    "build_macro_signal",
    "build_macro_path",
    "build_text_signal",
    "score_text",
    "score_statement",
    "build_combination",
    "decompose",
    "pool",
]
