"""CME 30-Day Fed Funds Futures (ZQ) 合约月份编码。"""

from __future__ import annotations

import re

MONTH_CODES = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}
CODE_TO_MONTH = {code: month for month, code in MONTH_CODES.items()}

_SYMBOL_RE = re.compile(r"^ZQ([FGHJKMNQUVXZ])(\d{2})$")


def contract_symbol(year: int, month: int) -> str:
    """返回指定年月的 ZQ 合约代码，例如 (2026, 12) -> ZQZ26。"""
    if month not in MONTH_CODES:
        raise ValueError(f"非法月份: {month}")
    if not 2000 <= year <= 2099:
        raise ValueError(f"年份超出两位编码范围: {year}")
    return f"ZQ{MONTH_CODES[month]}{year % 100:02d}"


def parse_contract(symbol: str) -> tuple[int, int]:
    """解析 ZQ 合约代码，返回 (year, month)。"""
    match = _SYMBOL_RE.match(symbol.strip().upper())
    if not match:
        raise ValueError(f"非法 ZQ 合约代码: {symbol!r}")
    code, yy = match.groups()
    return 2000 + int(yy), CODE_TO_MONTH[code]
