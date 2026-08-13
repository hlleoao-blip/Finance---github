"""Deterministic local technical indicators derived from validated OHLCV data."""

from __future__ import annotations

from typing import Any


def _number(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_ohlcv(value: Any) -> list[dict[str, Any]]:
    """Accept tool dictionaries, lists, or the MCP markdown-table representation."""
    if isinstance(value, dict):
        for key in ("data", "records", "items", "results", "content"):
            if key in value:
                return parse_ohlcv(value[key])
        return []
    if isinstance(value, list):
        rows = value
    elif isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip().startswith("|")]
        if len(lines) < 3:
            return []
        headers = [item.strip().casefold() for item in lines[0].strip("|").split("|")]
        rows = []
        for line in lines[2:]:
            cells = [item.strip() for item in line.strip("|").split("|")]
            if len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
    else:
        return []

    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = {str(key).casefold(): item for key, item in row.items()}
        close = _number(normalized.get("close"))
        high = _number(normalized.get("high"))
        low = _number(normalized.get("low"))
        if close is None or high is None or low is None:
            continue
        parsed.append(
            {
                "date": str(normalized.get("date") or normalized.get("datetime") or ""),
                "open": _number(normalized.get("open")),
                "high": high,
                "low": low,
                "close": close,
                "volume": _number(normalized.get("volume")),
            }
        )
    return parsed


def _ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    output = [values[0]]
    for value in values[1:]:
        output.append(alpha * value + (1 - alpha) * output[-1])
    return output


def _rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) <= period:
        return None
    changes = [right - left for left, right in zip(values, values[1:])]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]
    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period
    for gain, loss in zip(gains[period:], losses[period:]):
        average_gain = (average_gain * (period - 1) + gain) / period
        average_loss = (average_loss * (period - 1) + loss) / period
    if average_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + average_gain / average_loss)


def calculate_technical_snapshot(value: Any) -> dict[str, Any]:
    rows = parse_ohlcv(value)
    if len(rows) < 20:
        return {"available": False, "record_count": len(rows)}
    closes = [row["close"] for row in rows]
    highs = [row["high"] for row in rows]
    lows = [row["low"] for row in rows]
    volumes = [row["volume"] for row in rows if row["volume"] is not None]
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    dif = [left - right for left, right in zip(ema12, ema26)]
    dea = _ema(dif, 9)
    latest_window = rows[-20:]

    def mean(period: int) -> float | None:
        return sum(closes[-period:]) / period if len(closes) >= period else None

    volume_ratio = None
    if len(volumes) >= 25:
        recent = sum(volumes[-5:]) / 5
        baseline = sum(volumes[-25:-5]) / 20
        volume_ratio = recent / baseline if baseline else None

    snapshot = {
        "available": True,
        "record_count": len(rows),
        "as_of": rows[-1]["date"],
        "latest_close": closes[-1],
        "ma5": mean(5),
        "ma10": mean(10),
        "ma20": mean(20),
        "ma60": mean(60),
        "rsi14": _rsi(closes),
        "macd_dif": dif[-1],
        "macd_dea": dea[-1],
        "macd_histogram": 2 * (dif[-1] - dea[-1]),
        "support_20d": min(row["low"] for row in latest_window),
        "resistance_20d": max(row["high"] for row in latest_window),
        "period_low": min(lows),
        "period_high": max(highs),
        "volume_5d_to_prior_20d": volume_ratio,
    }
    return {
        key: round(item, 4) if isinstance(item, float) else item
        for key, item in snapshot.items()
    }


def merge_markdown_ohlcv(left: Any, right: Any) -> Any:
    """Merge incremental markdown rows while preserving one header and unique dates."""
    if not isinstance(left, str) or not isinstance(right, str):
        return right
    left_lines = [line for line in left.splitlines() if line.strip()]
    right_lines = [line for line in right.splitlines() if line.strip()]
    if len(left_lines) < 3 or len(right_lines) < 3:
        return right
    rows: dict[str, str] = {}
    for line in [*left_lines[2:], *right_lines[2:]]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells:
            rows[cells[0]] = line
    return "\n".join([left_lines[0], left_lines[1], *[rows[key] for key in sorted(rows)]])
