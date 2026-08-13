"""Type-specific deterministic validation and usable-record counting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class RecordValidation:
    data: Any
    record_count: int
    rejected_count: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)


def _markdown_rows(value: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in value.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return []

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    headers = cells(lines[0])
    separators = cells(lines[1])
    if len(headers) != len(separators) or not all(
        "-" in part and part.replace(":", "").replace("-", "") == ""
        for part in separators
    ):
        return []
    return [
        dict(zip(headers, values))
        for line in lines[2:]
        if len(values := cells(line)) == len(headers)
    ]


def _list_container(value: Any) -> tuple[str | None, list[Any]]:
    if isinstance(value, list):
        return None, value
    if isinstance(value, dict):
        for key in ("items", "results", "records", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                return key, nested
    return None, []


def _replace_rows(value: Any, key: str | None, rows: list[Any]) -> Any:
    if isinstance(value, list):
        return rows
    if isinstance(value, dict) and key is not None:
        return {**value, key: rows}
    return value


def _company_aliases(company_name: str) -> set[str]:
    compact = re.sub(r"[\s·・,，。()（）\-]", "", company_name or "")
    aliases = {compact} if len(compact) >= 2 else set()
    for suffix in ("股份有限公司", "有限责任公司", "集团股份", "集团", "股份", "有限公司"):
        if compact.endswith(suffix):
            shortened = compact[: -len(suffix)]
            if len(shortened) >= 2:
                aliases.add(shortened)
    return aliases


def _entity_text(item: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "entities",
        "entity_names",
        "mentioned_companies",
        "recognized_entities",
    ):
        value = item.get(key)
        if isinstance(value, dict):
            values.extend(str(nested) for nested in value.values())
        elif isinstance(value, (list, tuple, set)):
            values.extend(str(nested) for nested in value)
        elif value:
            values.append(str(value))
    return re.sub(r"\s+", "", " ".join(values)).lower()


def _news_validation(value: Any, arguments: dict[str, Any]) -> RecordValidation:
    key, rows = _list_container(value)
    company_name = str(arguments.get("company_name") or "")
    canonical = str(arguments.get("code") or "").lower()
    digits = canonical.rsplit(".", 1)[-1]
    aliases = _company_aliases(company_name)
    start_date = str(arguments.get("start_date") or "")
    end_date = str(arguments.get("end_date") or "")
    accepted: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()

    def reject(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for raw in rows:
        if not isinstance(raw, dict):
            reject("INVALID_SCHEMA")
            continue
        item = dict(raw)
        observed = str(
            item.get("date")
            or item.get("publish_date")
            or item.get("published_at")
            or item.get("pubDate")
            or ""
        ).strip()
        if not observed:
            reject("MISSING_DATE")
            continue
        try:
            normalized_date = date.fromisoformat(observed[:10]).isoformat()
        except ValueError:
            reject("INVALID_DATE")
            continue
        if start_date and normalized_date < start_date or end_date and normalized_date > end_date:
            reject("OUTSIDE_REQUEST_WINDOW")
            continue

        editorial = re.sub(
            r"\s+",
            "",
            f"{item.get('title') or ''} {item.get('summary') or ''}",
        ).lower()
        entities = _entity_text(item)
        relevant = bool(
            (digits and digits in editorial)
            or (canonical and canonical in editorial)
            or any(alias.lower() in editorial for alias in aliases)
            or (digits and digits in entities)
            or (canonical and canonical in entities)
            or any(alias.lower() in entities for alias in aliases)
        )
        if not relevant:
            reject("TARGET_NOT_MENTIONED")
            continue
        content_key = re.sub(
            r"\s+", "", str(item.get("title") or item.get("summary") or "")
        )
        dedupe_key = (normalized_date, content_key)
        if not content_key:
            reject("MISSING_CONTENT")
            continue
        if dedupe_key in seen:
            reject("DUPLICATE")
            continue
        seen.add(dedupe_key)
        item["date"] = normalized_date
        item["content_type"] = "news"
        accepted.append(item)

    return RecordValidation(
        data=_replace_rows(value, key, accepted),
        record_count=len(accepted),
        rejected_count=sum(reasons.values()),
        rejection_reasons=reasons,
    )


def _announcement_validation(value: Any) -> RecordValidation:
    key, rows = _list_container(value)
    accepted = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("date") and row.get("title")
    ]
    return RecordValidation(
        data=_replace_rows(value, key, accepted),
        record_count=len(accepted),
        rejected_count=len(rows) - len(accepted),
        rejection_reasons={"MISSING_DATE_OR_TITLE": len(rows) - len(accepted)}
        if len(rows) != len(accepted)
        else {},
    )


def _market_rows_validation(value: Any) -> RecordValidation:
    if isinstance(value, str):
        rows = _markdown_rows(value)
    else:
        _key, rows = _list_container(value)
    accepted = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        lowered = {str(field).lower(): nested for field, nested in row.items()}
        if not (lowered.get("date") or lowered.get("datetime")):
            continue
        if all(lowered.get(field) not in (None, "") for field in ("open", "high", "low", "close")):
            accepted.append(row)
    return RecordValidation(value, len(accepted), len(rows) - len(accepted))


def _generic_count(value: Any) -> int:
    if value is None or value == "" or value == [] or value == {}:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("items", "results", "records", "data"):
            if isinstance(value.get(key), list):
                return len(value[key])
        return 1
    if isinstance(value, str):
        rows = _markdown_rows(value)
        if rows:
            return len(rows)
        numbered = re.findall(r"(?m)^\s*\d+[\.、]\s+", value)
        return len(numbered) if numbered else 1
    return 1


def validate_tool_records(
    tool_name: str | None,
    value: Any,
    arguments: dict[str, Any] | None = None,
) -> RecordValidation:
    """Return sanitized data and a count whose meaning is specific to the tool."""
    arguments = arguments or {}
    if tool_name == "get_financial_news":
        return _news_validation(value, arguments)
    if tool_name == "get_official_announcements":
        return _announcement_validation(value)
    if tool_name in {"get_historical_k_data", "get_mootdx_bars"}:
        return _market_rows_validation(value)
    if tool_name == "get_eastmoney_signals":
        flows = value.get("capital_flow") if isinstance(value, dict) else None
        return RecordValidation(value, len(flows) if isinstance(flows, list) else 0)
    if tool_name == "get_tencent_quote":
        usable = isinstance(value, dict) and value.get("price") not in (None, "")
        return RecordValidation(value, int(usable))
    return RecordValidation(value, _generic_count(value))
