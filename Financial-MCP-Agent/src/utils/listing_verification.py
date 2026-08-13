"""Deterministic preflight checks for A-share listing identity."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from src.tools.mcp_client import get_all_tools_for_orchestrator


class ListingStatus(str, Enum):
    LISTED = "listed"
    UNLISTED = "unlisted"
    DELISTED = "delisted"
    AMBIGUOUS = "ambiguous"
    NOT_SUPPORTED = "not_supported"
    UNKNOWN = "unknown"


class ListingCheckResult(BaseModel):
    """Machine-readable decision made before any analysis agent is started."""

    model_config = ConfigDict(extra="forbid")

    status: ListingStatus
    message: str
    company_name: str | None = None
    stock_code: str | None = None
    exchange: str | None = None
    market: str = "A-share"
    source: str = "a_share_mcp.resolve_stock_listing"
    candidates: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def may_start_workflow(self) -> bool:
        return self.status == ListingStatus.LISTED and bool(self.stock_code)


def normalize_a_share_symbol(value: str | None) -> str | None:
    """Normalize a supported A-share symbol or reject other code formats."""
    if not value:
        return None
    candidate = str(value).strip().lower()
    if re.fullmatch(r"(?:sh|sz)\.\d{6}", candidate):
        return candidate
    if re.fullmatch(r"\d{6}", candidate):
        if candidate.startswith("6"):
            return f"sh.{candidate}"
        if candidate.startswith(("0", "3")):
            return f"sz.{candidate}"
    raise ValueError("仅支持 A 股代码，例如 600519、sh.600519 或 sz.300750。")


def _normalize_company_name(value: str | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[\s·•・,，.。()（）\-—_]", "", str(value)).lower()
    for suffix in ("股份有限公司", "有限责任公司", "有限公司", "集团", "公司"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def company_names_compatible(query_name: str | None, canonical_name: str | None) -> bool:
    """Allow common short names while still catching obvious code/name conflicts."""
    query = _normalize_company_name(query_name)
    canonical = _normalize_company_name(canonical_name)
    if not query or not canonical:
        return True
    return query == canonical or (
        min(len(query), len(canonical)) >= 2
        and (query in canonical or canonical in query)
    )


def _unsupported_market_reason(
    stock_code: str | None,
    market_hint: str | None,
) -> str | None:
    code = str(stock_code or "").strip().lower()
    hint = str(market_hint or "").strip().lower()
    if code.startswith("hk.") or re.fullmatch(r"\d{5}", code):
        return "当前证券主数据只支持 A 股，暂不支持港股代码。"
    unsupported_hints = ("hk", "港股", "香港", "us", "美股", "nasdaq", "nyse")
    if any(token in hint for token in unsupported_hints):
        return "当前证券主数据只支持 A 股，无法验证指定的境外市场。"
    return None


def _decode_json_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _unwrap_tool_result(result: Any) -> tuple[Any, dict[str, Any] | None]:
    """Unwrap LangChain/MCP content while retaining structured provider errors."""
    if isinstance(result, str):
        result = _decode_json_text(result)
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    if isinstance(result, dict) and result.get("type") == "text":
        return _unwrap_tool_result(result.get("text"))
    if isinstance(result, dict) and {"ok", "data", "error"} <= result.keys():
        if not result.get("ok"):
            error = result.get("error")
            return None, error if isinstance(error, dict) else {"message": str(error)}
        data = result.get("data") or {}
        if isinstance(data, dict) and "content" in data:
            return data["content"], None
        return data, None
    return result, None


def _find_tool(tools: Iterable[Any], name: str) -> Any | None:
    return next((tool for tool in tools if getattr(tool, "name", None) == name), None)


async def verify_a_share_listing(
    extraction_result: dict[str, Any],
    *,
    tools: Iterable[Any] | None = None,
) -> ListingCheckResult:
    """Resolve and gate a requested security before the analysis workflow starts."""
    company_name = extraction_result.get("company_name")
    raw_code = extraction_result.get("stock_code")

    if extraction_result.get("needs_clarification"):
        reason = extraction_result.get("clarification_reason") or "公司或市场存在歧义。"
        return ListingCheckResult(
            status=ListingStatus.AMBIGUOUS,
            company_name=company_name,
            message=f"无法唯一确定要分析的证券：{reason} 请提供 A 股代码。",
        )

    if not company_name and not raw_code:
        return ListingCheckResult(
            status=ListingStatus.AMBIGUOUS,
            message="没有识别到公司名称或 A 股代码，请提供例如“贵州茅台 600519”。",
        )

    unsupported_reason = _unsupported_market_reason(
        raw_code,
        extraction_result.get("llm_market"),
    )
    if unsupported_reason:
        return ListingCheckResult(
            status=ListingStatus.NOT_SUPPORTED,
            company_name=company_name,
            message=unsupported_reason + " 分析已停止。",
        )

    try:
        stock_code = normalize_a_share_symbol(raw_code)
    except ValueError as error:
        return ListingCheckResult(
            status=ListingStatus.NOT_SUPPORTED,
            company_name=company_name,
            message=f"{error} 分析已停止。",
        )

    available_tools = list(tools) if tools is not None else await get_all_tools_for_orchestrator()
    resolver = _find_tool(available_tools, "resolve_stock_listing")
    if resolver is None:
        return ListingCheckResult(
            status=ListingStatus.UNKNOWN,
            company_name=company_name,
            stock_code=stock_code,
            message="上市状态校验工具不可用。为避免分析错误标的，本次分析已停止。",
        )

    arguments = {"code": stock_code} if stock_code else {"company_name": company_name}
    try:
        raw_result = await resolver.ainvoke(arguments)
    except Exception as error:
        return ListingCheckResult(
            status=ListingStatus.UNKNOWN,
            company_name=company_name,
            stock_code=stock_code,
            message=f"上市状态校验失败：{error}。为避免误分析，本次分析已停止。",
        )

    content, provider_error = _unwrap_tool_result(raw_result)
    if provider_error:
        provider_message = provider_error.get("message") or provider_error.get("code") or "未知错误"
        return ListingCheckResult(
            status=ListingStatus.UNKNOWN,
            company_name=company_name,
            stock_code=stock_code,
            message=f"上市状态数据源返回错误：{provider_message}。本次分析已停止。",
        )
    if not isinstance(content, dict):
        return ListingCheckResult(
            status=ListingStatus.UNKNOWN,
            company_name=company_name,
            stock_code=stock_code,
            message="上市状态数据格式无效。为避免误分析，本次分析已停止。",
        )

    candidates = [item for item in content.get("candidates", []) if isinstance(item, dict)]
    if stock_code:
        candidates = [item for item in candidates if item.get("stock_code") == stock_code]

    if not candidates:
        label = company_name or stock_code or "该公司"
        return ListingCheckResult(
            status=ListingStatus.NOT_SUPPORTED,
            company_name=company_name,
            stock_code=stock_code,
            candidates=[],
            message=(
                f"未在当前 A 股证券主数据中找到“{label}”。分析已停止；"
                "这表示没有可由本系统分析的 A 股匹配项，不代表已验证全球所有市场。"
            ),
        )

    if len(candidates) > 1:
        return ListingCheckResult(
            status=ListingStatus.AMBIGUOUS,
            company_name=company_name,
            stock_code=stock_code,
            candidates=candidates,
            message="找到多个可能的 A 股证券，请提供准确的六位股票代码。",
        )

    candidate = candidates[0]
    canonical_name = str(candidate.get("company_name") or "") or None
    canonical_code = str(candidate.get("stock_code") or "") or None
    if company_name and not company_names_compatible(company_name, canonical_name):
        return ListingCheckResult(
            status=ListingStatus.AMBIGUOUS,
            company_name=canonical_name,
            stock_code=canonical_code,
            exchange=candidate.get("exchange"),
            candidates=candidates,
            message=(
                f"股票代码 {canonical_code} 对应“{canonical_name}”，与输入的"
                f"“{company_name}”不一致。请确认公司名称或代码。"
            ),
        )

    raw_status = str(candidate.get("listing_status") or "unknown").lower()
    status_map = {
        "listed": ListingStatus.LISTED,
        "unlisted": ListingStatus.UNLISTED,
        "delisted": ListingStatus.DELISTED,
    }
    status = status_map.get(raw_status, ListingStatus.UNKNOWN)
    messages = {
        ListingStatus.LISTED: f"已确认 A 股上市证券：{canonical_name}（{canonical_code}）。",
        ListingStatus.UNLISTED: f"{canonical_name or company_name}尚未上市，股票分析已停止。",
        ListingStatus.DELISTED: f"{canonical_name or company_name}已退市，股票分析已停止。",
        ListingStatus.UNKNOWN: "证券存在，但上市状态无法确认。为避免误分析，本次分析已停止。",
    }
    return ListingCheckResult(
        status=status,
        company_name=canonical_name or company_name,
        stock_code=canonical_code or stock_code,
        exchange=candidate.get("exchange"),
        candidates=candidates,
        message=messages[status],
    )


__all__ = [
    "ListingCheckResult",
    "ListingStatus",
    "company_names_compatible",
    "normalize_a_share_symbol",
    "verify_a_share_listing",
]
