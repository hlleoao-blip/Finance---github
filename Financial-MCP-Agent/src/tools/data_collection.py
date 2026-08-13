"""Deterministic shared-data plan and run-scoped collection cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable
from uuid import uuid4

from src.agent_loop.contracts import PlanStep, SymbolScope
from src.agent_loop.executor import MCPToolExecutor
from src.agent_loop.validator import FinancialDataValidator
from src.tools.call_control import (
    CategoryBudget,
    allocate_category_budget,
    get_run_call_control,
    provider_for_tool,
)
from src.tools.mcp_client import get_all_tools_for_orchestrator
from src.tools.persistent_cache import PersistentToolCache
from src.tools.data_quality import validate_tool_records
from src.utils.technical_indicators import (
    calculate_technical_snapshot,
    merge_markdown_ohlcv,
)
from src.utils.state_definition import DataFetchStatus, WorkflowState


ArgumentsFactory = Callable[[WorkflowState], dict[str, Any]]


FALLBACK_POLICY: dict[str, Any] = {
    "market": ["baostock", "tencent", "mootdx"],
    "capital_flow": ["eastmoney", "missing"],
    "news": ["cls", "sina_target_search"],
    "news_supplement": "official_announcements_separate_content_type",
    "retry": {
        "TIMEOUT": "retry_once_then_switch",
        "UPSTREAM_ERROR": "switch_immediately",
        "ProxyError": "switch_immediately",
        "NO_DATA": "do_not_retry_same_parameters",
        "INVALID_DATA": "switch_provider_without_changing_window",
    },
}


@dataclass(frozen=True)
class CollectionSpec:
    id: str
    requirement: str
    priority: str
    budget_category: str
    tool: str
    arguments_factory: ArgumentsFactory
    condition: str | None = None


def _completed_quarter(state: WorkflowState) -> tuple[int, int]:
    quarter = (state.as_of.month - 1) // 3
    if quarter == 0:
        return state.as_of.year - 1, 4
    return state.as_of.year, quarter


def _financial_arguments(state: WorkflowState, *, years_back: int = 0) -> dict[str, Any]:
    year, quarter = _completed_quarter(state)
    return {
        "code": state.target_symbol,
        "year": str(year - years_back),
        "quarter": quarter,
    }


def _annual_arguments(state: WorkflowState, *, years_back: int) -> dict[str, Any]:
    return {
        "code": state.target_symbol,
        "year": str(state.as_of.year - years_back),
        "quarter": 4,
    }


def _dividend_arguments(state: WorkflowState, *, years_back: int) -> dict[str, Any]:
    return {
        "code": state.target_symbol,
        "year": str(state.as_of.year - years_back),
        "year_type": "report",
    }


def _history_arguments(
    state: WorkflowState, *, days: int = 190
) -> dict[str, Any]:
    return {
        "code": state.target_symbol,
        "start_date": (state.as_of - timedelta(days=days)).isoformat(),
        "end_date": state.as_of.isoformat(),
        "frequency": "d",
        "adjust_flag": "1",
    }


def _event_arguments(state: WorkflowState) -> dict[str, Any]:
    return {
        "code": state.target_symbol,
        "start_date": (state.as_of - timedelta(days=120)).isoformat(),
        "end_date": state.as_of.isoformat(),
        "top_k": 20,
    }


def _news_arguments(state: WorkflowState) -> dict[str, Any]:
    return {
        **_event_arguments(state),
        "company_name": str(state.company_name or state.data.get("company_name") or ""),
        "top_k": 10,
    }


def _specs() -> tuple[CollectionSpec, ...]:
    specs: list[CollectionSpec] = [
        CollectionSpec("identity", "证券身份", "required", "required", "get_stock_basic_info", lambda state: {"code": state.target_symbol}),
        CollectionSpec("industry", "所属行业", "required", "required", "get_stock_industry", lambda state: {"code": state.target_symbol}),
        CollectionSpec("trading_date", "最新交易日", "required", "required", "get_latest_trading_date", lambda _state: {}),
        CollectionSpec("latest_quote", "最新行情", "required", "required", "get_tencent_quote", lambda state: {"code": state.target_symbol}),
        CollectionSpec("kline", "K线", "required", "required", "get_historical_k_data", _history_arguments),
    ]
    for tool in (
        "get_profit_data",
        "get_operation_data",
        "get_growth_data",
        "get_balance_data",
        "get_cash_flow_data",
    ):
        kind = tool.removeprefix("get_").removesuffix("_data")
        specs.append(
            CollectionSpec(
                f"latest_financial:{kind}",
                "最新财务",
                "required",
                "required",
                tool,
                _financial_arguments,
            )
        )
        specs.append(
            CollectionSpec(
                f"yoy_financial:{kind}",
                "同比财务",
                "required",
                "required",
                tool,
                lambda state, _tool=tool: _financial_arguments(state, years_back=1),
            )
        )
    specs.extend(
        [
            CollectionSpec("announcements", "公告", "required", "required", "get_official_announcements", _event_arguments),
            CollectionSpec("news", "新闻", "required", "required", "get_financial_news", _news_arguments),
            CollectionSpec("dividend:latest", "分红", "important", "cross_validation", "get_dividend_data", lambda state: _dividend_arguments(state, years_back=1)),
            CollectionSpec("dividend:prior", "分红", "important", "cross_validation", "get_dividend_data", lambda state: _dividend_arguments(state, years_back=2)),
            CollectionSpec("historical_valuation", "历史估值", "important", "cross_validation", "get_historical_k_data", lambda state: _history_arguments(state, days=370)),
            CollectionSpec("adjust_factor", "行情口径校验", "important", "cross_validation", "get_adjust_factor_data", lambda state: {"code": state.target_symbol, "start_date": (state.as_of - timedelta(days=190)).isoformat(), "end_date": state.as_of.isoformat()}),
            CollectionSpec("dupont", "杜邦", "important", "cross_validation", "get_dupont_data", _financial_arguments),
            CollectionSpec("capital_flow", "资金流", "optional", "supplemental", "get_eastmoney_signals", lambda state: {"code": state.target_symbol, "flow_days": 10}),
            CollectionSpec("backup_quote", "备用行情源", "optional", "supplemental", "get_mootdx_bars", lambda state: {"code": state.target_symbol, "frequency": "d", "count": 120}, condition="unless_two_consistent_market_sources"),
            CollectionSpec("earlier_annual", "更早年度数据", "optional", "supplemental", "get_profit_data", lambda state: _annual_arguments(state, years_back=2)),
        ]
    )
    return tuple(specs)


# Kept patchable for integration tests and deployments with a smaller tool surface.
COMMON_COLLECTION_CALLS: tuple[Any, ...] = _specs()


def collection_cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"tool": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def estimate_record_count(
    value: Any,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> int:
    """Count usable records with a validator selected for the data type."""
    return validate_tool_records(tool_name, value, arguments).record_count


def build_collection_plan(state: WorkflowState) -> list[dict[str, Any]]:
    """Materialize the complete, deterministic requirement list before execution."""
    plan: list[dict[str, Any]] = []
    for index, item in enumerate(COMMON_COLLECTION_CALLS):
        if isinstance(item, CollectionSpec):
            spec = item
        else:
            tool_name, factory = item
            spec = CollectionSpec(
                id=f"legacy:{index}:{tool_name}",
                requirement=tool_name,
                priority="required",
                budget_category="required",
                tool=tool_name,
                arguments_factory=factory,
            )
        arguments = spec.arguments_factory(state)
        materialized = {
            "id": spec.id,
            "requirement": spec.requirement,
            "priority": spec.priority,
            "budget_category": spec.budget_category,
            "tool": spec.tool,
            "provider": provider_for_tool(spec.tool),
            "arguments": arguments,
            "cache_key": collection_cache_key(spec.tool, arguments),
            "condition": spec.condition,
            "status": "PLANNED",
        }
        if spec.tool in {
            "get_historical_k_data",
            "get_tencent_quote",
            "get_mootdx_bars",
        }:
            materialized["fallback_chain"] = list(FALLBACK_POLICY["market"])
        elif spec.tool == "get_eastmoney_signals":
            materialized["fallback_chain"] = list(FALLBACK_POLICY["capital_flow"])
            materialized["failure_mode"] = "NON_BLOCKING_MISSING"
        elif spec.tool == "get_financial_news":
            materialized["fallback_chain"] = list(FALLBACK_POLICY["news"])
            materialized["announcement_supplement"] = FALLBACK_POLICY["news_supplement"]
        elif spec.tool == "get_official_announcements":
            materialized["content_type"] = "official_announcement"
            materialized["counts_as_news"] = False
        plan.append(materialized)
    return plan


def _last_markdown_row(value: Any) -> dict[str, str]:
    if not isinstance(value, str):
        return {}
    rows = [line for line in value.splitlines() if line.strip().startswith("|")]
    if len(rows) < 3:
        return {}
    headers = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
    values = [cell.strip() for cell in rows[-1].strip().strip("|").split("|")]
    return dict(zip(headers, values)) if len(headers) == len(values) else {}


def two_market_sources_consistent(cache: dict[str, Any]) -> bool:
    """Compare Tencent and Baostock on date and daily return, not price scale."""
    quote = next(
        (entry for entry in cache.values() if entry.get("tool") == "get_tencent_quote"),
        None,
    )
    history = next(
        (entry for entry in cache.values() if entry.get("tool") == "get_historical_k_data"),
        None,
    )
    if not quote or not history or not isinstance(quote.get("data"), dict):
        return False
    latest = _last_markdown_row(history.get("data"))
    try:
        quote_date = str(quote["data"].get("datetime") or "")[:10]
        history_date = str(latest.get("date") or "")[:10]
        quote_change = float(quote["data"]["change_pct"])
        history_change = float(latest["pctChg"])
    except (KeyError, TypeError, ValueError):
        return False
    return quote_date == history_date and abs(quote_change - history_change) <= 0.20


async def collect_common_target_data(state: WorkflowState) -> dict[str, Any]:
    """Execute the centralized plan once, then expose only its shared cache."""
    if not state.target_symbol or state.data.get("disable_common_collection"):
        return {
            "data": {
                "collection_cache": {},
                "collection_plan": [],
                "collection_report": {"completed": True, "skipped": True},
            }
        }

    plan = build_collection_plan(state)
    tools = await get_all_tools_for_orchestrator()
    by_name = {tool.name: tool for tool in tools}
    cache: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    persistent = PersistentToolCache.from_state(state)
    concurrency = max(1, int(state.data.get("collection_concurrency", 4)))
    semaphore = asyncio.Semaphore(concurrency)
    total_budget = int(state.data.get("collection_tool_call_budget", 27))
    category_limits = allocate_category_budget(
        total_budget, state.data.get("tool_budget_shares")
    )
    ledger = CategoryBudget(category_limits)
    ledger_lock = asyncio.Lock()
    control = get_run_call_control(
        state.run_id,
        circuit_threshold=int(state.data.get("provider_circuit_threshold", 2)),
    )

    async def reserve(category: str) -> bool:
        async with ledger_lock:
            return ledger.consume(category)

    async def execute_and_validate(
        tool: Any,
        plan_item: dict[str, Any],
        arguments: dict[str, Any],
        *,
        timeout_retry_remaining: bool = True,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        tool_name = plan_item["tool"]
        provider = plan_item["provider"]
        if control.provider_is_open(provider):
            return None, _failure(
                tool_name,
                arguments,
                "PROVIDER_CIRCUIT_OPEN",
                DataFetchStatus.UPSTREAM_ERROR,
                provider=provider,
            )
        validator = FinancialDataValidator()
        step = PlanStep(
            id=f"collection:{plan_item['id']}",
            objective=f"Collect {plan_item['requirement']} with {tool_name}",
            candidate_tools=[tool_name],
            arguments=arguments,
            agent_id="data_collector",
            target_symbol=state.target_symbol,
            scope=SymbolScope.TARGET,
        )
        # Serialize a provider's calls so the second consecutive transport failure
        # opens the circuit before later calls can reach that upstream.
        async with semaphore, control.provider_lock(provider):
            if control.provider_is_open(provider):
                return None, _failure(
                    tool_name,
                    arguments,
                    "PROVIDER_CIRCUIT_OPEN",
                    DataFetchStatus.UPSTREAM_ERROR,
                    provider=provider,
                )
            if not await reserve(plan_item["budget_category"]):
                return None, _failure(
                    tool_name,
                    arguments,
                    "CATEGORY_BUDGET_EXCEEDED",
                    DataFetchStatus.BUDGET_BLOCKED,
                    provider="data_collector",
                )
            result = await MCPToolExecutor(
                [tool], timeout_seconds=float(state.data.get("tool_timeout", 30.0))
            ).execute(step, state)
            report = await validator.validate(step, result, state)

        if not result.ok or not report.passed:
            error_code = (
                result.error.code
                if result.error
                else (report.issues[0].code if report.issues else "VALIDATION_FAILED")
            )
            error_message = (
                f"{result.error.message} {result.error.details}"
                if result.error
                else ""
            )
            control.record_provider_result(
                provider,
                ok=False,
                error_code=error_code,
                message=error_message,
            )
            combined_code = str(error_code).upper()
            if "TIMEOUT" in combined_code:
                status = DataFetchStatus.TIMEOUT
            elif "NO_DATA" in combined_code or "EMPTY" in combined_code:
                status = DataFetchStatus.NO_DATA
            elif any(
                token in combined_code
                for token in ("VALIDATION", "INVALID", "FILTER_BROKEN")
            ):
                status = DataFetchStatus.INVALID_DATA
            else:
                status = DataFetchStatus.UPSTREAM_ERROR
            failure = _failure(
                tool_name,
                arguments,
                str(error_code),
                status,
                provider=result.meta.provider or provider,
                call_id=result.meta.call_id,
            )
            failure["issues"] = [
                issue.model_dump(mode="json") for issue in report.issues
            ]
            if status == DataFetchStatus.TIMEOUT and timeout_retry_remaining:
                retry_entry, retry_failure = await execute_and_validate(
                    tool,
                    plan_item,
                    arguments,
                    timeout_retry_remaining=False,
                )
                if retry_failure is not None:
                    retry_failure["attempts"] = 2
                    retry_failure["retry_policy"] = "TIMEOUT_RETRIED_ONCE"
                elif retry_entry is not None:
                    retry_entry["attempts"] = 2
                    retry_entry["retry_policy"] = "TIMEOUT_RETRIED_ONCE"
                return retry_entry, retry_failure
            failure["attempts"] = 1
            failure["retry_policy"] = (
                "SWITCH_PROVIDER"
                if status in {DataFetchStatus.UPSTREAM_ERROR, DataFetchStatus.INVALID_DATA}
                else "NO_SAME_PARAMETER_RETRY"
            )
            return None, failure

        actual_provider = result.meta.provider or provider
        control.record_provider_result(provider, ok=True)
        return {
            "tool": tool_name,
            "arguments": arguments,
            "data": result.data,
            "call_id": result.meta.call_id,
            "request_hash": result.meta.request_hash,
            "raw_data_hash": result.meta.raw_data_hash,
            "provider": actual_provider,
            "upstream_meta": result.meta.upstream_meta,
            "quality_score": report.score,
            "record_count": estimate_record_count(
                result.data,
                tool_name,
                arguments,
            ),
        }, None

    async def collect_one(plan_item: dict[str, Any]):
        tool_name = plan_item["tool"]
        arguments = plan_item["arguments"]
        key = plan_item["cache_key"]

        if key in cache:
            return plan_item, cache[key], None, "run_cache"
        tool = by_name.get(tool_name)
        if tool is None:
            return plan_item, None, _failure(
                tool_name,
                arguments,
                "TOOL_UNAVAILABLE",
                DataFetchStatus.UPSTREAM_ERROR,
                provider="mcp_registry",
            ), "unavailable"
        lookup = persistent.lookup(
            tool_name,
            arguments,
            allow_stale=bool(state.data.get("stale_while_revalidate", True)),
        )
        if lookup.entry is not None:
            entry = dict(lookup.entry)
            entry.setdefault("tool", tool_name)
            entry.setdefault("arguments", arguments)
            typed = validate_tool_records(tool_name, entry.get("data"), arguments)
            # Never allow legacy caches created by count-only validation to
            # resurrect irrelevant news or malformed typed records.
            if tool_name != "get_financial_news" or typed.record_count > 0:
                entry["data"] = typed.data
                entry["record_count"] = typed.record_count
                entry["persistent_cache_status"] = lookup.status
                entry["cache_age_seconds"] = lookup.age_seconds
                return plan_item, entry, None, lookup.status

        prefix = persistent.find_historical_prefix(tool_name, arguments)
        if prefix is not None:
            cached_args, cached_entry = prefix
            next_day = (
                date.fromisoformat(cached_args["end_date"]) + timedelta(days=1)
            ).isoformat()
            delta_arguments = {**arguments, "start_date": next_day}
            delta_entry, failure = await execute_and_validate(
                tool, plan_item, delta_arguments
            )
            if delta_entry is not None:
                combined = dict(delta_entry)
                combined["arguments"] = arguments
                combined["data"] = merge_markdown_ohlcv(
                    cached_entry.get("data"), delta_entry.get("data")
                )
                combined["raw_data_hash"] = "sha256:" + hashlib.sha256(
                    json.dumps(
                        combined["data"], ensure_ascii=False, default=str
                    ).encode("utf-8")
                ).hexdigest()
                combined["record_count"] = estimate_record_count(
                    combined["data"],
                    tool_name,
                    arguments,
                )
                combined["incremental_from"] = cached_args["end_date"]
                persistent.store(tool_name, arguments, combined)
                return plan_item, combined, None, "incremental"
            if failure is not None:
                failure["incremental_from"] = cached_args["end_date"]

        entry, failure = await execute_and_validate(tool, plan_item, arguments)
        if entry is not None:
            persistent.store(tool_name, arguments, entry)
        return plan_item, entry, failure, "miss"

    # Priorities execute in order; aliases in later tiers reuse the run cache.
    for category in ("required", "cross_validation", "supplemental"):
        tier = [item for item in plan if item["budget_category"] == category]
        runnable: list[dict[str, Any]] = []
        for item in tier:
            if (
                item.get("condition") == "unless_two_consistent_market_sources"
                and two_market_sources_consistent(cache)
            ):
                item.update(
                    {
                        "status": "SKIPPED",
                        "code": "TWO_CONSISTENT_MARKET_SOURCES",
                        "cache_status": "not_needed",
                    }
                )
                continue
            runnable.append(item)

        batches: list[list[dict[str, Any]]] = [runnable]
        if category == "required":
            primary_market = [
                item for item in runnable if item["tool"] == "get_historical_k_data"
            ]
            remaining = [item for item in runnable if item not in primary_market]
            batches = [batch for batch in (primary_market, remaining) if batch]

        for batch in batches:
            results = await asyncio.gather(*(collect_one(item) for item in batch))
            for item, entry, failure, cache_status in results:
                item["cache_status"] = cache_status
                if failure is not None:
                    failures.append(failure)
                    item.update(
                        {
                            "status": failure["status"],
                            "code": failure["code"],
                            "call_id": failure["call_id"],
                            "attempts": failure.get("attempts", 1),
                            "retry_policy": failure.get("retry_policy"),
                        }
                    )
                if entry is not None:
                    cache[item["cache_key"]] = entry
                    item.update(
                        {
                            "status": DataFetchStatus.SUCCESS.value,
                            "call_id": entry.get("call_id"),
                            "record_count": entry.get("record_count", 0),
                            "attempts": entry.get("attempts", 1),
                            "retry_policy": entry.get("retry_policy"),
                        }
                    )

    historical_entry = next(
        (entry for entry in cache.values() if entry.get("tool") == "get_historical_k_data"),
        None,
    )
    technical_snapshot = calculate_technical_snapshot(
        historical_entry.get("data") if historical_entry else None
    )
    status_counts: dict[str, int] = {}
    for item in plan:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    return {
        "data": {
            "collection_cache": cache,
            "collection_plan": plan,
            "centralized_collection_enforced": True,
            "technical_snapshot": technical_snapshot,
            "collection_report": {
                "completed": True,
                "skipped": False,
                "cached_calls": len(cache),
                "planned_requirements": len(plan),
                "status_counts": status_counts,
                "concurrency": concurrency,
                "budget": ledger.snapshot(),
                "market_sources_consistent": two_market_sources_consistent(cache),
                "circuit_breakers": control.snapshot(),
                "fallback_policy": FALLBACK_POLICY,
                "cache_statuses": {
                    item["tool"]: item.get("cache_status", "not_run") for item in plan
                },
                "plan_cache_statuses": {
                    item["id"]: item.get("cache_status", "not_run") for item in plan
                },
                "background_refreshes": 0,
                "failures": failures,
            },
        }
    }


def _failure(
    tool_name: str,
    arguments: dict[str, Any],
    code: str,
    status: DataFetchStatus,
    *,
    provider: str,
    call_id: str | None = None,
) -> dict[str, Any]:
    return {
        "tool": tool_name,
        "call_id": call_id or uuid4().hex,
        "arguments": arguments,
        "parameters": arguments,
        "data_source": provider,
        "provider": provider,
        "status": status.value,
        "code": code,
        "error_code": code,
        "issues": [],
    }
