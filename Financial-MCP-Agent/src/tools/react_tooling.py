"""Validated, policy-scoped MCP tools for specialist ReAct agents."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from langchain_core.tools import StructuredTool, ToolException

from src.agent_loop.contracts import PlanStep, SymbolScope
from src.agent_loop.executor import MCPToolExecutor
from src.agent_loop.tracing import JSONLTraceRecorder
from src.agent_loop.validator import FinancialDataValidator
from src.utils.execution_logger import get_execution_logger
from src.utils.state_definition import (
    DataFetchStatus,
    WorkflowState,
    data_fetch_status_from_event,
)
from src.tools.data_collection import collection_cache_key, estimate_record_count
from src.tools.data_quality import validate_tool_records
from src.tools.call_control import (
    CategoryBudget,
    allocate_category_budget,
    get_run_call_control,
    provider_for_tool,
)
from src.tools.persistent_cache import CacheLookup, PersistentToolCache


_TRACE_RECORDERS: dict[tuple[str, str], JSONLTraceRecorder] = {}
_BACKGROUND_REFRESHES: set[asyncio.Task[Any]] = set()

DEFAULT_AGENT_TOOL_LIMITS = {
    "fundamental_agent": 14,
    "technical_agent": 6,
    "value_agent": 14,
    "event_agent": 6,
}

TOOL_BUDGET_CATEGORIES = {
    "get_stock_basic_info": "required",
    "get_stock_industry": "required",
    "get_latest_trading_date": "required",
    "get_historical_k_data": "required",
    "get_tencent_quote": "required",
    "get_profit_data": "required",
    "get_operation_data": "required",
    "get_growth_data": "required",
    "get_balance_data": "required",
    "get_cash_flow_data": "required",
    "get_official_announcements": "required",
    "get_financial_news": "required",
    "get_dupont_data": "cross_validation",
    "get_dividend_data": "cross_validation",
    "get_mootdx_bars": "cross_validation",
    "get_adjust_factor_data": "cross_validation",
    "get_eastmoney_signals": "supplemental",
}

YEAR_STRING_TOOLS = frozenset({
    "get_profit_data",
    "get_operation_data",
    "get_growth_data",
    "get_balance_data",
    "get_cash_flow_data",
    "get_dupont_data",
    "get_dividend_data",
})

BUSINESS_KIND_BY_TOOL = {
    "get_dividend_data": "dividend",
    "get_profit_data": "profit",
    "get_performance_express_report": "profit",
    "get_forecast_report": "profit",
    "get_tencent_quote": "quote",
    "get_stock_realtime_data": "quote",
    "get_realtime_stock_data": "quote",
}


def _latest_completed_quarter(as_of: Any) -> str:
    year, month = as_of.year, as_of.month
    quarter = (month - 1) // 3
    if quarter == 0:
        return f"{year - 1}Q4"
    return f"{year}Q{quarter}"


def business_key_for_tool(
    tool_name: str, arguments: dict[str, Any], state: WorkflowState
) -> str:
    """Build a stable business key without relying on an Agent's prose."""
    kind = BUSINESS_KIND_BY_TOOL.get(tool_name, tool_name.removeprefix("get_"))
    symbol = next(
        (
            str(arguments[key])
            for key in ("code", "symbol", "stock_code", "ts_code")
            if arguments.get(key)
        ),
        str(state.target_symbol or state.symbol or "unknown"),
    )
    if kind == "dividend":
        period = str(arguments.get("year") or state.as_of.year)
    elif kind == "profit":
        year = arguments.get("year")
        quarter = arguments.get("quarter")
        period = (
            f"{year}Q{quarter}"
            if year is not None and quarter is not None
            else str(year or _latest_completed_quarter(state.as_of))
        )
    elif kind == "quote":
        period = str(arguments.get("date") or state.as_of.isoformat())
    else:
        period = str(
            arguments.get("date")
            or arguments.get("end_date")
            or arguments.get("year")
            or state.as_of.isoformat()
        )
    return f"{kind}:{symbol}:{period}"


def _event_status(event: dict[str, Any]) -> str:
    return data_fetch_status_from_event(event).value


def normalize_tool_arguments(
    tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Normalize stable MCP contract quirks before validation and execution."""
    normalized = dict(arguments)
    if (
        tool_name in YEAR_STRING_TOOLS
        and normalized.get("year") is not None
        and not isinstance(normalized["year"], str)
    ):
        normalized["year"] = str(normalized["year"])
    return normalized


def select_runtime_tools(
    agent_name: str, state: WorkflowState, tools: Iterable[Any]
) -> list[Any]:
    """Remove optional upstream calls already made redundant by pre-collection."""
    selected = list(tools)
    if agent_name != "technical_agent":
        return selected

    cached_tools = {
        item.get("tool")
        for item in (state.data.get("collection_cache") or {}).values()
        if item.get("tool")
    }
    failed_common_tools = {
        item.get("tool")
        for item in (state.data.get("collection_report") or {}).get("failures", [])
        if item.get("tool")
    }
    suppressed: set[str] = set()
    if (state.data.get("collection_report") or {}).get(
        "market_sources_consistent"
    ) is True:
        suppressed.add("get_mootdx_bars")
    if "get_eastmoney_signals" in failed_common_tools:
        suppressed.add("get_eastmoney_signals")
    return [tool for tool in selected if getattr(tool, "name", None) not in suppressed]


def get_workflow_trace_recorder(state: WorkflowState) -> JSONLTraceRecorder:
    base_dir = str(state.data.get("trace_dir") or "logs/workflow")
    key = (str(Path(base_dir).resolve()), state.run_id)
    if key not in _TRACE_RECORDERS:
        _TRACE_RECORDERS[key] = JSONLTraceRecorder(base_dir)
    return _TRACE_RECORDERS[key]


def release_workflow_trace_recorder(state: WorkflowState) -> None:
    base_dir = str(state.data.get("trace_dir") or "logs/workflow")
    _TRACE_RECORDERS.pop((str(Path(base_dir).resolve()), state.run_id), None)


def serialize_react_messages(response: Any) -> list[dict[str, Any]]:
    """Serialize AI/tool messages without depending on a specific provider class."""
    if not isinstance(response, dict):
        return []
    serialized: list[dict[str, Any]] = []
    for message in response.get("messages", []):
        item = {
            "type": getattr(message, "type", type(message).__name__),
            "content": getattr(message, "content", str(message)),
        }
        for attribute in ("name", "tool_call_id", "status"):
            value = getattr(message, attribute, None)
            if value is not None:
                item[attribute] = value
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = tool_calls
        serialized.append(item)
    return serialized


class AuditedToolGateway:
    """Wrap a filtered tool set with validation, retry and append-only tracing."""

    def __init__(
        self,
        *,
        agent_name: str,
        state: WorkflowState,
        tools: Iterable[Any],
        timeout_seconds: float | None = None,
        trace_recorder: Any | None = None,
        execution_logger: Any | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.state = state
        self.raw_tools = list(tools)
        self.timeout_seconds = timeout_seconds or float(
            state.data.get("tool_timeout", 30.0)
        )
        self.validator = FinancialDataValidator()
        self.trace = trace_recorder or get_workflow_trace_recorder(state)
        self.execution_logger = execution_logger or get_execution_logger()
        self.events: list[dict[str, Any]] = []
        configured_limits = state.data.get("agent_tool_call_limits") or {}
        self.max_tool_calls = int(
            configured_limits.get(
                agent_name,
                state.data.get(
                    "max_tool_calls_per_agent",
                    DEFAULT_AGENT_TOOL_LIMITS.get(agent_name, 10),
                ),
            )
        )
        self.tool_call_count = 0
        self.category_budget = CategoryBudget(
            allocate_category_budget(
                self.max_tool_calls, state.data.get("tool_budget_shares")
            )
        )
        self.run_control = get_run_call_control(
            state.run_id,
            circuit_threshold=int(state.data.get("provider_circuit_threshold", 2)),
        )
        self.persistent_cache = PersistentToolCache.from_state(state)

    def wrap_tools(self) -> list[StructuredTool]:
        return [self._wrap(tool) for tool in self.raw_tools]

    def _wrap(self, tool: Any) -> StructuredTool:
        async def invoke_validated(**arguments: Any) -> Any:
            return await self._invoke(tool, arguments)

        args_schema = getattr(tool, "args_schema", None)
        if args_schema is None and hasattr(tool, "get_input_schema"):
            args_schema = tool.get_input_schema()
        return StructuredTool.from_function(
            coroutine=invoke_validated,
            name=tool.name,
            description=getattr(tool, "description", "") or tool.name,
            args_schema=args_schema,
            infer_schema=args_schema is None,
            handle_tool_error=True,
        )

    async def _invoke(self, tool: Any, arguments: dict[str, Any]) -> Any:
        arguments = normalize_tool_arguments(tool.name, arguments)
        if self.agent_name in self.run_control.blocked_agents:
            message = (
                f"{self.agent_name} is blocked after a prior BUDGET_BLOCKED event; "
                "no further tool calls are permitted."
            )
            event = self._blocked_event(
                tool.name,
                arguments,
                code="AGENT_BUDGET_CIRCUIT_OPEN",
                message=message,
            )
            self.events.append(event)
            self.trace.record(self.state, "react_tool_agent_blocked", event)
            raise ToolException(message)
        requested_symbols = self.validator._argument_symbols(arguments)
        target_symbol = self.state.target_symbol or self.state.symbol
        if not requested_symbols or requested_symbols == {target_symbol}:
            scope = SymbolScope.TARGET
            allowed_symbols: list[str] = []
        elif self.agent_name == "value_agent" and requested_symbols.issubset(
            set(self.state.allowed_symbols)
        ):
            scope = SymbolScope.PEER
            allowed_symbols = list(self.state.allowed_symbols)
        elif requested_symbols.issubset(set(self.state.benchmark_symbols)):
            scope = SymbolScope.BENCHMARK
            allowed_symbols = list(self.state.benchmark_symbols)
        else:
            # Preserve the attempted intent so request validation can fail closed
            # with a precise scope error before the external tool is called.
            scope = (
                SymbolScope.PEER
                if self.agent_name == "value_agent"
                else SymbolScope.TARGET
            )
            allowed_symbols = list(self.state.allowed_symbols)

        step = PlanStep(
            id=f"react:{self.agent_name}:{tool.name}:{uuid4().hex}",
            objective=f"{self.agent_name} ReAct tool call: {tool.name}",
            candidate_tools=[tool.name],
            arguments=arguments,
            agent_id=self.agent_name,
            target_symbol=target_symbol,
            scope=scope,
            allowed_symbols=allowed_symbols,
        )
        request_issues = self.validator.validate_request(step, self.state)
        if request_issues:
            call_id = uuid4().hex
            error_text = "; ".join(issue.message for issue in request_issues)
            event = {
                "agent_name": self.agent_name,
                "step_id": step.id,
                "tool": tool.name,
                "call_id": call_id,
                "attempt": 0,
                "arguments": arguments,
                "parameters": arguments,
                "target_symbol": target_symbol,
                "scope": scope.value,
                "allowed_symbols": allowed_symbols,
                "ok": False,
                "quality_score": 0.0,
                "request_hash": "",
                "raw_data_hash": "",
                "data_source": "gateway",
                "error_code": request_issues[0].code,
                "issues": [issue.model_dump(mode="json") for issue in request_issues],
            }
            event["status"] = _event_status(event)
            event["business_key"] = business_key_for_tool(
                tool.name, arguments, self.state
            )
            self.events.append(event)
            self.trace.record(self.state, "react_tool_rejected", event)
            self.execution_logger.log_tool_usage(
                self.agent_name,
                tool.name,
                arguments,
                {"issues": event["issues"]},
                0.0,
                success=False,
                error=error_text,
            )
            raise ToolException(f"{tool.name} rejected before execution: {error_text}")

        cached = (self.state.data.get("collection_cache") or {}).get(
            collection_cache_key(tool.name, arguments)
        )
        if cached is not None:
            event = {
                "agent_name": self.agent_name,
                "step_id": step.id,
                "tool": tool.name,
                "call_id": cached["call_id"],
                "attempt": 0,
                "arguments": arguments,
                "parameters": arguments,
                "target_symbol": target_symbol,
                "scope": scope.value,
                "allowed_symbols": allowed_symbols,
                "ok": True,
                "cache_hit": True,
                "quality_score": cached["quality_score"],
                "record_count": cached.get("record_count", 1),
                "request_hash": cached["request_hash"],
                "raw_data_hash": cached["raw_data_hash"],
                "provider": cached.get("provider", "mcp"),
                "data_source": cached.get("provider", "mcp"),
                "error_code": None,
                "upstream_meta": cached.get("upstream_meta", {}),
                "issues": [],
                "evidence_value": json_safe(cached.get("data")),
            }
            event["status"] = DataFetchStatus.SUCCESS.value
            event["business_key"] = business_key_for_tool(
                tool.name, arguments, self.state
            )
            self.events.append(event)
            self.trace.record(self.state, "react_tool_cache_hit", event)
            return cached["data"]

        if self.state.data.get("centralized_collection_enforced") and scope == SymbolScope.TARGET:
            planned = next(
                (
                    item
                    for item in (self.state.data.get("collection_plan") or [])
                    if item.get("cache_key")
                    == collection_cache_key(tool.name, arguments)
                ),
                None,
            )
            code = "CENTRAL_COLLECTION_MISS" if planned else "UNPLANNED_TOOL_CALL"
            message = (
                f"{tool.name} may not call upstream from {self.agent_name}; "
                "specialists must use the centralized collection cache."
            )
            planned_status = (planned or {}).get("status")
            if planned_status == "SKIPPED":
                rejection_status = DataFetchStatus.NOT_ATTEMPTED.value
            elif planned_status in {
                status.value for status in DataFetchStatus if status != DataFetchStatus.SUCCESS
            }:
                rejection_status = str(planned_status)
            else:
                rejection_status = DataFetchStatus.INVALID_DATA.value
            event = self._rejection_event(
                step=step,
                tool_name=tool.name,
                arguments=arguments,
                target_symbol=target_symbol,
                scope=scope,
                allowed_symbols=allowed_symbols,
                code=code,
                message=message,
                status=rejection_status,
            )
            self.events.append(event)
            self.trace.record(self.state, "react_tool_centralized_rejected", event)
            raise ToolException(message)

        persistent = self.persistent_cache.lookup(
            tool.name,
            arguments,
            allow_stale=bool(self.state.data.get("stale_while_revalidate", True)),
        )
        if persistent.entry is not None:
            typed_cache = validate_tool_records(
                tool.name,
                persistent.entry.get("data"),
                arguments,
            )
            if tool.name == "get_financial_news" and typed_cache.record_count == 0:
                persistent = CacheLookup(None, "invalid", persistent.age_seconds)
            else:
                persistent.entry["data"] = typed_cache.data
                persistent.entry["record_count"] = typed_cache.record_count
        if persistent.entry is not None:
            cached = persistent.entry
            event = {
                "agent_name": self.agent_name,
                "step_id": step.id,
                "tool": tool.name,
                "call_id": cached["call_id"],
                "attempt": 0,
                "arguments": arguments,
                "parameters": arguments,
                "target_symbol": target_symbol,
                "scope": scope.value,
                "allowed_symbols": allowed_symbols,
                "ok": True,
                "cache_hit": True,
                "cache_tier": "persistent",
                "cache_status": persistent.status,
                "cache_age_seconds": persistent.age_seconds,
                "quality_score": cached["quality_score"],
                "record_count": cached.get("record_count", 1),
                "request_hash": cached["request_hash"],
                "raw_data_hash": cached["raw_data_hash"],
                "provider": cached.get("provider", "mcp"),
                "data_source": cached.get("provider", "mcp"),
                "error_code": None,
                "upstream_meta": cached.get("upstream_meta", {}),
                "issues": [],
                "evidence_value": json_safe(cached.get("data")),
            }
            event["status"] = DataFetchStatus.SUCCESS.value
            event["business_key"] = business_key_for_tool(
                tool.name, arguments, self.state
            )
            self.events.append(event)
            self.trace.record(self.state, "react_tool_persistent_cache_hit", event)
            if (
                persistent.status == "stale"
                and not self.state.data.get("centralized_collection_enforced")
            ):
                task = asyncio.create_task(
                    self._refresh_persistent_entry(tool, step, arguments)
                )
                _BACKGROUND_REFRESHES.add(task)
                task.add_done_callback(_BACKGROUND_REFRESHES.discard)
            return cached["data"]

        # The budget bounds slow or costly upstream work. Run-scoped and persistent
        # cache hits above are deterministic local reads and must not consume it;
        # otherwise shared pre-collection can paradoxically starve required calls.
        expected_provider = provider_for_tool(tool.name)
        if self.run_control.provider_is_open(expected_provider):
            message = (
                f"{expected_provider} circuit is open after consecutive "
                "transport failures."
            )
            event = self._rejection_event(
                step=step,
                tool_name=tool.name,
                arguments=arguments,
                target_symbol=target_symbol,
                scope=scope,
                allowed_symbols=allowed_symbols,
                code="PROVIDER_CIRCUIT_OPEN",
                message=message,
                status=DataFetchStatus.UPSTREAM_ERROR.value,
            )
            self.events.append(event)
            self.trace.record(self.state, "react_tool_provider_blocked", event)
            raise ToolException(message)
        budget_category = self._budget_category(tool.name, arguments)
        if (
            self.tool_call_count >= self.max_tool_calls
            or not self.category_budget.consume(budget_category)
        ):
            message = (
                f"{self.agent_name} exceeded its {budget_category} category "
                f"reserve within the {self.max_tool_calls}-external-call budget."
            )
            event = self._blocked_event(
                tool.name,
                arguments,
                code="TOOL_CALL_BUDGET_EXCEEDED",
                message=message,
                step_id=step.id,
            )
            self.events.append(event)
            self.run_control.blocked_agents.add(self.agent_name)
            self.trace.record(self.state, "react_tool_budget_exceeded", event)
            raise ToolException(message)
        self.tool_call_count += 1

        executor = MCPToolExecutor([tool], timeout_seconds=self.timeout_seconds)
        max_attempts = self.state.max_retries_per_step + 1
        last_error = "Tool call failed validation."

        for attempt in range(1, max_attempts + 1):
            if self.run_control.provider_is_open(expected_provider):
                event = self._rejection_event(
                    step=step,
                    tool_name=tool.name,
                    arguments=arguments,
                    target_symbol=target_symbol,
                    scope=scope,
                    allowed_symbols=allowed_symbols,
                    code="PROVIDER_CIRCUIT_OPEN",
                    message=f"{expected_provider} circuit is open after consecutive transport failures.",
                    status=DataFetchStatus.UPSTREAM_ERROR.value,
                )
                self.events.append(event)
                self.trace.record(self.state, "react_tool_provider_blocked", event)
                raise ToolException(event["issues"][0]["message"])
            started = time.time()
            step.attempts = attempt
            self.trace.record(
                self.state,
                "react_tool_started",
                {
                    "agent_name": self.agent_name,
                    "step_id": step.id,
                    "tool": tool.name,
                    "attempt": attempt,
                    "arguments": arguments,
                },
            )
            async with self.run_control.provider_lock(expected_provider):
                if self.run_control.provider_is_open(expected_provider):
                    raise ToolException(
                        f"{expected_provider} circuit is open after consecutive transport failures."
                    )
                result = await executor.execute(step, self.state)
                report = await self.validator.validate(step, result, self.state)
            error = result.error.message if result.error else None
            if report.issues:
                last_error = "; ".join(issue.message for issue in report.issues)
            elif error:
                last_error = error

            event = {
                "agent_name": self.agent_name,
                "step_id": step.id,
                "tool": tool.name,
                "call_id": result.meta.call_id,
                "attempt": attempt,
                "arguments": arguments,
                "parameters": arguments,
                "target_symbol": target_symbol,
                "scope": scope.value,
                "allowed_symbols": allowed_symbols,
                "ok": result.ok and report.passed,
                "quality_score": report.score,
                "record_count": estimate_record_count(
                    result.data,
                    tool.name,
                    arguments,
                ),
                "request_hash": result.meta.request_hash,
                "raw_data_hash": result.meta.raw_data_hash,
                "provider": result.meta.provider,
                "data_source": result.meta.provider,
                "error_code": result.error.code if result.error else (
                    report.issues[0].code if report.issues else None
                ),
                "upstream_meta": result.meta.upstream_meta,
                "issues": [issue.model_dump(mode="json") for issue in report.issues],
            }
            event["status"] = _event_status(event)
            event["business_key"] = business_key_for_tool(
                tool.name, arguments, self.state
            )
            if event["status"] == DataFetchStatus.SUCCESS.value:
                event["evidence_value"] = json_safe(result.data)
            self.events.append(event)
            self.trace.record(self.state, "react_tool_completed", event)
            self.execution_logger.log_tool_usage(
                self.agent_name,
                tool.name,
                arguments,
                result.data if result.ok else result.error.model_dump(mode="json"),
                time.time() - started,
                success=bool(result.ok and report.passed),
                error=None if result.ok and report.passed else last_error,
            )

            self.run_control.record_provider_result(
                expected_provider,
                ok=bool(result.ok and report.passed),
                error_code=event.get("error_code"),
                message=(
                    f"{last_error} {result.error.details}"
                    if result.error
                    else last_error
                ),
            )

            if result.ok and report.passed:
                self.persistent_cache.store(
                    tool.name,
                    arguments,
                    {
                        "tool": tool.name,
                        "arguments": arguments,
                        "data": result.data,
                        "call_id": result.meta.call_id,
                        "request_hash": result.meta.request_hash,
                        "raw_data_hash": result.meta.raw_data_hash,
                        "provider": result.meta.provider,
                        "upstream_meta": result.meta.upstream_meta,
                        "quality_score": report.score,
                        "record_count": estimate_record_count(
                            result.data,
                            tool.name,
                            arguments,
                        ),
                    },
                )
                return result.data
            # Only timeouts may repeat identical parameters, and at most once.
            # NO_DATA, INVALID_DATA and upstream/proxy failures must switch source
            # (or degrade) instead of hammering the same endpoint.
            if (
                attempt < min(max_attempts, 2)
                and event["status"] == DataFetchStatus.TIMEOUT.value
            ):
                self.trace.record(
                    self.state,
                    "react_tool_retry",
                    {
                        "agent_name": self.agent_name,
                        "step_id": step.id,
                        "tool": tool.name,
                        "next_attempt": attempt + 1,
                        "reason": last_error,
                    },
                )
                continue
            break

        raise ToolException(
            f"{tool.name} failed after {step.attempts} attempt(s): {last_error}"
        )

    def _budget_category(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        cache_key = collection_cache_key(tool_name, arguments)
        planned = next(
            (
                item
                for item in (self.state.data.get("collection_plan") or [])
                if item.get("cache_key") == cache_key
            ),
            None,
        )
        if planned:
            return str(planned.get("budget_category") or "supplemental")
        return TOOL_BUDGET_CATEGORIES.get(tool_name, "supplemental")

    def _blocked_event(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        code: str,
        message: str,
        step_id: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "agent_name": self.agent_name,
            "step_id": step_id or f"react:{self.agent_name}:{tool_name}:{uuid4().hex}",
            "tool": tool_name,
            "call_id": uuid4().hex,
            "attempt": 0,
            "arguments": arguments,
            "parameters": arguments,
            "ok": False,
            "cache_hit": False,
            "data_source": "gateway",
            "error_code": code,
            "issues": [{"code": code, "message": message}],
            "status": DataFetchStatus.BUDGET_BLOCKED.value,
            "budget": self.category_budget.snapshot(),
        }
        event["business_key"] = business_key_for_tool(
            tool_name, arguments, self.state
        )
        return event

    def _rejection_event(
        self,
        *,
        step: PlanStep,
        tool_name: str,
        arguments: dict[str, Any],
        target_symbol: str | None,
        scope: SymbolScope,
        allowed_symbols: list[str],
        code: str,
        message: str,
        status: str,
    ) -> dict[str, Any]:
        event = {
            "agent_name": self.agent_name,
            "step_id": step.id,
            "tool": tool_name,
            "call_id": uuid4().hex,
            "attempt": 0,
            "arguments": arguments,
            "parameters": arguments,
            "target_symbol": target_symbol,
            "scope": scope.value,
            "allowed_symbols": allowed_symbols,
            "ok": False,
            "cache_hit": False,
            "quality_score": 0.0,
            "request_hash": "",
            "raw_data_hash": "",
            "data_source": "gateway",
            "error_code": code,
            "issues": [{"code": code, "message": message}],
            "status": status,
        }
        event["business_key"] = business_key_for_tool(
            tool_name, arguments, self.state
        )
        return event

    async def _refresh_persistent_entry(
        self, tool: Any, step: PlanStep, arguments: dict[str, Any]
    ) -> None:
        """Refresh a stale entry without extending the foreground critical path."""
        executor = MCPToolExecutor([tool], timeout_seconds=self.timeout_seconds)
        result = await executor.execute(step, self.state)
        report = await self.validator.validate(step, result, self.state)
        if result.ok and report.passed:
            self.persistent_cache.store(
                tool.name,
                arguments,
                {
                    "tool": tool.name,
                    "arguments": arguments,
                    "data": result.data,
                    "call_id": result.meta.call_id,
                    "request_hash": result.meta.request_hash,
                    "raw_data_hash": result.meta.raw_data_hash,
                    "provider": result.meta.provider,
                    "upstream_meta": result.meta.upstream_meta,
                    "quality_score": report.score,
                    "record_count": estimate_record_count(
                        result.data,
                        tool.name,
                        arguments,
                    ),
                },
            )

    @property
    def evidence(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": item["tool"],
                "call_id": item["call_id"],
                "arguments": item.get("arguments", {}),
                "business_key": item.get("business_key"),
                "status": item.get("status", DataFetchStatus.SUCCESS.value),
                "raw_data_hash": item["raw_data_hash"],
                "quality_score": item["quality_score"],
                "record_count": item.get("record_count", 0),
                "provider": item.get("provider", "mcp"),
                "data_source": item.get("data_source", item.get("provider", "mcp")),
                "error_code": item.get("error_code"),
                "evidence_value": item.get("evidence_value"),
                "source_chain": (item.get("upstream_meta") or {}).get(
                    "source_chain", []
                ),
                "source_failures": (item.get("upstream_meta") or {}).get(
                    "source_failures", []
                ),
            }
            for item in self.events
            if item["ok"]
        ]


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
