"""Deterministic financial-data validation; no numeric checks are delegated to an LLM."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from src.agent_loop.contracts import (
    IssueSeverity,
    PlanStep,
    ToolResultEnvelope,
    ValidationIssue,
    ValidationReport,
    SymbolScope,
)
from src.agent_loop.state import AgentLoopState
from src.tools.data_quality import validate_tool_records


class FinancialDataValidator:
    EMPTY_MARKERS = (
        "no data",
        "not found",
        "暂无数据",
        "无数据",
        "未找到",
        "查询结果为空",
    )

    async def validate(
        self,
        step: PlanStep,
        result: ToolResultEnvelope,
        state: AgentLoopState,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        issues.extend(self.validate_request(step, state))
        if not result.ok:
            error = result.error
            issues.append(
                ValidationIssue(
                    code=error.code if error else "TOOL_FAILED",
                    message=error.message if error else "Tool reported failure.",
                    retryable=error.retryable if error else False,
                    details=error.details if error else {},
                )
            )
        elif self._is_empty(result.data):
            issues.append(
                ValidationIssue(
                    code="EMPTY_DATA",
                    message="Tool returned no usable financial data.",
                    retryable=False,
                )
            )

        if result.ok and isinstance(result.data, str):
            lowered = result.data.lower()
            if any(marker in lowered for marker in self.EMPTY_MARKERS):
                issues.append(
                    ValidationIssue(
                        code="NO_DATA_MARKER",
                        message="Tool response indicates that no matching data was found.",
                        retryable=False,
                    )
                )

        if result.ok:
            typed = validate_tool_records(
                step.selected_tool,
                result.data,
                step.arguments,
            )
            result.data = typed.data
            if step.selected_tool == "get_financial_news":
                if typed.record_count == 0:
                    issues.append(
                        ValidationIssue(
                            code="INVALID_NEWS_DATA",
                            message=(
                                "News provider returned zero dated, target-relevant "
                                "records; transport success is not usable data."
                            ),
                            retryable=False,
                            details={
                                "rejected_count": typed.rejected_count,
                                "rejection_reasons": typed.rejection_reasons,
                            },
                        )
                    )
                elif typed.rejected_count:
                    issues.append(
                        ValidationIssue(
                            code="NEWS_RECORDS_REJECTED",
                            message="Irrelevant, undated, out-of-window, or duplicate news was removed.",
                            severity=IssueSeverity.WARNING,
                            retryable=False,
                            details={
                                "accepted_count": typed.record_count,
                                "rejected_count": typed.rejected_count,
                                "rejection_reasons": typed.rejection_reasons,
                            },
                        )
                    )
            elif typed.record_count == 0 and step.selected_tool in {
                "get_historical_k_data",
                "get_mootdx_bars",
                "get_tencent_quote",
            }:
                issues.append(
                    ValidationIssue(
                        code="INVALID_DATA_TYPE_RECORDS",
                        message=(
                            f"{step.selected_tool} returned no records satisfying its "
                            "type-specific schema."
                        ),
                        retryable=False,
                        details={"rejected_count": typed.rejected_count},
                    )
                )
            elif typed.record_count == 0 and step.selected_tool in {
                "get_official_announcements",
                "get_eastmoney_signals",
            }:
                issues.append(
                    ValidationIssue(
                        code="NO_DATA",
                        message=f"{step.selected_tool} returned no usable records.",
                        retryable=False,
                    )
                )
            issues.extend(self._validate_ohlc(result.data))
            issues.extend(self._validate_symbol(result.data, step, state))
            issues.extend(self._validate_as_of_dates(result.data, state.as_of))

        error_count = sum(issue.severity == IssueSeverity.ERROR for issue in issues)
        warning_count = sum(issue.severity == IssueSeverity.WARNING for issue in issues)
        score = max(0.0, 1.0 - error_count * 0.5 - warning_count * 0.1)
        passed = error_count == 0
        result.quality.status = "valid" if passed else "invalid"
        result.quality.score = score
        result.quality.warnings = [
            issue.message for issue in issues if issue.severity == IssueSeverity.WARNING
        ]
        return ValidationReport(passed=passed, score=score, issues=issues)

    def validate_request(
        self,
        step: PlanStep,
        state: AgentLoopState,
    ) -> list[ValidationIssue]:
        """Reject unauthorized security queries before an MCP call is executed."""
        requested = self._argument_symbols(step.arguments)
        target = step.target_symbol or state.target_symbol or state.symbol
        allowed = set(step.allowed_symbols)

        if step.scope == SymbolScope.TARGET:
            unauthorized = requested - ({target} if target else set())
            if unauthorized:
                return [
                    ValidationIssue(
                        code="SYMBOL_SCOPE_VIOLATION",
                        message="Target-scoped tool call requested a different security.",
                        retryable=False,
                        details={
                            "target_symbol": target,
                            "scope": step.scope.value,
                            "requested": sorted(requested),
                        },
                    )
                ]
        elif step.scope == SymbolScope.PEER:
            if step.agent_id != "value_agent":
                return [
                    ValidationIssue(
                        code="PEER_SCOPE_FORBIDDEN",
                        message="Only the valuation agent may query peer securities.",
                        retryable=False,
                        details={"agent_id": step.agent_id},
                    )
                ]
            unauthorized = requested - allowed
            if not requested or unauthorized:
                return [
                    ValidationIssue(
                        code="PEER_SYMBOL_NOT_ALLOWED",
                        message="Peer query must use an explicitly allowed security.",
                        retryable=False,
                        details={
                            "requested": sorted(requested),
                            "allowed_symbols": sorted(allowed),
                        },
                    )
                ]
        elif step.scope == SymbolScope.BENCHMARK:
            unauthorized = requested - allowed
            if not requested or unauthorized:
                return [
                    ValidationIssue(
                        code="BENCHMARK_SYMBOL_NOT_ALLOWED",
                        message="Benchmark query must use an explicitly allowed symbol.",
                        retryable=False,
                        details={
                            "requested": sorted(requested),
                            "allowed_symbols": sorted(allowed),
                        },
                    )
                ]
        return []

    @staticmethod
    def _is_empty(value: Any) -> bool:
        return value is None or value == "" or value == [] or value == {}

    def _validate_ohlc(self, value: Any) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for index, row in enumerate(self._dict_rows(value)):
            keys = {str(key).lower(): key for key in row}
            if not {"open", "high", "low", "close"} <= keys.keys():
                continue
            try:
                open_price = float(row[keys["open"]])
                high = float(row[keys["high"]])
                low = float(row[keys["low"]])
                close = float(row[keys["close"]])
            except (TypeError, ValueError):
                issues.append(
                    ValidationIssue(
                        code="INVALID_OHLC_TYPE",
                        message="OHLC fields must be numeric.",
                        details={"row": index},
                    )
                )
                continue
            if low > high or not (low <= open_price <= high) or not (low <= close <= high):
                issues.append(
                    ValidationIssue(
                        code="INVALID_OHLC_RELATION",
                        message="OHLC values violate low <= open/close <= high.",
                        details={"row": index},
                    )
                )
        return issues

    def _validate_as_of_dates(
        self,
        value: Any,
        as_of: date,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if isinstance(value, str) and len(value.strip()) >= 10:
            try:
                scalar_date = date.fromisoformat(value.strip()[:10])
            except ValueError:
                scalar_date = None
            if scalar_date and scalar_date > as_of:
                issues.append(
                    ValidationIssue(
                        code="FUTURE_DATED_DATA",
                        message="Tool returned data after the analysis as-of date.",
                        retryable=False,
                        details={
                            "field": "scalar",
                            "value": value.strip(),
                            "as_of": as_of.isoformat(),
                        },
                    )
                )
        seen: set[tuple[int, str, str]] = set()
        for index, row in enumerate(self._dict_rows(value)):
            for key, raw in row.items():
                normalized_key = str(key).lower()
                if "date" not in normalized_key and normalized_key not in {
                    "time",
                    "datetime",
                }:
                    continue
                if not isinstance(raw, str) or len(raw) < 10:
                    continue
                try:
                    observed = date.fromisoformat(raw[:10])
                except ValueError:
                    continue
                marker = (index, str(key), raw)
                if observed > as_of and marker not in seen:
                    seen.add(marker)
                    issues.append(
                        ValidationIssue(
                            code="FUTURE_DATED_DATA",
                            message="Tool returned data after the analysis as-of date.",
                            retryable=False,
                            details={
                                "row": index,
                                "field": str(key),
                                "value": raw,
                                "as_of": as_of.isoformat(),
                            },
                        )
                    )
        return issues

    def _validate_symbol(
        self,
        value: Any,
        step: PlanStep,
        state: AgentLoopState,
    ) -> list[ValidationIssue]:
        requested = self._argument_symbols(step.arguments)
        if step.scope == SymbolScope.TARGET:
            expected = {
                step.target_symbol or state.target_symbol or state.symbol
            } - {None}
        else:
            expected = requested
        if not expected:
            return []
        observed = {
            str(row[key])
            for row in self._dict_rows(value)
            for key in row
            if str(key).lower() in {"code", "symbol"} and row[key]
        }
        if observed and not observed.issubset(expected):
            return [
                ValidationIssue(
                    code="SYMBOL_MISMATCH",
                    message="Returned data belongs to a different security.",
                    details={
                        "expected": sorted(expected),
                        "observed": sorted(observed),
                        "scope": step.scope.value,
                    },
                )
            ]
        return []

    @classmethod
    def _argument_symbols(cls, value: Any) -> set[str]:
        symbols: set[str] = set()
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).lower() in {"code", "symbol", "codes", "symbols"}:
                    symbols.update(cls._symbol_values(nested))
                elif isinstance(nested, (dict, list, tuple, set)):
                    symbols.update(cls._argument_symbols(nested))
        return symbols

    @staticmethod
    def _symbol_values(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value} if value else set()
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value if item}
        return set()

    @classmethod
    def _dict_rows(cls, value: Any) -> Iterable[dict]:
        if isinstance(value, dict):
            yield value
            for nested in value.values():
                yield from cls._dict_rows(nested)
        elif isinstance(value, list):
            for item in value:
                yield from cls._dict_rows(item)
        elif isinstance(value, str):
            yield from cls._markdown_rows(value)

    @staticmethod
    def _markdown_rows(value: str) -> Iterable[dict[str, str]]:
        """Parse the simple Markdown tables returned by the current MCP server."""
        lines = [line.strip() for line in value.splitlines() if line.strip().startswith("|")]
        if len(lines) < 3:
            return

        def cells(line: str) -> list[str]:
            return [cell.strip() for cell in line.strip().strip("|").split("|")]

        headers = cells(lines[0])
        separators = cells(lines[1])
        if len(headers) != len(separators) or not all(
            part.replace(":", "").replace("-", "") == "" and "-" in part
            for part in separators
        ):
            return
        for line in lines[2:]:
            values = cells(line)
            if len(values) == len(headers):
                yield dict(zip(headers, values))
