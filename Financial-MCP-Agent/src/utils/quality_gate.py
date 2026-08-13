"""Deterministic quality checks between specialist analysis and LLM decisions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.utils.state_definition import AnalysisResult, WorkflowState


PLACEHOLDER_MARKERS = (
    "sorry, need more steps",
    "need more steps to process",
    "no analysis generated",
    "not available",
    "analysis failed",
    "分析过程中出现错误",
    "无法完成分析",
    "暂无分析",
)

REQUIRED_TOOL_GROUPS: dict[str, tuple[frozenset[str], ...]] = {
    "fundamental": (
        frozenset({"get_stock_basic_info", "get_stock_industry"}),
        frozenset(
            {
                "get_profit_data",
                "get_operation_data",
                "get_growth_data",
                "get_balance_data",
                "get_cash_flow_data",
                "get_dupont_data",
                "get_performance_express_report",
                "get_forecast_report",
            }
        ),
    ),
    "technical": (
        frozenset({"get_historical_k_data", "get_mootdx_bars"}),
    ),
    "value": (
        frozenset({"get_historical_k_data", "get_mootdx_bars"}),
        frozenset(
            {
                "get_profit_data",
                "get_growth_data",
                "get_balance_data",
                "get_cash_flow_data",
                "get_dupont_data",
            }
        ),
    ),
    "event": (
        frozenset({"get_official_announcements"}),
        frozenset({"get_financial_news"}),
    ),
}

QUALITY_PASS = "PASS"
QUALITY_DEGRADED = "DEGRADED"
QUALITY_FAIL = "FAIL"
MINIMUM_USABLE_COMPLETENESS = 0.5
MINIMUM_WORKFLOW_COVERAGE = 0.75


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def is_substantive_content(content: str, *, minimum_length: int = 80) -> bool:
    text = (content or "").strip()
    lowered = text.casefold()
    return len(text) >= minimum_length and not any(
        marker in lowered for marker in PLACEHOLDER_MARKERS
    )


def _successful_tools(result: AnalysisResult) -> set[str]:
    return {
        str(call.get("tool"))
        for call in result.tool_calls
        if call.get("ok") is True and call.get("tool")
    }


def _valid_evidence(result: AnalysisResult) -> list[dict[str, Any]]:
    return [
        item
        for item in result.evidence
        if item.get("call_id")
        and item.get("raw_data_hash")
        and float(item.get("quality_score") or 0.0) > 0.0
    ]


def evaluate_analysis_result(
    result: AnalysisResult,
    *,
    allowed_symbols: Iterable[str] = (),
) -> AnalysisResult:
    """Mutate and return one result after deterministic minimum-quality checks."""
    issues: list[dict[str, Any]] = []
    content = (result.content or "").strip()
    lowered = content.casefold()

    if len(content) < 80:
        issues.append(
            _issue(
                "INSUBSTANTIAL_CONTENT",
                "Analysis content is too short to contain a substantive conclusion.",
                length=len(content),
                minimum=80,
            )
        )
    markers = [marker for marker in PLACEHOLDER_MARKERS if marker in lowered]
    if markers:
        issues.append(
            _issue(
                "INCOMPLETE_PLACEHOLDER",
                "Analysis contains an unfinished or unavailable-data placeholder.",
                markers=markers,
            )
        )

    evidence = _valid_evidence(result)
    if not evidence:
        issues.append(
            _issue(
                "NO_VALID_EVIDENCE",
                "At least one validated evidence reference is required.",
            )
        )

    if result.analysis_type == "event":
        event_record_count = sum(
            int(item.get("record_count") or 0) for item in evidence
        )
        if event_record_count < 5:
            issues.append(
                _issue(
                    "EVENT_RECORDS_INCOMPLETE",
                    "Event analysis requires at least five validated source records.",
                    record_count=event_record_count,
                    minimum=5,
                )
            )

    successful_tools = _successful_tools(result)
    groups = REQUIRED_TOOL_GROUPS.get(result.analysis_type, ())
    satisfied = sum(bool(group & successful_tools) for group in groups)
    completeness_checks = len(groups)

    if groups and satisfied < len(groups):
        missing_groups = [
            sorted(group)
            for group in groups
            if not group.intersection(successful_tools)
        ]
        issues.append(
            _issue(
                "MINIMUM_DATA_INCOMPLETE",
                "Required data categories did not reach the minimum completeness threshold.",
                missing_tool_groups=missing_groups,
            )
        )

    allowed = set(allowed_symbols)
    if result.analysis_type == "value" and allowed:
        completeness_checks += 1
        peer_calls = [
            call
            for call in result.tool_calls
            if call.get("ok") is True and call.get("scope") == "peer"
        ]
        if not peer_calls:
            issues.append(
                _issue(
                    "PEER_DATA_MISSING",
                    "Explicit peers were configured but no validated peer evidence was collected.",
                    allowed_symbols=sorted(allowed),
                )
            )
        else:
            satisfied += 1

    completeness = (
        satisfied / completeness_checks
        if completeness_checks
        else (1.0 if evidence else 0.0)
    )

    hard_issues = [
        issue for issue in issues if issue["code"] != "MINIMUM_DATA_INCOMPLETE"
    ]
    upstream_failed = bool(
        not result.success
        and result.error
        and result.error
        not in {
            "MINIMUM_DATA_INCOMPLETE",
            "EVENT_RECORDS_INCOMPLETE; MINIMUM_DATA_INCOMPLETE",
        }
    )
    if hard_issues or upstream_failed:
        quality_status = QUALITY_FAIL
    elif issues and evidence and completeness >= MINIMUM_USABLE_COMPLETENESS:
        quality_status = QUALITY_DEGRADED
    elif issues:
        quality_status = QUALITY_FAIL
    else:
        quality_status = QUALITY_PASS

    result.data_completeness = completeness
    result.quality_issues = issues
    result.quality_status = quality_status
    result.quality_passed = quality_status != QUALITY_FAIL
    result.success = result.quality_passed
    result.quality_score = max(
        0.0,
        min(
            1.0,
            0.35 * min(1.0, len(content) / 400)
            + 0.25 * (1.0 if evidence else 0.0)
            + 0.40 * completeness,
        ),
    )
    if result.success and result.error in {
        "MINIMUM_DATA_INCOMPLETE",
        "EVENT_RECORDS_INCOMPLETE; MINIMUM_DATA_INCOMPLETE",
    }:
        result.error = None
    if not result.success and result.error is None:
        result.error = "; ".join(issue["code"] for issue in issues) or "QUALITY_GATE_FAILED"
    return result


def build_analysis_result(
    *,
    agent_id: str,
    analysis_type: str,
    content: str,
    state: WorkflowState,
    evidence: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> AnalysisResult:
    """Create the common specialist result and immediately apply strict quality rules."""
    result = AnalysisResult(
        agent_id=agent_id,
        analysis_type=analysis_type,
        content=content,
        symbol=state.symbol or state.data.get("stock_code"),
        company_name=state.company_name or state.data.get("company_name"),
        as_of=state.as_of,
        confidence=0.7 if evidence else 0.0,
        evidence=evidence,
        tool_calls=tool_calls,
    )
    return evaluate_analysis_result(result, allowed_symbols=state.allowed_symbols)


def run_quality_gate(
    state: WorkflowState,
    required_analysis_types: Iterable[str],
) -> dict[str, Any]:
    """Re-check all required branches and produce an auditable gate report."""
    required = list(required_analysis_types)
    checked: dict[str, AnalysisResult] = {}
    failures: dict[str, list[dict[str, Any]]] = {}
    degradations: dict[str, list[dict[str, Any]]] = {}

    for analysis_type in required:
        result = state.analysis_results.get(analysis_type)
        if result is None:
            failures[analysis_type] = [
                _issue("MISSING_ANALYSIS", "Required specialist result is missing.")
            ]
            continue
        checked[analysis_type] = evaluate_analysis_result(
            result.model_copy(deep=True),
            allowed_symbols=state.allowed_symbols,
        )
        if checked[analysis_type].quality_status == QUALITY_FAIL:
            failures[analysis_type] = checked[analysis_type].quality_issues
        elif checked[analysis_type].quality_status == QUALITY_DEGRADED:
            degradations[analysis_type] = checked[analysis_type].quality_issues

    passed_types = [
        key
        for key, result in checked.items()
        if result.quality_status == QUALITY_PASS
    ]
    usable_types = [
        key
        for key, result in checked.items()
        if result.quality_status != QUALITY_FAIL
    ]
    coverage = len(usable_types) / len(required) if required else 0.0
    minimum_coverage = 1.0 if len(required) <= 1 else MINIMUM_WORKFLOW_COVERAGE
    all_full = len(passed_types) == len(required)
    can_continue = coverage >= minimum_coverage
    gate_status = (
        QUALITY_PASS
        if all_full
        else QUALITY_DEGRADED
        if can_continue
        else QUALITY_FAIL
    )

    report = {
        # ``passed`` is retained as the graph's continuation flag. Consumers that
        # require full coverage should use ``fully_passed`` or ``status``.
        "passed": gate_status != QUALITY_FAIL,
        "fully_passed": gate_status == QUALITY_PASS,
        "status": gate_status,
        "coverage": coverage,
        "minimum_coverage": minimum_coverage,
        "required_analysis_types": required,
        "passed_analysis_types": passed_types,
        "usable_analysis_types": usable_types,
        "degraded_analysis_types": sorted(degradations),
        "failed_analysis_types": sorted(failures),
        "degradations": degradations,
        "failures": failures,
    }
    return {
        "data": {"quality_gate": report},
        "analysis_results": checked,
    }
