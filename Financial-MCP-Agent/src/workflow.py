"""Canonical LangGraph workflow used by every CLI/API entry point."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from langgraph.graph import END, StateGraph

from src.agents.fundamental_agent import fundamental_agent
from src.agents.news_agent import event_agent
from src.agents.summary_agent import decision_agent, deterministic_decision_update
from src.agents.risk_review_agent import (
    deterministic_risk_review_update,
    risk_review_agent,
)
from src.agents.technical_agent import technical_agent
from src.agents.value_agent import value_agent
from src.tools.react_tooling import (
    get_workflow_trace_recorder,
    release_workflow_trace_recorder,
)
from src.agent_loop.contracts import StopReason
from src.utils.state_definition import WorkflowState
from src.utils.execution_logger import get_execution_logger
from src.utils.quality_gate import run_quality_gate
from src.utils.report_renderer import reconcile_evidence
from src.tools.data_collection import collect_common_target_data
from src.tools.call_control import release_run_call_control


ANALYSIS_NODES = {
    "fundamental": ("fundamental_analyst", fundamental_agent),
    "technical": ("technical_analyst", technical_agent),
    "value": ("value_analyst", value_agent),
    "event": ("event_analyst", event_agent),
}

ANALYSIS_TYPE_ALIASES = {"news": "event"}

DEFAULT_AGENT_TIMEOUTS = {
    "fundamental_analyst": 180.0,
    "technical_analyst": 180.0,
    "value_analyst": 180.0,
    "event_analyst": 120.0,
    "risk_reviewer": 90.0,
    "decision_maker": 90.0,
}


def _agent_timeout(state: WorkflowState, node_name: str) -> float:
    configured = state.data.get("agent_timeout_seconds")
    if isinstance(configured, dict):
        return float(configured.get(node_name, DEFAULT_AGENT_TIMEOUTS[node_name]))
    if configured is not None:
        return float(configured)
    return DEFAULT_AGENT_TIMEOUTS[node_name]


async def _run_budgeted_node(
    state: WorkflowState,
    node: object,
    node_name: str,
    analysis_type: str | None = None,
) -> dict:
    """Apply a whole-Agent deadline and send timeouts through the quality gate."""
    normalized = (
        state if isinstance(state, WorkflowState) else WorkflowState.model_validate(state)
    )
    timeout = _agent_timeout(normalized, node_name)
    try:
        return await asyncio.wait_for(node(normalized), timeout=timeout)
    except asyncio.TimeoutError:
        error = f"AGENT_TIMEOUT: {node_name} exceeded {timeout:.1f}s; data is incomplete."
        data = dict(normalized.data)
        degradations = list(data.get("latency_degradations") or [])
        degradations.append(
            {
                "node": node_name,
                "code": "AGENT_TIMEOUT",
                "timeout_seconds": timeout,
                "data_incomplete": True,
            }
        )
        data["latency_degradations"] = degradations
        fallback_state = normalized.model_copy(deep=True)
        fallback_state.data = data
        if node_name == "risk_reviewer":
            get_execution_logger().log_agent_complete(
                "risk_review_agent",
                {"fallback": "deterministic", "reason": error},
                timeout,
                True,
            )
            return deterministic_risk_review_update(fallback_state, reason=error)
        if node_name == "decision_maker":
            get_execution_logger().log_agent_complete(
                "decision_agent",
                {"fallback": "deterministic", "reason": error},
                timeout,
                True,
            )
            return deterministic_decision_update(fallback_state, reason=error)
        if analysis_type is None:
            error_key = "decision_error" if node_name == "decision_maker" else f"{node_name}_error"
            data[error_key] = error
            return {"data": data}
        data[f"{analysis_type}_analysis_error"] = error
        from src.utils.state_definition import failed_analysis_update

        return failed_analysis_update(
            agent_id=node_name,
            analysis_type=analysis_type,
            data=data,
            messages=normalized.messages,
            metadata=normalized.metadata,
            error=error,
        )


def _budgeted(node: object, node_name: str, analysis_type: str | None = None):
    async def run(state: WorkflowState) -> dict:
        return await _run_budgeted_node(state, node, node_name, analysis_type)

    return run


def evidence_reconciliation_node(state: WorkflowState) -> dict:
    """Create the cross-Agent evidence view before risk and decision synthesis."""
    normalized = WorkflowState.model_validate(state)
    registry = reconcile_evidence(
        normalized.analysis_results,
        current_data=normalized.data,
        symbol=str(normalized.symbol or normalized.data.get("stock_code") or ""),
        as_of=normalized.data.get("current_date") or normalized.as_of,
    )
    data = dict(normalized.data)
    data["evidence_registry"] = registry.model_dump(mode="json")
    return {"data": data, "evidence_registry": registry}


def build_financial_workflow(
    analysis_types: Iterable[str] | None = None,
    *,
    include_summary: bool = True,
):
    """Build the single supported orchestration graph.

    ``include_summary=False`` is the canonical data-only path: specialist ReAct
    results, evidence and tool traces are returned without generating a report.
    """
    selected = tuple(
        ANALYSIS_TYPE_ALIASES.get(item, item)
        for item in (analysis_types or ANALYSIS_NODES)
    )
    unknown = set(selected) - set(ANALYSIS_NODES)
    if unknown:
        raise ValueError(f"Unknown analysis types: {sorted(unknown)}")
    if not selected:
        raise ValueError("At least one analysis type is required.")
    if include_summary and set(selected) != set(ANALYSIS_NODES):
        raise ValueError(
            "Risk review and final decision require all four specialist analyses."
        )

    graph = StateGraph(WorkflowState)
    graph.add_node("start", lambda _state: {})
    graph.add_node("data_collection", collect_common_target_data)
    graph.add_node("analysis_join", lambda _state: {})
    graph.add_node(
        "quality_gate",
        lambda state: run_quality_gate(
            WorkflowState.model_validate(state), selected
        ),
    )
    graph.set_entry_point("start")
    graph.add_edge("start", "data_collection")

    for analysis_type in selected:
        node_name, node = ANALYSIS_NODES[analysis_type]
        graph.add_node(node_name, _budgeted(node, node_name, analysis_type))
        graph.add_edge("data_collection", node_name)
        graph.add_edge(node_name, "analysis_join")

    if include_summary:
        graph.add_node("evidence_reconciliation", evidence_reconciliation_node)
        graph.add_node(
            "risk_reviewer",
            _budgeted(risk_review_agent, "risk_reviewer", "risk_review"),
        )
        graph.add_node(
            "decision_maker",
            _budgeted(decision_agent, "decision_maker"),
        )
        graph.add_edge("analysis_join", "quality_gate")
        graph.add_conditional_edges(
            "quality_gate",
            lambda state: (
                "review"
                if (state.get("data", {}).get("quality_gate") or {}).get("passed")
                else "stop"
            ),
            {"review": "evidence_reconciliation", "stop": END},
        )
        graph.add_edge("evidence_reconciliation", "risk_reviewer")
        graph.add_conditional_edges(
            "risk_reviewer",
            lambda state: (
                "decide"
                if (
                    state.get("analysis_results", {}).get("risk_review")
                    and state.get("analysis_results", {}).get("risk_review").success
                )
                else "stop"
            ),
            {"decide": "decision_maker", "stop": END},
        )
        graph.add_edge("decision_maker", END)
    else:
        graph.add_edge("analysis_join", "quality_gate")
        graph.add_edge("quality_gate", END)
    return graph.compile()


async def run_financial_workflow(
    state: WorkflowState,
    analysis_types: Iterable[str] | None = None,
    *,
    data_only: bool = False,
) -> WorkflowState:
    """Execute the canonical graph and persist a unified final state snapshot."""
    selected = tuple(
        ANALYSIS_TYPE_ALIASES.get(item, item)
        for item in (analysis_types or ANALYSIS_NODES)
    )
    if not data_only and len(selected) > 1 and set(selected) != set(ANALYSIS_NODES):
        raise ValueError(
            "A final decision requires fundamental, technical, value and event analyses. "
            "Use data_only=True for a partial specialist run."
        )
    include_summary = not data_only and set(selected) == set(ANALYSIS_NODES)
    state.data["requested_analysis_types"] = list(selected)
    trace = get_workflow_trace_recorder(state)
    trace.record(
        state,
        "workflow_started",
        {
            "analysis_types": list(selected),
            "data_only": data_only,
            "task": state.task,
            "symbol": state.symbol,
        },
    )
    app = build_financial_workflow(selected, include_summary=include_summary)
    try:
        raw_result = await app.ainvoke(state)
    except Exception as error:
        state.finish(StopReason.TOOL_FAILURE, str(error))
        trace.record(
            state,
            "workflow_failed",
            {"error": str(error), "exception_type": type(error).__name__},
        )
        trace.save_state(state)
        release_workflow_trace_recorder(state)
        release_run_call_control(state.run_id)
        raise
    result = WorkflowState.model_validate(raw_result)
    gate = result.data.get("quality_gate") or {}
    risk_result = result.analysis_results.get("risk_review")
    workflow_passed = bool(gate.get("passed")) and (
        data_only
        or not include_summary
        or bool(
            risk_result
            and risk_result.success
            and result.data.get("final_report")
            and result.data.get("decision_status") == "completed"
        )
    )
    if workflow_passed:
        result.finish(StopReason.COMPLETED, "Evidence-gated financial workflow completed.")
    else:
        result.finish(
            StopReason.INSUFFICIENT_DATA,
            "Quality gate, risk review, or final decision did not pass; no report was produced.",
        )
    result.output = {
        "run_id": result.run_id,
        "trace_id": result.trace_id,
        "data_only": data_only,
        "analysis_results": {
            key: value.model_dump(mode="json")
            for key, value in result.analysis_results.items()
        },
        "collection_plan": result.data.get("collection_plan"),
        "collection_report": result.data.get("collection_report"),
        "evidence_registry": result.data.get("evidence_registry"),
        "quality_gate": gate,
        "risk_review": result.data.get("risk_review"),
        "risk_review_error": result.data.get("risk_review_error"),
        "final_report": result.data.get("final_report"),
        "report_path": result.data.get("report_path"),
        "decision_error": result.data.get("decision_error"),
    }
    trace.record(
        result,
        "workflow_finished",
        {
            "analysis_types": list(result.analysis_results),
            "data_only": data_only,
            "report_generated": bool(result.data.get("final_report")),
        },
    )
    trace.save_state(result)
    release_workflow_trace_recorder(result)
    release_run_call_control(result.run_id)
    return result
