"""Unified state and result contracts for the financial workflow."""

from __future__ import annotations

import operator
from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agent_loop.contracts import (
    EvidenceRef,
    LoopStatus,
    PlanStep,
    StopReason,
    ToolCallRecord,
    ToolResultEnvelope,
    ValidationReport,
)


def merge_dicts(d1: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
    """Merge parallel LangGraph branch updates without sharing mutable dicts."""
    return {**d1, **d2}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DataFetchStatus(str, Enum):
    """Canonical outcome for every requested financial-data dependency."""

    SUCCESS = "SUCCESS"
    NO_DATA = "NO_DATA"
    BUDGET_BLOCKED = "BUDGET_BLOCKED"
    TIMEOUT = "TIMEOUT"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    INVALID_DATA = "INVALID_DATA"
    SCOPE_BLOCKED = "SCOPE_BLOCKED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


def data_fetch_status_from_event(event: dict[str, Any]) -> DataFetchStatus:
    """Translate legacy gateway events into the closed fetch-status vocabulary."""
    explicit = event.get("status")
    if explicit:
        try:
            return DataFetchStatus(str(explicit))
        except ValueError:
            pass
    if event.get("ok") is True:
        return (
            DataFetchStatus.SUCCESS
            if int(event.get("record_count", 1) or 0) > 0
            else DataFetchStatus.NO_DATA
        )
    codes = [str(event.get("error_code") or "")]
    codes.extend(
        str(issue.get("code") or "")
        for issue in (event.get("issues") or [])
        if isinstance(issue, dict)
    )
    combined = " ".join(codes).upper()
    if "BUDGET" in combined:
        return DataFetchStatus.BUDGET_BLOCKED
    if "TIMEOUT" in combined or "TIMED_OUT" in combined:
        return DataFetchStatus.TIMEOUT
    if "SCOPE" in combined or "SYMBOL_NOT_ALLOWED" in combined:
        return DataFetchStatus.SCOPE_BLOCKED
    if "NO_DATA" in combined or "EMPTY" in combined or "NOT_FOUND" in combined:
        return DataFetchStatus.NO_DATA
    if any(
        token in combined
        for token in ("VALIDATION", "INVALID", "SCHEMA", "QUALITY", "FILTER_BROKEN")
    ):
        return DataFetchStatus.INVALID_DATA
    return DataFetchStatus.UPSTREAM_ERROR


class DataRequestRecord(BaseModel):
    """Auditable record for a call, a blocked attempt, or an intentional omission."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    requirement: str
    status: DataFetchStatus
    call_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    data_source: str = "unknown"
    error_code: str | None = None
    agent_id: str | None = None
    business_key: str | None = None
    scope: str | None = None
    value: Any = None
    details: dict[str, Any] = Field(default_factory=dict)


class EvidenceRegistryEntry(BaseModel):
    """All observations and failures retained for one deduplicated business key."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    business_key: str
    status: DataFetchStatus
    records: list[DataRequestRecord] = Field(default_factory=list)
    values: list[dict[str, Any]] = Field(default_factory=list)
    conflict: bool = False


class EvidenceRegistry(BaseModel):
    """Cross-agent, business-keyed evidence view consumed by final synthesis."""

    model_config = ConfigDict(extra="forbid")

    entries: dict[str, EvidenceRegistryEntry] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=utc_now)


class AnalysisResult(BaseModel):
    """Stable hand-off contract produced by every specialist agent."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    agent_id: str
    analysis_type: str
    content: str
    success: bool = True
    error: str | None = None
    symbol: str | None = None
    company_name: str | None = None
    as_of: date | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    claims: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    data_requests: list[DataRequestRecord] = Field(default_factory=list)
    quality_status: str = "FAIL"
    quality_passed: bool = False
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    data_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_issues: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def populate_data_requests(self) -> "AnalysisResult":
        """Keep older producers compatible while exposing structured outcomes."""
        if self.data_requests:
            return self
        for event in self.tool_calls:
            issues = event.get("issues") or []
            error_code = event.get("error_code") or next(
                (
                    issue.get("code")
                    for issue in issues
                    if isinstance(issue, dict) and issue.get("code")
                ),
                None,
            )
            self.data_requests.append(
                DataRequestRecord(
                    requirement=str(event.get("tool") or "unknown"),
                    status=data_fetch_status_from_event(event),
                    call_id=event.get("call_id"),
                    parameters=dict(
                        event.get("parameters") or event.get("arguments") or {}
                    ),
                    data_source=str(
                        event.get("data_source")
                        or event.get("provider")
                        or "unknown"
                    ),
                    error_code=str(error_code) if error_code else None,
                    agent_id=str(event.get("agent_name") or self.agent_id),
                    business_key=event.get("business_key"),
                    scope=event.get("scope"),
                    value=event.get("evidence_value"),
                    details={
                        "step_id": event.get("step_id"),
                        "cache_hit": bool(event.get("cache_hit", False)),
                        "issues": issues,
                    },
                )
            )
        return self


class WorkflowState(BaseModel):
    """One Pydantic state shared by LangGraph and bounded tool execution.

    The ``data``/``messages``/``metadata`` fields preserve compatibility with the
    original LangGraph nodes.  The remaining fields replace ``AgentLoopState`` so
    orchestration, validation and tracing use the same run and trace identifiers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    task: str = ""
    symbol: str | None = None
    target_symbol: str | None = None
    allowed_symbols: list[str] = Field(default_factory=list)
    benchmark_symbols: list[str] = Field(default_factory=list)
    company_name: str | None = None
    as_of: date = Field(default_factory=date.today)
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)

    messages: Annotated[list[Any], operator.add] = Field(default_factory=list)
    data: Annotated[dict[str, Any], merge_dicts] = Field(default_factory=dict)
    metadata: Annotated[dict[str, Any], merge_dicts] = Field(default_factory=dict)
    analysis_results: Annotated[dict[str, AnalysisResult], merge_dicts] = Field(
        default_factory=dict
    )

    plan: list[PlanStep] = Field(default_factory=list)
    current_step_index: int = 0
    iteration_count: int = 0
    max_iterations: int = Field(default=20, ge=1, le=100)
    max_retries_per_step: int = Field(default=1, ge=0, le=10)
    tool_results: dict[str, ToolResultEnvelope] = Field(default_factory=dict)
    validation_reports: dict[str, ValidationReport] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    call_history: list[ToolCallRecord] = Field(default_factory=list)
    status: LoopStatus = LoopStatus.CREATED
    stop_reason: StopReason | None = None
    stop_detail: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    evidence_registry: EvidenceRegistry = Field(default_factory=EvidenceRegistry)

    @model_validator(mode="after")
    def synchronize_legacy_fields(self) -> "WorkflowState":
        if not self.task:
            self.task = str(self.data.get("query") or "")
        if not self.symbol:
            self.symbol = self.data.get("stock_code")
        if not self.target_symbol:
            self.target_symbol = self.symbol or self.data.get("target_symbol")
        if not self.allowed_symbols:
            self.allowed_symbols = list(self.data.get("allowed_symbols") or [])
        if not self.benchmark_symbols:
            self.benchmark_symbols = list(self.data.get("benchmark_symbols") or [])
        if not self.company_name:
            self.company_name = self.data.get("company_name")
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Provide the mapping-style access used by the original agent nodes."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Preserve legacy read-only mapping access used by the original CLI."""
        return getattr(self, key)

    @property
    def terminal(self) -> bool:
        return self.status in {LoopStatus.COMPLETED, LoopStatus.FAILED}

    @property
    def current_step(self) -> PlanStep | None:
        if self.current_step_index >= len(self.plan):
            return None
        return self.plan[self.current_step_index]

    def finish(self, reason: StopReason, detail: str) -> None:
        self.stop_reason = reason
        self.stop_detail = detail
        self.completed_at = utc_now()
        self.status = (
            LoopStatus.COMPLETED if reason == StopReason.COMPLETED else LoopStatus.FAILED
        )


# Compatibility alias for imports in existing agent modules and integrations.
AgentState = WorkflowState


def failed_analysis_update(
    *,
    agent_id: str,
    analysis_type: str,
    data: dict[str, Any],
    messages: list[Any],
    metadata: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    """Build a schema-compliant branch update for every failure exit."""
    content = str(data.get(f"{analysis_type}_analysis") or "")
    result = AnalysisResult(
        agent_id=agent_id,
        analysis_type=analysis_type,
        content=content or f"{analysis_type} analysis failed: {error}",
        success=False,
        error=error,
        symbol=data.get("stock_code"),
        company_name=data.get("company_name"),
        as_of=data.get("current_date"),
        confidence=0.0,
    )
    return {
        "data": data,
        "messages": [],
        "metadata": metadata,
        "analysis_results": {analysis_type: result},
    }
