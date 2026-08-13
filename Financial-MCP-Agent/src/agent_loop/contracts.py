"""Stable contracts shared by every stage of the financial agent loop."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class LoopStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StopReason(str, Enum):
    COMPLETED = "COMPLETED"
    INVALID_TASK = "INVALID_TASK"
    REQUIRED_TOOL_UNAVAILABLE = "REQUIRED_TOOL_UNAVAILABLE"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    TOOL_FAILURE = "TOOL_FAILURE"
    MAX_ITERATIONS_EXCEEDED = "MAX_ITERATIONS_EXCEEDED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class IssueSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class ReplanAction(str, Enum):
    ADVANCE = "advance"
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    STOP = "stop"
    COMPLETE = "complete"


class SymbolScope(str, Enum):
    """Declared security scope for every audited tool call."""

    TARGET = "target"
    PEER = "peer"
    BENCHMARK = "benchmark"


class ToolError(StrictModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolProvenance(StrictModel):
    provider: str = "mcp"
    tool: str
    run_id: str
    trace_id: str
    call_id: str = Field(default_factory=lambda: uuid4().hex)
    requested_at: datetime
    received_at: datetime
    schema_version: str = "1.0"
    request_hash: str
    raw_data_hash: str
    upstream_meta: dict[str, Any] = Field(default_factory=dict)


class QualitySummary(StrictModel):
    status: str = "unchecked"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class ToolResultEnvelope(StrictModel):
    ok: bool
    data: Any = None
    error: ToolError | None = None
    meta: ToolProvenance
    quality: QualitySummary = Field(default_factory=QualitySummary)

    @model_validator(mode="after")
    def check_success_error_invariant(self) -> "ToolResultEnvelope":
        if self.ok and self.error is not None:
            raise ValueError("Successful tool results cannot contain an error.")
        if not self.ok and self.error is None:
            raise ValueError("Failed tool results must contain a structured error.")
        return self


class PlanStep(StrictModel):
    id: str
    objective: str
    candidate_tools: list[str] = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    agent_id: str | None = None
    target_symbol: str | None = None
    scope: SymbolScope = SymbolScope.TARGET
    allowed_symbols: list[str] = Field(default_factory=list)
    required: bool = True
    status: StepStatus = StepStatus.PENDING
    selected_tool_index: int = 0
    attempts: int = 0

    @property
    def selected_tool(self) -> str:
        return self.candidate_tools[self.selected_tool_index]

    @property
    def has_fallback(self) -> bool:
        return self.selected_tool_index + 1 < len(self.candidate_tools)


class ValidationIssue(StrictModel):
    code: str
    message: str
    severity: IssueSeverity = IssueSeverity.ERROR
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(StrictModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: list[ValidationIssue] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utc_now)

    @property
    def retryable(self) -> bool:
        errors = [issue for issue in self.issues if issue.severity == IssueSeverity.ERROR]
        return bool(errors) and all(issue.retryable for issue in errors)


class EvidenceRef(StrictModel):
    step_id: str
    objective: str
    tool: str
    call_id: str
    raw_data_hash: str
    retrieved_at: datetime
    quality_score: float


class ToolCallRecord(StrictModel):
    step_id: str
    tool: str
    call_id: str
    attempt: int
    arguments: dict[str, Any]
    started_at: datetime
    completed_at: datetime
    success: bool
    error_code: str | None = None


class ReplanDecision(StrictModel):
    action: ReplanAction
    reason: str
    stop_reason: StopReason | None = None


class TraceEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str
    trace_id: str
    sequence: int
    stage: str
    timestamp: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)
