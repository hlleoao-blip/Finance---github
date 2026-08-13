"""MCP execution adapter and result normalization."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

from src.agent_loop.contracts import (
    PlanStep,
    QualitySummary,
    ToolError,
    ToolProvenance,
    ToolResultEnvelope,
)
from src.agent_loop.state import AgentLoopState


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(value: Any) -> str:
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


class MCPToolExecutor:
    def __init__(self, tools: Iterable[Any], *, timeout_seconds: float = 30.0) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self.timeout_seconds = timeout_seconds

    @property
    def available_tools(self) -> frozenset[str]:
        return frozenset(self._tools)

    async def execute(
        self, step: PlanStep, state: AgentLoopState
    ) -> ToolResultEnvelope:
        tool_name = step.selected_tool
        requested_at = _utc_now()
        call_id = uuid4().hex
        raw: Any = None
        normalized_error: ToolError | None = None

        tool = self._tools.get(tool_name)
        if tool is None:
            normalized_error = ToolError(
                code="TOOL_UNAVAILABLE",
                message=f"MCP tool is not available: {tool_name}",
                retryable=False,
            )
        elif any(value is None for value in step.arguments.values()):
            normalized_error = ToolError(
                code="MISSING_ARGUMENT",
                message="Required tool arguments could not be derived from the task.",
                retryable=False,
                details={
                    "fields": [key for key, value in step.arguments.items() if value is None]
                },
            )
        else:
            try:
                raw = await asyncio.wait_for(
                    self._invoke(tool, step.arguments), timeout=self.timeout_seconds
                )
            except TimeoutError:
                normalized_error = ToolError(
                    code="TOOL_TIMEOUT",
                    message=f"Tool exceeded {self.timeout_seconds:.1f}s timeout.",
                    retryable=True,
                )
            except Exception as error:  # external adapter boundary
                normalized_error = ToolError(
                    code="TOOL_EXECUTION_ERROR",
                    message="Tool invocation raised an exception.",
                    retryable=True,
                    details={"exception_type": type(error).__name__},
                )

        received_at = _utc_now()
        ok, data, upstream_error, upstream_meta = self._normalize_raw(raw)
        if normalized_error is not None:
            ok, data, upstream_error = False, None, normalized_error

        provenance = ToolProvenance(
            provider=str(upstream_meta.get("provider", "mcp")),
            tool=tool_name,
            run_id=state.run_id,
            trace_id=state.trace_id,
            call_id=call_id,
            requested_at=requested_at,
            received_at=received_at,
            request_hash=_sha256({"tool": tool_name, "arguments": step.arguments}),
            raw_data_hash=_sha256(raw),
            upstream_meta=upstream_meta,
        )
        return ToolResultEnvelope(
            ok=ok,
            data=data,
            error=upstream_error,
            meta=provenance,
            quality=QualitySummary(status="unchecked", score=0.0),
        )

    @staticmethod
    async def _invoke(tool: Any, arguments: dict[str, Any]) -> Any:
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(arguments)
        if hasattr(tool, "invoke"):
            result = tool.invoke(arguments)
            if inspect.isawaitable(result):
                return await result
            return result
        if callable(tool):
            result = tool(**arguments)
            if inspect.isawaitable(result):
                return await result
            return result
        raise TypeError("Tool must expose ainvoke(), invoke(), or be callable.")

    @staticmethod
    def _normalize_raw(raw: Any) -> tuple[bool, Any, ToolError | None, dict[str, Any]]:
        if hasattr(raw, "content") and not isinstance(raw, (dict, str, bytes)):
            raw = raw.content
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return True, raw, None, {}
        if isinstance(raw, dict) and {"ok", "data", "error", "meta"} <= raw.keys():
            data = raw.get("data")
            if isinstance(data, dict) and set(data) == {"content"}:
                data = data["content"]
            error_payload = raw.get("error")
            error = ToolError.model_validate(error_payload) if error_payload else None
            ok = bool(raw.get("ok"))
            if not ok and error is None:
                error = ToolError(
                    code="MALFORMED_UPSTREAM_ERROR",
                    message="Upstream tool failed without a structured error payload.",
                )
            return ok, data, error, dict(raw.get("meta") or {})
        return True, raw, None, {}
