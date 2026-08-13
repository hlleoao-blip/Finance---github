"""Append-only JSONL traces and final state snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent_loop.contracts import TraceEvent
from src.agent_loop.state import AgentLoopState


class JSONLTraceRecorder:
    def __init__(self, base_dir: str | Path = "logs/agent_loop") -> None:
        self.base_dir = Path(base_dir)
        self._sequence = 0
        self._run_dir: Path | None = None
        self._run_id: str | None = None

    def _ensure_run_dir(self, state: AgentLoopState) -> Path:
        if self._run_dir is None or self._run_id != state.run_id:
            self._run_dir = self.base_dir / state.run_id
            self._run_id = state.run_id
            self._sequence = 0
            self._run_dir.mkdir(parents=True, exist_ok=True)
        return self._run_dir

    def record(self, state: AgentLoopState, stage: str, payload: dict) -> None:
        trace_path = self._ensure_run_dir(state) / "trace.jsonl"
        self._sequence += 1
        event = TraceEvent(
            run_id=state.run_id,
            trace_id=state.trace_id,
            sequence=self._sequence,
            stage=stage,
            payload=self._json_safe(payload),
        )
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")

    def save_state(self, state: AgentLoopState) -> None:
        state_path = self._ensure_run_dir(state) / "final_state.json"
        state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _json_safe(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class InMemoryTraceRecorder:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.final_state: dict | None = None

    def record(self, state: AgentLoopState, stage: str, payload: dict) -> None:
        self.events.append({"stage": stage, "payload": payload})

    def save_state(self, state: AgentLoopState) -> None:
        self.final_state = state.model_dump(mode="json")
