"""Audited tool-execution infrastructure used by the canonical workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent_loop.state import AgentLoopState

__all__ = ["AgentLoopState"]


def __getattr__(name: str) -> Any:
    """Load state exports lazily so contract imports cannot form a cycle."""
    if name == "AgentLoopState":
        from src.agent_loop.state import AgentLoopState

        return AgentLoopState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
