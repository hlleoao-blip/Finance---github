"""Compatibility import for the unified workflow state."""

from src.utils.state_definition import WorkflowState


# Existing callers keep working while both execution paths now share one model.
AgentLoopState = WorkflowState

__all__ = ["AgentLoopState", "WorkflowState"]
