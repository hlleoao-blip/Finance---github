"""Extract ReAct output and recover from LangGraph step-limit placeholders."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.utils.retry_utils import ainvoke_with_retry


_STEP_LIMIT_MARKERS = (
    "sorry, need more steps",
    "sorry i need more steps",
    "need more steps to process",
)

DEFAULT_AGENT_ITERATIONS = {
    "fundamental_agent": 14,
    "technical_agent": 12,
    "value_agent": 14,
    "event_agent": 8,
}


def react_iteration_limit(state: Any, agent_name: str) -> int:
    configured = state.data.get("agent_iteration_limits") or {}
    default = DEFAULT_AGENT_ITERATIONS.get(agent_name, state.max_iterations)
    return max(2, min(state.max_iterations, int(configured.get(agent_name, default))))


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _extract_output(response: Any) -> tuple[str, list[Any], AIMessage | None]:
    if not isinstance(response, dict) or not isinstance(response.get("messages"), list):
        return "No analysis generated.", [], None

    messages = response["messages"]
    ai_messages = [message for message in messages if isinstance(message, AIMessage)]
    if ai_messages:
        last_ai_message = ai_messages[-1]
        return _message_text(last_ai_message), messages, last_ai_message

    all_content = [_message_text(message) for message in messages]
    combined = "\n".join(item for item in all_content if item)
    return combined or "No analysis generated.", messages, None


def is_step_limit_placeholder(content: str) -> bool:
    """Return whether content is LangGraph's graceful recursion-limit response."""
    text = (content or "").strip().casefold()
    return len(text) < 200 and any(marker in text for marker in _STEP_LIMIT_MARKERS)


async def extract_react_output(
    response: Any,
    *,
    llm: Any,
    logger: Any,
    operation_name: str,
    analysis_name: str,
) -> str:
    """Return final ReAct text, forcing synthesis if the tool loop ran out of steps.

    ``create_react_agent`` replaces a model response that still contains tool calls
    near the recursion limit with a short placeholder.  The tool results collected
    before that point remain useful, so make one tool-free model call over the same
    completed history and require an evidence-bounded conclusion.
    """
    output, messages, placeholder_message = _extract_output(response)
    if not is_step_limit_placeholder(output):
        return output

    logger.warning(
        "%s reached the ReAct step limit; forcing a tool-free synthesis from "
        "the collected evidence.",
        operation_name,
    )
    history = [message for message in messages if message is not placeholder_message]
    synthesis_request = HumanMessage(
        content=(
            f"工具调用阶段已经结束。请立即完成{analysis_name}，不得再请求调用任何工具，"
            "不得假设或补写工具结果中不存在的数据。只使用上述已经返回且可核验的工具结果，"
            "给出包含关键数据、趋势判断、数据局限、风险和结论的完整中文报告。"
            "若某个报告期或指标缺失，请明确披露缺口并基于最近可用期间完成分析，"
            "不要回复需要更多步骤或等待更多数据。"
        )
    )

    try:
        recovered_message = await ainvoke_with_retry(
            llm,
            [*history, synthesis_request],
            logger=logger,
            operation_name=f"{operation_name} forced synthesis",
        )
    except Exception as error:
        logger.error(
            "%s forced synthesis failed: %s",
            operation_name,
            error,
            exc_info=True,
        )
        return output

    recovered = _message_text(recovered_message)
    if not recovered or is_step_limit_placeholder(recovered):
        logger.error("%s forced synthesis returned no usable analysis.", operation_name)
        return output

    logger.info(
        "%s recovered from the ReAct step limit with %d characters of analysis.",
        operation_name,
        len(recovered),
    )
    return recovered
