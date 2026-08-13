"""
异步调用重试工具。

用于处理 OpenAI 兼容接口偶发的连接中断、超时和 5xx 错误。
"""
import asyncio
from typing import Any


def is_transient_api_error(error: Exception) -> bool:
    """判断异常是否适合重试，避免对明确的参数错误反复请求"""
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        return status_code == 429 or status_code >= 500

    error_type = type(error).__name__.lower()
    retryable_type_keywords = [
        "apiconnectionerror",
        "apitimeouterror",
        "connecterror",
        "connectionerror",
        "readtimeout",
        "timeout",
        "remoteprotocolerror",
    ]
    if any(keyword in error_type for keyword in retryable_type_keywords):
        return True

    error_text = str(error).lower()
    retryable_text_keywords = [
        "connection error",
        "connection reset",
        "connection aborted",
        "connection closed",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "rate limit",
        "too many requests",
        "server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
    ]
    return any(keyword in error_text for keyword in retryable_text_keywords)


async def ainvoke_with_retry(
    runnable: Any,
    input_data: Any,
    *,
    logger,
    operation_name: str,
    max_attempts: int = 5,
    initial_delay: float = 2.0,
    max_delay: float = 30.0,
    config: dict | None = None,
) -> Any:
    """对 LangChain/LangGraph 的 ainvoke 做指数退避重试"""
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await runnable.ainvoke(input_data, config=config)
        except Exception as error:
            last_error = error
            should_retry = is_transient_api_error(error)

            if not should_retry or attempt >= max_attempts:
                logger.error(
                    f"{operation_name} failed on attempt {attempt}/{max_attempts}: {error}"
                )
                raise

            delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                f"{operation_name} transient failure on attempt "
                f"{attempt}/{max_attempts}: {error}. Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

    raise last_error
