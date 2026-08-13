"""Independent evidence/risk review between the quality gate and final decision."""

from __future__ import annotations

import json
import os
import time
from datetime import date
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.utils.execution_logger import get_execution_logger
from src.utils.logging_config import ERROR_ICON, SUCCESS_ICON, WAIT_ICON, setup_logger
from src.utils.quality_gate import build_analysis_result
from src.utils.report_renderer import compact_analysis_payload
from src.utils.retry_utils import ainvoke_with_retry
from src.utils.state_definition import AnalysisResult, AgentState, failed_analysis_update


load_dotenv(override=True)
logger = setup_logger(__name__)


def _deterministic_risk_audit(state: AgentState) -> dict[str, Any]:
    results = {
        key: value
        for key, value in state.analysis_results.items()
        if key in {"fundamental", "technical", "value", "event"}
    }
    inference_markers = ("推断", "假设", "预计", "可能", "估算", "大致")
    future_dated_calls: list[dict[str, Any]] = []
    scope_issues: list[dict[str, Any]] = []

    for analysis_type, result in results.items():
        for call in result.tool_calls:
            for issue in call.get("issues", []):
                if issue.get("code") in {
                    "SYMBOL_MISMATCH",
                    "SYMBOL_SCOPE_VIOLATION",
                    "PEER_SCOPE_FORBIDDEN",
                    "PEER_SYMBOL_NOT_ALLOWED",
                    "BENCHMARK_SYMBOL_NOT_ALLOWED",
                }:
                    scope_issues.append(
                        {"analysis_type": analysis_type, "issue": issue}
                    )
            for key, value in (call.get("arguments") or {}).items():
                if "date" not in str(key).lower() or not isinstance(value, str):
                    continue
                try:
                    parsed = date.fromisoformat(value[:10])
                except ValueError:
                    continue
                if parsed > state.as_of:
                    future_dated_calls.append(
                        {
                            "analysis_type": analysis_type,
                            "tool": call.get("tool"),
                            "field": key,
                            "value": value,
                        }
                    )

    metric_keys = {
        key: {str(metric).casefold() for metric in result.metrics}
        for key, result in results.items()
    }
    all_specialists_present = set(results) == {
        "fundamental",
        "technical",
        "value",
        "event",
    }
    metric_values: dict[str, list[dict[str, Any]]] = {}
    for analysis_type, result in results.items():
        for key, value in result.metrics.items():
            metric_values.setdefault(str(key).casefold(), []).append(
                {"analysis_type": analysis_type, "value": value}
            )
    metric_conflicts = {
        key: values
        for key, values in metric_values.items()
        if len({json.dumps(item["value"], sort_keys=True, default=str) for item in values}) > 1
    }
    return {
        "quality_gate": state.data.get("quality_gate"),
        "evidence_counts": {
            key: len(result.evidence) for key, result in results.items()
        },
        "inference_marker_counts": {
            key: sum(result.content.count(marker) for marker in inference_markers)
            for key, result in results.items()
        },
        "future_dated_calls": future_dated_calls,
        "symbol_scope_issues": scope_issues,
        "metric_conflicts": metric_conflicts,
        "evidence_gaps": {
            key: {
                "quality_passed": result.quality_passed,
                "quality_score": result.quality_score,
                "data_completeness": result.data_completeness,
                "warnings": result.warnings,
                "quality_issues": result.quality_issues,
            }
            for key, result in results.items()
        },
        "allowed_peer_symbols": list(state.allowed_symbols),
        "decision_permissions": {
            "rating": all_specialists_present
            and all(result.success for result in results.values()),
            "target_price": any(
                keys.intersection({"target_price", "target price", "目标价"})
                for keys in metric_keys.values()
            ),
            "probability": any(
                keys.intersection({"probability", "scenario_probability", "概率"})
                for keys in metric_keys.values()
            ),
        },
    }


def deterministic_risk_review_update(
    state: AgentState,
    *,
    reason: str,
) -> dict[str, Any]:
    """Return a conservative, evidence-only review when the review LLM fails."""
    current_data = dict(state.data)
    audit = _deterministic_risk_audit(state)
    gate = current_data.get("quality_gate") or {}
    degraded = gate.get("degraded_analysis_types") or []
    failed = gate.get("failed_analysis_types") or []
    coverage = float(gate.get("coverage") or 0.0)
    content = f"""# 自动风险审查（降级）

风险审查模型未能在时限内完成，系统已切换为确定性审查。原因：`{reason}`。

- 质量门状态：{gate.get('status', 'UNKNOWN')}；可用覆盖率：{coverage:.0%}。
- 降级模块：{', '.join(degraded) if degraded else '无'}。
- 失败模块：{', '.join(failed) if failed else '无'}。
- 日期越界调用：{len(audit.get('future_dated_calls') or [])} 个。
- 证券范围问题：{len(audit.get('symbol_scope_issues') or [])} 个。
- 多模块指标冲突：{len(audit.get('metric_conflicts') or {})} 组；所有冲突口径必须并列保留。

## 可证伪的下行情景

1. 后续定期报告显示盈利、现金流或偿债指标较当前证据恶化时，基本面结论应下调。
2. 后续行情跌破技术模块已识别的关键支撑且无法收复时，趋势结论应下调。
3. 后续出现经核验的负面公告、监管事项或目标公司新闻时，事件面结论应重新评估。

## 最终决策限制

- 不得填补缺失的新闻、同行、行业或财务数字。
- 不得从无效、无日期或未提及目标证券的记录推导事实。
- 不得对冲突数据自行选择口径；不得输出证据池没有支持的目标价或概率。
- 本审查为自动降级结果，最终结论必须降低置信度并显式披露该限制。
""".strip()
    evidence = [
        item for result in state.analysis_results.values() for item in result.evidence
    ]
    tool_calls = [
        item for result in state.analysis_results.values() for item in result.tool_calls
    ]
    result = AnalysisResult(
        agent_id="risk_review_agent",
        analysis_type="risk_review",
        content=content,
        success=True,
        symbol=state.symbol,
        company_name=state.company_name,
        as_of=state.as_of,
        confidence=0.35,
        evidence=evidence,
        tool_calls=tool_calls,
        warnings=[reason],
        quality_status="DEGRADED",
        quality_passed=True,
        quality_score=0.5,
        data_completeness=coverage,
    )
    current_data["risk_review"] = content
    current_data["risk_audit"] = audit
    current_data["risk_review_fallback"] = {
        "used": True,
        "reason": reason,
    }
    current_data.pop("risk_review_error", None)
    return {
        "data": current_data,
        "metadata": dict(state.metadata),
        "analysis_results": {"risk_review": result},
    }


async def risk_review_agent(state: AgentState) -> dict[str, Any]:
    """Challenge evidence, contradictions and downside scenarios without new tools."""
    agent_name = "risk_review_agent"
    execution_logger = get_execution_logger()
    started = time.time()
    current_data = dict(state.data)
    current_metadata = dict(state.metadata)
    execution_logger.log_agent_start(
        agent_name,
        {
            "stock_code": state.target_symbol,
            "analysis_types": list(state.analysis_results),
            "quality_gate": current_data.get("quality_gate"),
        },
    )
    logger.info(f"{WAIT_ICON} RiskReviewAgent: Starting evidence and downside review.")

    gate = current_data.get("quality_gate") or {}
    if not gate.get("passed"):
        error = "Risk review blocked because the deterministic quality gate did not pass."
        current_data["risk_review_error"] = error
        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - started, False, error
        )
        return failed_analysis_update(
            agent_id=agent_name,
            analysis_type="risk_review",
            data=current_data,
            messages=state.messages,
            metadata=current_metadata,
            error=error,
        )

    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")
    if not all([api_key, base_url, model_name]):
        error = "Missing OpenAI environment variables."
        current_data["risk_review_error"] = error
        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - started, False, error
        )
        return failed_analysis_update(
            agent_id=agent_name,
            analysis_type="risk_review",
            data=current_data,
            messages=state.messages,
            metadata=current_metadata,
            error=error,
        )

    specialist_payload = compact_analysis_payload(
        state.analysis_results, content_limit=1600
    )
    deterministic_audit = _deterministic_risk_audit(state)
    prompt = f"""你是风险审查 Agent，不负责为股票找利好，也不直接写投资报告。
目标证券：{state.company_name}（{state.target_symbol}）
分析基准日：{state.as_of.isoformat()}

日期越界、证券范围、证据缺口、指标口径冲突和决策权限已由确定性代码审计；
不要重复推导这些规则。请只审查代码难以判断的语义矛盾、因果跳跃和下行情景，
并输出简洁 Markdown 风险意见。必须逐项回答：
1. 哪些结论缺少直接证据，哪些只是推断；
2. 哪些数据或结论互相矛盾；
3. 是否使用了分析基准日之后的数据；
4. 是否把同行、基准或其他证券误当成目标证券；
5. 至少三个可证伪的下行情景，以及触发指标；
6. 当前证据是否足以分别给出评级、目标价和概率（逐项写“足够/不足”并说明理由）；
7. 给最终决策 Agent 的硬性限制，尤其是禁止编造缺失数字。

确定性审计结果：
{json.dumps(deterministic_audit, ensure_ascii=False, default=str)}

专业分析结果：
{json.dumps(specialist_payload, ensure_ascii=False, default=str)}
"""

    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            max_tokens=1600,
        )
        llm_started = time.time()
        message = await ainvoke_with_retry(
            llm,
            [
                {
                    "role": "system",
                    "content": "你是保守、证据优先的股票研究风险审查员。",
                },
                {"role": "user", "content": prompt},
            ],
            logger=logger,
            operation_name="RiskReviewAgent review generation",
            max_attempts=3,
        )
        content = str(message.content).strip()
        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="risk_review",
            input_messages=[{"role": "user", "content": prompt}],
            output_content=content,
            model_config={"model": model_name, "temperature": 0.1},
            execution_time=time.time() - llm_started,
        )
        evidence = [
            item
            for result in state.analysis_results.values()
            for item in result.evidence
        ]
        tool_calls = [
            item
            for result in state.analysis_results.values()
            for item in result.tool_calls
        ]
        result = build_analysis_result(
            agent_id=agent_name,
            analysis_type="risk_review",
            content=content,
            state=state,
            evidence=evidence,
            tool_calls=tool_calls,
        )
        current_data["risk_review"] = content
        current_data["risk_audit"] = deterministic_audit
        if not result.success:
            current_data["risk_review_error"] = result.error
        execution_logger.log_agent_complete(
            agent_name,
            {
                "review_length": len(content),
                "quality_score": result.quality_score,
                "deterministic_audit": deterministic_audit,
            },
            time.time() - started,
            result.success,
            result.error,
        )
        logger.info(
            "%s RiskReviewAgent: Review %s.",
            SUCCESS_ICON if result.success else ERROR_ICON,
            "completed" if result.success else "rejected",
        )
        return {
            "data": current_data,
            "metadata": current_metadata,
            "analysis_results": {"risk_review": result},
        }
    except Exception as error:
        message = f"Risk review generation failed: {error}"
        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - started, True, message
        )
        return deterministic_risk_review_update(state, reason=message)
