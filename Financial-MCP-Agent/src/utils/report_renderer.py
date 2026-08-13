"""Compact decision payloads and deterministic Markdown report rendering."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from src.utils.state_definition import (
    DataFetchStatus,
    DataRequestRecord,
    EvidenceRegistry,
    EvidenceRegistryEntry,
)


ANALYSIS_LABELS = {
    "fundamental": "基本面分析",
    "technical": "技术分析",
    "value": "估值分析",
    "event": "事件分析",
}

PUBLIC_SOURCE_LABELS = {
    "get_tencent_quote": "腾讯行情",
    "get_official_announcements": "巨潮资讯、深交所公告",
    "get_financial_news": "财联社、新浪财经新闻",
}

PUBLIC_LIMIT_LABELS = {
    "get_financial_news": "新闻与舆情",
    "get_eastmoney_signals": "资金流向",
}

BUSINESS_KIND_BY_TOOL = {
    "get_dividend_data": "dividend",
    "get_profit_data": "profit",
    "get_performance_express_report": "profit",
    "get_forecast_report": "profit",
    "get_tencent_quote": "quote",
    "get_stock_realtime_data": "quote",
    "get_realtime_stock_data": "quote",
}

STATUS_PRECEDENCE = (
    DataFetchStatus.SUCCESS,
    DataFetchStatus.INVALID_DATA,
    DataFetchStatus.NO_DATA,
    DataFetchStatus.TIMEOUT,
    DataFetchStatus.UPSTREAM_ERROR,
    DataFetchStatus.SCOPE_BLOCKED,
    DataFetchStatus.BUDGET_BLOCKED,
    DataFetchStatus.NOT_ATTEMPTED,
)


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return date.today()


def _latest_completed_quarter(value: Any) -> str:
    as_of = _as_date(value)
    quarter = (as_of.month - 1) // 3
    return f"{as_of.year - 1}Q4" if quarter == 0 else f"{as_of.year}Q{quarter}"


def _business_key(
    tool: str, parameters: dict[str, Any], symbol: str, as_of: Any
) -> str:
    kind = BUSINESS_KIND_BY_TOOL.get(tool, tool.removeprefix("get_"))
    target = next(
        (
            str(parameters[key])
            for key in ("code", "symbol", "stock_code", "ts_code")
            if parameters.get(key)
        ),
        symbol or "unknown",
    )
    if kind == "dividend":
        period = str(parameters.get("year") or _as_date(as_of).year)
    elif kind == "profit":
        year, quarter = parameters.get("year"), parameters.get("quarter")
        period = (
            f"{year}Q{quarter}"
            if year is not None and quarter is not None
            else str(year or _latest_completed_quarter(as_of))
        )
    elif kind == "quote":
        period = str(parameters.get("date") or _as_date(as_of).isoformat())
    else:
        period = str(
            parameters.get("date")
            or parameters.get("end_date")
            or parameters.get("year")
            or _as_date(as_of).isoformat()
        )
    return f"{kind}:{target}:{period}"


def _record_identity(record: DataRequestRecord) -> str:
    return json.dumps(
        {
            "call_id": record.call_id,
            "requirement": record.requirement,
            "status": record.status.value,
            "parameters": record.parameters,
            "agent_id": record.agent_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def reconcile_evidence(
    results: dict[str, Any],
    *,
    current_data: dict[str, Any] | None = None,
    symbol: str = "",
    as_of: Any = None,
) -> EvidenceRegistry:
    """Merge all Agent and pre-collection outcomes before narrative synthesis."""
    current_data = current_data or {}
    as_of = as_of or current_data.get("current_date") or date.today()
    records: list[DataRequestRecord] = []

    for result in results.values():
        for item in list(getattr(result, "data_requests", []) or []):
            record = (
                item.model_copy(deep=True)
                if isinstance(item, DataRequestRecord)
                else DataRequestRecord.model_validate(item)
            )
            record.business_key = record.business_key or _business_key(
                record.requirement, record.parameters, symbol, as_of
            )
            records.append(record)

    for item in (current_data.get("collection_cache") or {}).values():
        tool = str(item.get("tool") or "unknown")
        parameters = dict(item.get("arguments") or {})
        records.append(
            DataRequestRecord(
                requirement=tool,
                status=DataFetchStatus.SUCCESS,
                call_id=item.get("call_id"),
                parameters=parameters,
                data_source=str(item.get("provider") or "mcp"),
                agent_id="data_collector",
                business_key=_business_key(tool, parameters, symbol, as_of),
                scope="target",
                value=item.get("data"),
                details={"cache_status": item.get("persistent_cache_status")},
            )
        )

    for item in (current_data.get("collection_report") or {}).get("failures", []):
        tool = str(item.get("tool") or "unknown")
        parameters = dict(item.get("parameters") or item.get("arguments") or {})
        records.append(
            DataRequestRecord(
                requirement=tool,
                status=DataFetchStatus(str(item.get("status") or "UPSTREAM_ERROR")),
                call_id=item.get("call_id"),
                parameters=parameters,
                data_source=str(item.get("data_source") or "mcp"),
                error_code=item.get("error_code") or item.get("code"),
                agent_id="data_collector",
                business_key=_business_key(tool, parameters, symbol, as_of),
                scope="target",
                details={"issues": item.get("issues") or []},
            )
        )

    # These two optional disclosures were previously misreported as failed even
    # when no Agent called either endpoint.  Preserve the omission explicitly.
    attempted_requirements = {record.requirement for record in records}
    profit_key = f"profit:{symbol or 'unknown'}:{_latest_completed_quarter(as_of)}"
    for requirement, label in (
        ("get_forecast_report", "业绩预告"),
        ("get_performance_express_report", "业绩快报"),
    ):
        if requirement not in attempted_requirements:
            records.append(
                DataRequestRecord(
                    requirement=requirement,
                    status=DataFetchStatus.NOT_ATTEMPTED,
                    parameters={},
                    data_source="none",
                    error_code=None,
                    agent_id="evidence_reconciliation",
                    business_key=profit_key,
                    scope="target",
                    details={"reason": f"{label} endpoint was not invoked by any Agent."},
                )
            )

    grouped: dict[str, list[DataRequestRecord]] = {}
    seen: set[str] = set()
    for record in records:
        identity = _record_identity(record)
        if identity in seen:
            continue
        seen.add(identity)
        grouped.setdefault(str(record.business_key), []).append(record)

    entries: dict[str, EvidenceRegistryEntry] = {}
    for business_key, grouped_records in sorted(grouped.items()):
        statuses = {record.status for record in grouped_records}
        resolved_status = next(
            status for status in STATUS_PRECEDENCE if status in statuses
        )
        values: list[dict[str, Any]] = []
        value_fingerprints: set[str] = set()
        for record in grouped_records:
            if record.status != DataFetchStatus.SUCCESS or record.value is None:
                continue
            fingerprint = json.dumps(
                record.value, ensure_ascii=False, sort_keys=True, default=str
            )
            if fingerprint in value_fingerprints:
                continue
            value_fingerprints.add(fingerprint)
            values.append(
                {
                    "value": record.value,
                    "source": record.data_source,
                    "tool": record.requirement,
                    "call_id": record.call_id,
                    "parameters": record.parameters,
                    "scope": record.scope,
                    "agent_id": record.agent_id,
                }
            )
        entries[business_key] = EvidenceRegistryEntry(
            business_key=business_key,
            status=resolved_status,
            records=grouped_records,
            values=values,
            conflict=len(value_fingerprints) > 1,
        )
    return EvidenceRegistry(entries=entries)


def compact_analysis_payload(
    results: dict[str, Any],
    *,
    content_limit: int = 2800,
    evidence_registry: EvidenceRegistry | None = None,
) -> dict:
    """Build the unified JSON hand-off without raw tool payload duplication."""
    payload: dict[str, Any] = {}
    for analysis_type in ANALYSIS_LABELS:
        result = results.get(analysis_type)
        if result is None:
            continue
        content = str(getattr(result, "content", "") or "")
        success = bool(getattr(result, "success", False))
        summary = (
            content
            if success
            else "该模块未通过质量门；其正文已从最终决策输入中移除。"
        )
        evidence = getattr(result, "evidence", []) or []
        payload[analysis_type] = {
            "summary": summary[:content_limit],
            "summary_truncated": len(summary) > content_limit,
            "success": success,
            "quality_status": getattr(result, "quality_status", "FAIL"),
            "claims": list(getattr(result, "claims", []) or [])[:12],
            "metrics": dict(getattr(result, "metrics", {}) or {}),
            "warnings": list(getattr(result, "warnings", []) or [])[:12],
            "confidence": getattr(result, "confidence", 0.0),
            "quality_score": getattr(result, "quality_score", 0.0),
            "data_completeness": getattr(result, "data_completeness", 0.0),
            "quality_issues": list(getattr(result, "quality_issues", []) or [])[:12],
            "data_requests": [
                item.model_dump(mode="json")
                if isinstance(item, DataRequestRecord)
                else item
                for item in list(getattr(result, "data_requests", []) or [])[:32]
            ],
            "evidence": [
                {
                    "tool": item.get("tool"),
                    "provider": item.get("provider"),
                    "record_count": item.get("record_count"),
                    "quality_score": item.get("quality_score"),
                    "raw_data_hash": item.get("raw_data_hash"),
                }
                for item in evidence[:16]
            ],
        }
    if evidence_registry is not None:
        payload["evidence_registry"] = evidence_registry.model_dump(mode="json")
    return payload


def parse_decision_narrative(value: Any) -> dict[str, str]:
    text = str(getattr(value, "content", value) or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    try:
        parsed = json.loads(cleaned)
    except (TypeError, ValueError):
        parsed = {"executive_summary": text}
    aliases = {
        "executive_summary": "执行摘要",
        "integrated_assessment": "综合评估",
        "investment_recommendation": "投资建议",
    }
    return {
        key: str(parsed.get(key) or parsed.get(label) or "").strip()
        for key, label in aliases.items()
    }


def _has_public_narrative(content: Any) -> bool:
    """Return False for empty or purely mechanical placeholder narratives."""
    normalize = lambda value: re.sub(  # noqa: E731 - tiny local normalizer
        r"[\s，。；：、!！*_`#>]+", "", str(value or "")
    )
    text = normalize(content)
    placeholders = {
        "证据不足，未形成该部分结论",
        "证据不足未形成该部分结论",
        "未形成该部分结论",
        "暂无结论",
        "无可用内容",
    }
    return bool(text) and text not in {normalize(item) for item in placeholders}


def _clean_embedded_section(content: str, label: str) -> str:
    """Normalize an Agent report before embedding it in a public report.

    Specialist Agents often return a preamble, their own H1 title, horizontal
    rules, internal tool traces and a repeated disclaimer.  Those are useful in
    execution logs but make the final report noisy and can break its heading
    hierarchy.  Keep the actual analysis while removing those artifacts.
    """
    text = _clean_public_text(content)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line in {"---", "***", "___"}:
            lines.append("")
            continue
        if re.match(r"^#\s+.+(?:报告|分析)\s*$", line):
            continue
        if re.match(rf"^##\s*{re.escape(label)}\s*$", line):
            continue
        if re.match(r"^(?:所有|全部|数据已).*(?:获取|读取|采集|核验).*(?:成功|完毕|完成)", line):
            continue
        if "以下为" in line and "报告" in line:
            continue
        if re.match(r"^\*\*(?:附：)?数据来源与完整性说明\*\*$", line):
            continue
        if re.match(r"^>\s*.*免责声明", line):
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            # The outer analysis section is H2.  Embedded titles therefore
            # start at H3 and retain their relative depth.
            depth = min(6, max(3, len(heading.group(1)) + 1))
            line = f"{'#' * depth} {heading.group(2)}"
        lines.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned or "该部分暂无可用内容。"


def _clean_public_text(content: Any) -> str:
    """Remove encoding/model artifacts and internal trace jargon."""
    text = str(content or "")
    text = text.lstrip("\ufeff")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    text = re.sub(r"```(?:markdown)?\s*", "", text, flags=re.I)
    text = text.replace("```", "")
    text = text.replace("\ufffd", "")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    text = re.sub(
        r"(?ms)^\|[^\n]*(?:工具/来源链|业务键|call_id|数据需求)[^\n]*\n(?:^\|.*\n?)+",
        "",
        text,
    )

    # Trace identifiers and raw tool statuses belong in execution logs, not in
    # a customer-facing investment report.
    kept_lines = []
    for line in text.splitlines():
        if re.search(r"\bcall_id\s*=|\|\s*业务键\s*\|", line, flags=re.I):
            continue
        if re.search(r"\bget_[a-z0-9_]+", line, flags=re.I):
            continue
        if re.search(
            r"集中取数|取数清单|清单项目|source_failures|top_k\s*=|"
            r"allowed_peer_symbols|禁止专家|工具明确要求|备用行情源",
            line,
            flags=re.I,
        ):
            continue
        kept_lines.append(line)
    text = "\n".join(kept_lines)

    replacements = {
        "INVALID_DATA": "暂无有效数据",
        "UPSTREAM_ERROR": "数据源暂不可用",
        "BUDGET_BLOCKED": "受数据获取限制",
        "NOT_ATTEMPTED": "未查询",
        "DEGRADED": "信息有限",
        "PASS": "数据充分",
        "SUCCESS": "已获取",
    }
    for internal, public in replacements.items():
        text = text.replace(internal, public)
    text = re.sub(r"（?业务键\s+[^，。；)）]+[，；]?", "", text)
    text = re.sub(r"错误码\s*[:：=]?\s*[A-Z0-9_/-]+[，；]?", "", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def _public_risk_section(risk_review: str) -> str:
    """Prefer actionable downside scenarios over internal audit prose."""
    cleaned = _clean_public_text(risk_review)
    match = re.search(
        r"(?ims)^#{1,6}\s*[^\n]*(?:下行情景|风险情景)[^\n]*\n(.*?)(?=^#{1,6}\s|\Z)",
        cleaned,
    )
    selected = match.group(1).strip() if match else cleaned
    selected = re.sub(r"^#.+?\n+", "", selected, count=1)
    return selected or "风险审查未返回可用内容。"


def _public_sources_and_limits(
    evidence_registry: EvidenceRegistry | None,
) -> str:
    """Render a concise user-facing source note without trace-table details."""
    if evidence_registry is None or not evidence_registry.entries:
        return "- 数据来源及可用性以各专项分析中的说明为准。"

    requirements = {
        record.requirement
        for entry in evidence_registry.entries.values()
        for record in entry.records
    }
    sources: list[str] = []
    if "get_tencent_quote" in requirements:
        sources.append(PUBLIC_SOURCE_LABELS["get_tencent_quote"])
    if requirements.intersection(
        {
            "get_profit_data",
            "get_balance_data",
            "get_cash_flow_data",
            "get_growth_data",
            "get_operation_data",
            "get_dupont_data",
            "get_historical_k_data",
            "get_dividend_data",
        }
    ):
        sources.append("Baostock 财务与历史行情")
    if "get_official_announcements" in requirements:
        sources.append(PUBLIC_SOURCE_LABELS["get_official_announcements"])

    lines = [f"- **主要数据来源**：{'、'.join(dict.fromkeys(sources))}。"] if sources else []
    failed_requirements: set[str] = set()
    for entry in evidence_registry.entries.values():
        for record in entry.records:
            if (
                record.requirement in PUBLIC_LIMIT_LABELS
                and record.status != DataFetchStatus.SUCCESS
            ):
                failed_requirements.add(record.requirement)
    if "get_financial_news" in failed_requirements:
        lines.append(
            "- **新闻与舆情局限**：截至分析日未取得有效新闻记录，事件判断仅依据官方公告，相关结论置信度较低。"
        )
    if "get_eastmoney_signals" in failed_requirements:
        lines.append(
            "- **资金流向局限**：资金流信号暂不可用，本报告不据此判断主力资金方向。"
        )
    lines.append(
        "- **价格口径**：复权行情用于趋势与技术指标，实际交易价格请以券商实时行情为准。"
    )
    return "\n".join(lines)


def reconcile_analysis_sections(
    analyses: dict[str, str], evidence_registry: EvidenceRegistry
) -> dict[str, str]:
    """Remove known prose contradictions after deterministic evidence merging."""
    reconciled = dict(analyses)
    dividend_success = any(
        key.startswith("dividend:") and entry.status == DataFetchStatus.SUCCESS
        for key, entry in evidence_registry.entries.items()
    )
    missing = r"(?:未获取|没有获取|无法获取|获取失败|未能取得|数据缺失|暂无数据)"
    if dividend_success:
        pattern = re.compile(
            rf"(?=[^。\n]*(?:分红|股息|派息))(?=[^。\n]*{missing})[^。\n]+[。]?",
            flags=re.I,
        )
        for key, content in reconciled.items():
            if pattern.search(content or ""):
                reconciled[key] = pattern.sub(
                    "分红数据已由其他 Agent 成功取得，统一证据登记状态为 SUCCESS。",
                    content,
                )

    requirement_labels = {
        "get_forecast_report": "业绩预告",
        "get_performance_express_report": "业绩快报",
    }
    for requirement, label in requirement_labels.items():
        records = [
            record
            for entry in evidence_registry.entries.values()
            for record in entry.records
            if record.requirement == requirement
        ]
        if not records:
            continue
        successful = next(
            (record for record in records if record.status == DataFetchStatus.SUCCESS),
            None,
        )
        trace_record = successful or records[0]
        trace = (
            f"{label}取数状态：{trace_record.status.value}；"
            f"call_id={trace_record.call_id or '无（未调用）'}；"
            f"数据源={trace_record.data_source}。"
        )
        pattern = re.compile(
            rf"(?=[^。\n]*{label})(?=[^。\n]*{missing})[^。\n]+[。]?",
            flags=re.I,
        )
        for key, content in reconciled.items():
            reconciled[key] = pattern.sub(trace, content or "")

    requirement_keywords = {
        "get_profit_data": r"(?:盈利|利润|净利润)",
        "get_operation_data": r"(?:营收|经营|周转)",
        "get_growth_data": r"(?:成长|增长率|同比)",
        "get_balance_data": r"(?:资产负债|资产|负债)",
        "get_cash_flow_data": r"(?:现金流|经营现金)",
        "get_dupont_data": r"(?:杜邦|ROE|净资产收益率)",
        "get_historical_k_data": r"(?:历史行情|K线|复权行情)",
        "get_mootdx_bars": r"(?:历史行情|K线|复权行情)",
        "get_tencent_quote": r"(?:实时行情|报价|最新价)",
        "get_official_announcements": r"(?:官方公告|公告)",
        "get_financial_news": r"(?:财经新闻|新闻)",
    }
    for requirement, keywords in requirement_keywords.items():
        records = [
            record
            for entry in evidence_registry.entries.values()
            for record in entry.records
            if record.requirement == requirement
        ]
        if not records:
            continue
        success = next(
            (record for record in records if record.status == DataFetchStatus.SUCCESS),
            None,
        )
        if success:
            trace = (
                f"{requirement} 取数状态：SUCCESS；call_id={success.call_id}；"
                f"数据源={success.data_source}。"
            )
        else:
            trace = "；".join(
                (
                    f"{record.status.value}[call_id={record.call_id or '无（未调用）'},"
                    f"数据源={record.data_source},错误码={record.error_code or '-'}]"
                )
                for record in records
            )
            trace = f"{requirement} 取数状态：{trace}。"
        pattern = re.compile(
            rf"(?=[^。\n]*{keywords})(?=[^。\n]*{missing})[^。\n]+[。]?",
            flags=re.I,
        )
        for key, content in reconciled.items():
            reconciled[key] = pattern.sub(trace, content or "")

    budget_records = [
        record
        for entry in evidence_registry.entries.values()
        for record in entry.records
        if record.status == DataFetchStatus.BUDGET_BLOCKED
    ]
    if budget_records:
        budget_trace = "；".join(
            f"{record.requirement}[call_id={record.call_id},错误码={record.error_code}]"
            for record in budget_records
        )
        free_budget_claim = re.compile(r"[^。\n]*因预算[^。\n]*(?:失败|未获取|无法获取)[^。\n]*[。]?")
        for key, content in reconciled.items():
            reconciled[key] = free_budget_claim.sub(
                f"结构化取数状态：BUDGET_BLOCKED；{budget_trace}。", content or ""
            )
    return reconciled


def _compact_json(value: Any, limit: int = 180) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_evidence_registry(evidence_registry: EvidenceRegistry | None) -> str:
    if evidence_registry is None or not evidence_registry.entries:
        return "- 未建立统一证据登记表。"
    lines = [
        "| 业务键 | 合并状态 | 数据需求 | call_id | 参数 | 数据源 | 错误码 |",
        "|---|---|---|---|---|---|---|",
    ]
    for business_key, entry in evidence_registry.entries.items():
        for record in entry.records:
            lines.append(
                "| "
                + " | ".join(
                    str(value).replace("|", "\\|")
                    for value in (
                        business_key,
                        entry.status.value,
                        record.requirement,
                        record.call_id or "无（未调用）",
                        _compact_json(record.parameters),
                        record.data_source,
                        record.error_code or "-",
                    )
                )
                + " |"
            )
    conflict_lines = []
    for business_key, entry in evidence_registry.entries.items():
        if not entry.conflict:
            continue
        conflict_lines.append(f"- `{business_key}`：检测到多源冲突，未自动选值。")
        conflict_lines.extend(
            (
                f"  - source={item['source']}；tool={item['tool']}；"
                f"call_id={item['call_id']}；口径={_compact_json(item['parameters'])}；"
                f"value={_compact_json(item['value'], 360)}"
            )
            for item in entry.values
        )
    if conflict_lines:
        lines.extend(["", "### 多源冲突（保留全部值与口径）", *conflict_lines])
    return "\n".join(lines)


def render_financial_report(
    *,
    company_name: str,
    stock_code: str,
    narrative: dict[str, str],
    analyses: dict[str, str],
    risk_review: str,
    decision_permissions: dict[str, Any],
    structured_payload: dict[str, Any],
    current_time_info: str,
    evidence_registry: EvidenceRegistry | None = None,
) -> str:
    """Render a concise public report; keep audit detail in structured logs."""
    del decision_permissions, structured_payload  # Internal controls are enforced upstream.
    as_of_match = re.search(r"\d{4}-\d{2}-\d{2}", str(current_time_info))
    as_of = as_of_match.group(0) if as_of_match else str(current_time_info)
    sections = [
        f"# {company_name}（{stock_code}）投资研究报告",
        f"- **证券代码**：{stock_code}",
        f"- **分析日期**：{as_of}",
        "- **报告类型**：多维度综合研究",
        "> 本报告由 AI 基于公开数据自动生成，仅供研究参考，不构成投资建议。",
        "---",
    ]
    for key, title in (
        ("executive_summary", "核心结论"),
        ("investment_recommendation", "投资建议"),
        ("integrated_assessment", "综合研判"),
    ):
        content = _clean_public_text(narrative.get(key, ""))
        if _has_public_narrative(content):
            sections.extend([f"## {title}", content])
    for key, label in ANALYSIS_LABELS.items():
        sections.extend(
            [f"## {label}", _clean_embedded_section(analyses.get(key, ""), label)]
        )
    sections.extend(
        [
            "## 关键风险",
            _public_risk_section(risk_review),
            "## 数据来源与局限",
            _public_sources_and_limits(evidence_registry),
            f"**分析基准时间：{current_time_info}**",
        ]
    )
    return "\n\n".join(sections).strip()
