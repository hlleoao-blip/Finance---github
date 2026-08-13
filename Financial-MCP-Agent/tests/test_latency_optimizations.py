import asyncio
import shutil
import time
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from pydantic import BaseModel

from src import workflow
from src.agent_loop.tracing import InMemoryTraceRecorder
from src.tools import data_collection
from src.tools.data_collection import (
    CollectionSpec,
    build_collection_plan,
    collection_cache_key,
    two_market_sources_consistent,
)
from src.tools.call_control import allocate_category_budget
from src.tools.persistent_cache import PersistentToolCache
from src.tools.react_tooling import AuditedToolGateway, select_runtime_tools
from src.utils.report_renderer import (
    parse_decision_narrative,
    render_financial_report,
)
from src.utils.state_definition import AnalysisResult, WorkflowState
from src.utils.technical_indicators import calculate_technical_snapshot


class Tool:
    description = "test tool"
    args_schema = None

    def __init__(self, name, responses=None):
        self.name = name
        self.responses = list(responses or [])
        self.calls = []

    async def ainvoke(self, arguments):
        self.calls.append(arguments)
        return self.responses.pop(0)


class CodeArgs(BaseModel):
    code: str


class ExecutionLogger:
    def log_tool_usage(self, *args, **kwargs):
        pass


def envelope(value):
    return {
        "ok": True,
        "data": {"content": value},
        "error": None,
        "meta": {"provider": "test"},
    }


def failure_envelope(code="TOOL_TIMEOUT", message="timed out", provider="test"):
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message, "retryable": True},
        "meta": {"provider": provider},
    }


class PersistentCacheTests(unittest.TestCase):
    def test_ttl_and_stale_while_revalidate_windows(self):
        directory = Path("tests") / f"cache_test_{uuid4().hex}"
        directory.mkdir()
        try:
            cache = PersistentToolCache(directory)
            entry = {
                "call_id": "c1",
                "request_hash": "r1",
                "raw_data_hash": "d1",
                "quality_score": 1.0,
                "data": {"price": 10},
            }
            cache.store("get_tencent_quote", {"code": "sh.600519"}, entry, stored_at=100)

            fresh = cache.lookup(
                "get_tencent_quote", {"code": "sh.600519"}, now=150
            )
            stale = cache.lookup(
                "get_tencent_quote", {"code": "sh.600519"}, now=300
            )
            expired = cache.lookup(
                "get_tencent_quote", {"code": "sh.600519"}, now=2000
            )

            self.assertEqual(fresh.status, "fresh")
            self.assertEqual(stale.status, "stale")
            self.assertEqual(expired.status, "expired")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_historical_prefix_enables_incremental_range_fetch(self):
        directory = Path("tests") / f"cache_test_{uuid4().hex}"
        directory.mkdir()
        try:
            cache = PersistentToolCache(directory)
            cached_arguments = {
                "code": "sh.600519",
                "start_date": "2026-01-01",
                "end_date": "2026-08-10",
                "frequency": "d",
                "adjust_flag": "1",
            }
            cache.store(
                "get_historical_k_data",
                cached_arguments,
                {"data": "cached", "call_id": "c1"},
            )
            requested = {
                **cached_arguments,
                "start_date": "2026-02-01",
                "end_date": "2026-08-11",
            }

            prefix = cache.find_historical_prefix(
                "get_historical_k_data", requested
            )

            self.assertIsNotNone(prefix)
            self.assertEqual(prefix[0]["end_date"], "2026-08-10")
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class TechnicalIndicatorTests(unittest.TestCase):
    def test_markdown_ohlcv_is_calculated_locally(self):
        lines = [
            "| date | open | high | low | close | volume |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for index in range(1, 41):
            close = 100 + index
            lines.append(
                f"| 2026-01-{index:02d} | {close - 1} | {close + 2} | "
                f"{close - 2} | {close} | {1000 + index} |"
            )

        result = calculate_technical_snapshot("\n".join(lines))

        self.assertTrue(result["available"])
        self.assertEqual(result["record_count"], 40)
        self.assertEqual(result["latest_close"], 140.0)
        self.assertIn("rsi14", result)
        self.assertIn("macd_histogram", result)
        self.assertEqual(result["support_20d"], 119.0)


class CollectionConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_common_calls_are_bounded_and_concurrent(self):
        active = 0
        maximum = 0

        async def execute(_executor, step, _state):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.04)
            active -= 1
            return SimpleNamespace(
                ok=True,
                data={"tool": step.candidate_tools[0]},
                error=None,
                meta=SimpleNamespace(
                    call_id=step.id,
                    request_hash="sha256:request",
                    raw_data_hash="sha256:data",
                    provider="test",
                    upstream_meta={},
                ),
            )

        async def validate(*_args, **_kwargs):
            return SimpleNamespace(passed=True, score=1.0, issues=[])

        tools = [Tool("one"), Tool("two")]
        calls = (("one", lambda state: {"code": state.target_symbol}), ("two", lambda _state: {}))
        state = WorkflowState(
            task="test",
            symbol="sh.600519",
            data={
                "persistent_cache_enabled": False,
                "collection_concurrency": 2,
            },
        )
        started = time.perf_counter()
        with (
            patch.object(data_collection, "COMMON_COLLECTION_CALLS", calls),
            patch.object(data_collection, "get_all_tools_for_orchestrator", return_value=tools),
            patch.object(data_collection.MCPToolExecutor, "execute", execute),
            patch.object(data_collection.FinancialDataValidator, "validate", validate),
        ):
            update = await data_collection.collect_common_target_data(state)
        elapsed = time.perf_counter() - started

        self.assertEqual(maximum, 2)
        # ``maximum`` is the deterministic concurrency assertion. Keep only a
        # generous wall-clock guard because Windows CI scheduling can add >40ms.
        self.assertLess(elapsed, 0.15)
        self.assertEqual(update["data"]["collection_report"]["cached_calls"], 2)

    async def test_collector_skips_mootdx_after_two_consistent_sources(self):
        executed = []

        async def execute(_executor, step, _state):
            tool_name = step.candidate_tools[0]
            executed.append(tool_name)
            if tool_name == "get_tencent_quote":
                value = {
                    "datetime": "2026-08-11T16:00:00+08:00",
                    "change_pct": -0.17,
                }
                provider = "tencent"
            else:
                value = (
                    "| date | close | pctChg |\n"
                    "| --- | --- | --- |\n"
                    "| 2026-08-11 | 10326.65 | -0.175 |"
                )
                provider = "baostock"
            return SimpleNamespace(
                ok=True,
                data=value,
                error=None,
                meta=SimpleNamespace(
                    call_id=step.id,
                    request_hash="sha256:request",
                    raw_data_hash="sha256:data",
                    provider=provider,
                    upstream_meta={},
                ),
            )

        async def validate(*_args, **_kwargs):
            return SimpleNamespace(passed=True, score=1.0, issues=[])

        specs = (
            CollectionSpec(
                "quote", "最新行情", "required", "required",
                "get_tencent_quote", lambda state: {"code": state.target_symbol},
            ),
            CollectionSpec(
                "bars", "K线", "required", "required",
                "get_historical_k_data", lambda state: {"code": state.target_symbol},
            ),
            CollectionSpec(
                "backup", "备用行情源", "optional", "supplemental",
                "get_mootdx_bars", lambda state: {"code": state.target_symbol},
                condition="unless_two_consistent_market_sources",
            ),
        )
        tools = [
            Tool("get_tencent_quote"),
            Tool("get_historical_k_data"),
            Tool("get_mootdx_bars"),
        ]
        state = WorkflowState(
            task="test",
            symbol="sh.600519",
            data={"persistent_cache_enabled": False},
        )
        with (
            patch.object(data_collection, "COMMON_COLLECTION_CALLS", specs),
            patch.object(data_collection, "get_all_tools_for_orchestrator", return_value=tools),
            patch.object(data_collection.MCPToolExecutor, "execute", execute),
            patch.object(data_collection.FinancialDataValidator, "validate", validate),
        ):
            update = await data_collection.collect_common_target_data(state)

        backup = next(
            item for item in update["data"]["collection_plan"]
            if item["id"] == "backup"
        )
        self.assertEqual(set(executed), {"get_tencent_quote", "get_historical_k_data"})
        self.assertEqual(backup["status"], "SKIPPED")
        self.assertEqual(backup["code"], "TWO_CONSISTENT_MARKET_SOURCES")


class CentralCollectionPlanTests(unittest.TestCase):
    def test_plan_materializes_required_important_and_optional_data(self):
        state = WorkflowState(
            task="test",
            symbol="sh.600519",
            company_name="贵州茅台",
            as_of=date(2026, 8, 11),
        )

        plan = build_collection_plan(state)
        requirements = {item["requirement"] for item in plan}
        latest_profit = next(
            item for item in plan if item["id"] == "latest_financial:profit"
        )
        yoy_profit = next(
            item for item in plan if item["id"] == "yoy_financial:profit"
        )

        self.assertTrue(
            {"最新行情", "K线", "最新财务", "同比财务", "公告", "新闻"}
            <= requirements
        )
        self.assertTrue({"分红", "历史估值", "杜邦"} <= requirements)
        self.assertTrue({"资金流", "备用行情源", "更早年度数据"} <= requirements)
        self.assertEqual(latest_profit["arguments"]["year"], "2026")
        self.assertEqual(latest_profit["arguments"]["quarter"], 2)
        self.assertEqual(yoy_profit["arguments"]["year"], "2025")
        self.assertEqual(yoy_profit["arguments"]["quarter"], 2)
        self.assertTrue(all(item["status"] == "PLANNED" for item in plan))

    def test_category_budget_reserves_at_least_sixty_percent_for_required(self):
        self.assertEqual(
            allocate_category_budget(14),
            {"required": 9, "cross_validation": 3, "supplemental": 2},
        )
        self.assertEqual(
            allocate_category_budget(6),
            {"required": 4, "cross_validation": 1, "supplemental": 1},
        )

    def test_two_consistent_market_sources_suppress_a_third_source(self):
        cache = {
            "quote": {
                "tool": "get_tencent_quote",
                "data": {
                    "datetime": "2026-08-11T16:00:00+08:00",
                    "change_pct": -0.17,
                },
            },
            "bars": {
                "tool": "get_historical_k_data",
                "data": (
                    "| date | close | pctChg |\n"
                    "| --- | --- | --- |\n"
                    "| 2026-08-11 | 10326.65 | -0.175 |"
                ),
            },
        }

        self.assertTrue(two_market_sources_consistent(cache))


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_timeout_is_marked_incomplete_for_quality_gate(self):
        async def slow(_state):
            await asyncio.sleep(0.05)
            return {}

        state = WorkflowState(
            task="technical",
            data={"agent_timeout_seconds": {"technical_analyst": 0.01}},
        )
        update = await workflow._run_budgeted_node(
            state, slow, "technical_analyst", "technical"
        )

        result = update["analysis_results"]["technical"]
        self.assertFalse(result.success)
        self.assertIn("AGENT_TIMEOUT", result.error)
        self.assertTrue(update["data"]["latency_degradations"][0]["data_incomplete"])

    async def test_risk_timeout_uses_deterministic_degraded_review(self):
        async def slow(_state):
            await asyncio.sleep(0.05)
            return {}

        state = WorkflowState(
            task="full analysis",
            symbol="sz.300750",
            company_name="宁德时代",
            data={
                "agent_timeout_seconds": {"risk_reviewer": 0.01},
                "quality_gate": {
                    "status": "DEGRADED",
                    "passed": True,
                    "coverage": 1.0,
                    "degraded_analysis_types": ["event"],
                    "failed_analysis_types": [],
                },
            },
        )

        update = await workflow._run_budgeted_node(
            state, slow, "risk_reviewer", "risk_review"
        )

        result = update["analysis_results"]["risk_review"]
        self.assertTrue(result.success)
        self.assertEqual(result.quality_status, "DEGRADED")
        self.assertTrue(update["data"]["risk_review_fallback"]["used"])
        self.assertIn("自动风险审查", result.content)

    async def test_decision_timeout_renders_deterministic_final_report(self):
        async def slow(_state):
            await asyncio.sleep(0.05)
            return {}

        results = {
            analysis_type: AnalysisResult(
                agent_id=f"{analysis_type}_agent",
                analysis_type=analysis_type,
                content=f"{analysis_type}已验证分析内容。" * 20,
                success=True,
                quality_status="PASS",
                quality_passed=True,
            )
            for analysis_type in ("fundamental", "technical", "value", "event")
        }
        results["risk_review"] = AnalysisResult(
            agent_id="risk_review_agent",
            analysis_type="risk_review",
            content="风险审查已验证内容。" * 20,
            success=True,
            quality_status="DEGRADED",
            quality_passed=True,
        )
        state = WorkflowState(
            task="full analysis",
            symbol="sz.300750",
            company_name="宁德时代",
            analysis_results=results,
            data={
                "stock_code": "sz.300750",
                "company_name": "宁德时代",
                "current_time_info": "2026-08-11 20:24:24",
                "agent_timeout_seconds": {"decision_maker": 0.01},
                "quality_gate": {
                    "status": "DEGRADED",
                    "passed": True,
                    "coverage": 1.0,
                },
            },
        )

        update = await workflow._run_budgeted_node(
            state, slow, "decision_maker"
        )
        report_path = Path(update["data"]["report_path"])
        try:
            self.assertEqual(update["data"]["decision_status"], "completed")
            self.assertTrue(update["data"]["decision_fallback"]["used"])
            self.assertIn("确定性降级报告", update["data"]["final_report"])
            self.assertTrue(report_path.exists())
        finally:
            report_path.unlink(missing_ok=True)

    async def test_gateway_rejects_calls_beyond_agent_budget(self):
        tool = Tool("get_stock_basic_info", [envelope({"code": "sh.600519"})])
        state = WorkflowState(
            task="test",
            symbol="sh.600519",
            data={
                "persistent_cache_enabled": False,
                "agent_tool_call_limits": {"fundamental_agent": 1},
            },
        )
        gateway = AuditedToolGateway(
            agent_name="fundamental_agent",
            state=state,
            tools=[tool],
            trace_recorder=InMemoryTraceRecorder(),
            execution_logger=ExecutionLogger(),
        )
        wrapped = gateway.wrap_tools()[0]

        await wrapped.ainvoke({"code": "sh.600519"})
        second = await wrapped.ainvoke({"code": "sh.600519"})

        self.assertEqual(len(tool.calls), 1)
        self.assertIn("exceeded", str(second))
        self.assertEqual(
            gateway.events[-1]["issues"][0]["code"], "TOOL_CALL_BUDGET_EXCEEDED"
        )

    async def test_run_cache_hits_do_not_consume_external_call_budget(self):
        basic_args = {"code": "sh.600519"}
        basic = Tool("get_stock_basic_info")
        basic.args_schema = CodeArgs
        latest = Tool("get_latest_trading_date", [envelope("2026-08-11")])
        timeframe = Tool(
            "get_market_analysis_timeframe", [envelope("2026-02-01 to 2026-08-11")]
        )
        state = WorkflowState(
            task="test",
            symbol="sh.600519",
            data={
                "persistent_cache_enabled": False,
                "agent_tool_call_limits": {"value_agent": 1},
                "collection_cache": {
                    collection_cache_key("get_stock_basic_info", basic_args): {
                        "data": {"code": "sh.600519", "name": "贵州茅台"},
                        "call_id": "collection-basic",
                        "request_hash": "sha256:request",
                        "raw_data_hash": "sha256:data",
                        "quality_score": 1.0,
                        "record_count": 1,
                    }
                },
            },
        )
        gateway = AuditedToolGateway(
            agent_name="value_agent",
            state=state,
            tools=[basic, latest, timeframe],
            trace_recorder=InMemoryTraceRecorder(),
            execution_logger=ExecutionLogger(),
        )
        wrapped = {tool.name: tool for tool in gateway.wrap_tools()}

        cached = await wrapped["get_stock_basic_info"].ainvoke(basic_args)
        external = await wrapped["get_latest_trading_date"].ainvoke({})
        rejected = await wrapped["get_market_analysis_timeframe"].ainvoke(
            {"period": "half_year"}
        )

        self.assertIn("sh.600519", str(cached))
        self.assertEqual(external, "2026-08-11")
        self.assertEqual(basic.calls, [])
        self.assertEqual(len(latest.calls), 1)
        self.assertEqual(timeframe.calls, [])
        self.assertTrue(gateway.events[0]["cache_hit"])
        self.assertIn("external-call budget", str(rejected))

    async def test_budget_block_immediately_disables_all_followup_calls(self):
        tool = Tool(
            "get_stock_basic_info",
            [envelope({"code": "sh.600519"})],
        )
        state = WorkflowState(
            task="test",
            symbol="sh.600519",
            data={
                "persistent_cache_enabled": False,
                "agent_tool_call_limits": {"fundamental_agent": 1},
            },
        )
        gateway = AuditedToolGateway(
            agent_name="fundamental_agent",
            state=state,
            tools=[tool],
            trace_recorder=InMemoryTraceRecorder(),
            execution_logger=ExecutionLogger(),
        )
        wrapped = gateway.wrap_tools()[0]

        await wrapped.ainvoke({"code": "sh.600519"})
        await wrapped.ainvoke({"code": "sh.600519"})
        third = await wrapped.ainvoke({"code": "sh.600519"})

        self.assertEqual(len(tool.calls), 1)
        self.assertIn("prior BUDGET_BLOCKED", str(third))
        self.assertEqual(
            gateway.events[-1]["error_code"], "AGENT_BUDGET_CIRCUIT_OPEN"
        )

    async def test_provider_opens_circuit_after_two_transport_failures(self):
        tool = Tool(
            "get_tencent_quote",
            [
                failure_envelope(provider="tencent"),
                failure_envelope(provider="tencent"),
                envelope({"price": 10}),
            ],
        )
        state = WorkflowState(
            task="test",
            symbol="sh.600519",
            data={"persistent_cache_enabled": False},
        )
        gateway = AuditedToolGateway(
            agent_name="technical_agent",
            state=state,
            tools=[tool],
            trace_recorder=InMemoryTraceRecorder(),
            execution_logger=ExecutionLogger(),
        )
        wrapped = gateway.wrap_tools()[0]

        first = await wrapped.ainvoke({"code": "sh.600519"})
        second = await wrapped.ainvoke({"code": "sh.600519"})

        self.assertIn("failed after 2", str(first))
        self.assertIn("circuit is open", str(second))
        self.assertEqual(len(tool.calls), 2)
        self.assertEqual(gateway.events[-1]["error_code"], "PROVIDER_CIRCUIT_OPEN")

    async def test_centralized_collection_rejects_unplanned_upstream_call(self):
        tool = Tool(
            "get_stock_basic_info",
            [envelope({"code": "sh.600519"})],
        )
        state = WorkflowState(
            task="test",
            symbol="sh.600519",
            data={
                "persistent_cache_enabled": False,
                "centralized_collection_enforced": True,
                "collection_plan": [],
            },
        )
        gateway = AuditedToolGateway(
            agent_name="fundamental_agent",
            state=state,
            tools=[tool],
            trace_recorder=InMemoryTraceRecorder(),
            execution_logger=ExecutionLogger(),
        )

        rejected = await gateway.wrap_tools()[0].ainvoke({"code": "sh.600519"})

        self.assertEqual(tool.calls, [])
        self.assertIn("centralized collection cache", str(rejected))
        self.assertEqual(gateway.events[-1]["error_code"], "UNPLANNED_TOOL_CALL")

    def test_value_agent_has_budget_for_required_and_optional_categories(self):
        from src.tools.react_tooling import DEFAULT_AGENT_TOOL_LIMITS

        self.assertEqual(
            DEFAULT_AGENT_TOOL_LIMITS,
            {
                "fundamental_agent": 14,
                "technical_agent": 6,
                "value_agent": 14,
                "event_agent": 6,
            },
        )

    async def test_financial_year_argument_is_normalized_to_string(self):
        tool = Tool(
            "get_profit_data",
            [envelope({"code": "sh.600519", "roe": 0.10})],
        )
        state = WorkflowState(
            task="valuation",
            symbol="sh.600519",
            data={"persistent_cache_enabled": False},
        )
        gateway = AuditedToolGateway(
            agent_name="value_agent",
            state=state,
            tools=[tool],
            trace_recorder=InMemoryTraceRecorder(),
            execution_logger=ExecutionLogger(),
        )

        await gateway._invoke(
            tool,
            {"code": "sh.600519", "year": 2025, "quarter": 4},
        )

        self.assertEqual(tool.calls[0]["year"], "2025")
        self.assertEqual(tool.calls[0]["quarter"], 4)

    def test_technical_agent_skips_redundant_slow_or_failed_optional_tools(self):
        tools = [
            Tool("get_historical_k_data"),
            Tool("get_mootdx_bars"),
            Tool("get_eastmoney_signals"),
            Tool("get_tencent_quote"),
        ]
        state = WorkflowState(
            task="technical",
            symbol="sh.600519",
            data={
                "collection_cache": {
                    "history": {"tool": "get_historical_k_data"},
                },
                "collection_report": {
                    "market_sources_consistent": True,
                    "failures": [{"tool": "get_eastmoney_signals"}],
                },
            },
        )

        selected = select_runtime_tools("technical_agent", state, tools)
        selected_names = {tool.name for tool in selected}

        self.assertIn("get_historical_k_data", selected_names)
        self.assertIn("get_tencent_quote", selected_names)
        self.assertNotIn("get_mootdx_bars", selected_names)
        self.assertNotIn("get_eastmoney_signals", selected_names)

        state.data["collection_report"]["market_sources_consistent"] = False
        selected_after_conflict = {
            tool.name for tool in select_runtime_tools("technical_agent", state, tools)
        }
        self.assertIn("get_mootdx_bars", selected_after_conflict)


class ReportRenderingTests(unittest.TestCase):
    def test_summary_agent_has_json_serializer_for_decision_payload(self):
        from src.agents import summary_agent

        payload = {"company": "贵州茅台", "stock_code": "sh.600519"}
        self.assertEqual(
            summary_agent.json.loads(
                summary_agent.json.dumps(payload, ensure_ascii=False)
            ),
            payload,
        )

    def test_decision_json_is_rendered_with_deterministic_sections(self):
        narrative = parse_decision_narrative(
            '{"executive_summary":"摘要", "integrated_assessment":"综合", '
            '"investment_recommendation":"建议"}'
        )
        report = render_financial_report(
            company_name="贵州茅台",
            stock_code="sh.600519",
            narrative=narrative,
            analyses={key: f"{key} evidence section" for key in ("fundamental", "technical", "value", "event")},
            risk_review="风险复核内容",
            decision_permissions={"rating": True, "target_price": False},
            structured_payload={},
            current_time_info="2026-08-11",
        )

        self.assertIn("## 核心结论", report)
        self.assertIn("## 投资建议", report)
        self.assertIn("## 技术分析", report)
        self.assertIn("## 关键风险", report)
        self.assertIn("## 数据来源与局限", report)
        self.assertNotIn("决策权限", report)
        self.assertNotIn("证据不足，不允许", report)
        self.assertNotIn("统一证据登记表", report)
        self.assertTrue(report.endswith("**分析基准时间：2026-08-11**"))

    def test_public_report_strips_internal_traces_and_normalizes_headings(self):
        narrative = parse_decision_narrative(
            '{"executive_summary":"结论（PASS）。业务键 quote:x，错误码 INVALID_DATA。", '
            '"integrated_assessment":"综合", "investment_recommendation":"建议"}'
        )
        report = render_financial_report(
            company_name="示例公司",
            stock_code="sz.000001",
            narrative=narrative,
            analyses={
                "fundamental": (
                    "数据已全部获取并核验完毕。\n\n"
                    "# 示例公司基本面分析报告\n\n"
                    "## 一、财务概览\n\n"
                    "| 证据类别 | 工具/来源链 | 状态 |\n"
                    "|---|---|---|\n"
                    "| 财务 | get_profit_data | SUCCESS |\n\n"
                    "get_profit_data 取数状态：SUCCESS；call_id=abc；数据源=mcp。\n\n"
                    "正文。\n\n> 免责声明：重复声明。"
                ),
                "technical": "技术正文。",
                "value": "估值正文。",
                "event": "事件正文。",
            },
            risk_review="## 1. 内部审查\n审查噪音。\n## 2. 可证伪的下行情景\n- 风险一。\n- 风险二。",
            decision_permissions={"rating": True},
            structured_payload={},
            current_time_info="2026-08-11 20:00",
        )

        self.assertIn("### 一、财务概览", report)
        self.assertIn("- 风险一。", report)
        self.assertNotIn("call_id", report)
        self.assertNotIn("get_profit_data", report)
        self.assertNotIn("工具/来源链", report)
        self.assertNotIn("SUCCESS", report)
        self.assertNotIn("统一证据登记", report)
        self.assertNotIn("INVALID_DATA", report)
        self.assertNotIn("�", report)

    def test_placeholder_narrative_sections_are_omitted(self):
        narrative = parse_decision_narrative('{"executive_summary":"有效摘要"}')
        narrative["investment_recommendation"] = "证据不足，未形成该部分结论。"
        report = render_financial_report(
            company_name="示例公司",
            stock_code="sz.000001",
            narrative=narrative,
            analyses={
                "fundamental": "基本面正文。",
                "technical": "技术正文。",
                "value": "估值正文。",
                "event": "事件正文。",
            },
            risk_review="风险正文。",
            decision_permissions={},
            structured_payload={},
            current_time_info="2026-08-11",
        )

        self.assertIn("## 核心结论", report)
        self.assertNotIn("## 投资建议", report)
        self.assertNotIn("## 综合研判", report)
        self.assertNotIn("证据不足，未形成该部分结论", report)


if __name__ == "__main__":
    unittest.main()
