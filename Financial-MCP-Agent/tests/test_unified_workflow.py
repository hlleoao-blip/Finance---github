import unittest
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pydantic import BaseModel

from src.agent_loop.state import AgentLoopState
from src.agent_loop.tracing import InMemoryTraceRecorder
from src.tools.mcp_client import get_mcp_tools
from src.tools.react_tooling import AuditedToolGateway
from src.tools.data_collection import collection_cache_key
from src.utils.state_definition import AnalysisResult, WorkflowState, failed_analysis_update
from src import workflow


class ToolArgs(BaseModel):
    code: str


class FakeMCPTool:
    name = "get_stock_basic_info"
    description = "load stock identity"
    args_schema = ToolArgs

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, arguments):
        self.calls.append(arguments)
        return self.responses.pop(0)


class FakeExecutionLogger:
    def __init__(self):
        self.tool_calls = []

    def log_tool_usage(self, *args, **kwargs):
        self.tool_calls.append((args, kwargs))


def success(data):
    return {
        "ok": True,
        "data": {"content": data},
        "error": None,
        "meta": {"provider": "test"},
    }


def retryable_failure():
    return {
        "ok": False,
        "data": None,
        "error": {
            "code": "UPSTREAM_TIMEOUT",
            "message": "temporary failure",
            "retryable": True,
            "details": {},
        },
        "meta": {"provider": "test"},
    }


class UnifiedStateTests(unittest.TestCase):
    def test_agent_loop_state_is_the_unified_workflow_state(self):
        self.assertIs(AgentLoopState, WorkflowState)

    def test_failure_update_always_contains_analysis_result(self):
        update = failed_analysis_update(
            agent_id="technical_agent",
            analysis_type="technical",
            data={"stock_code": "sh.600519"},
            messages=[],
            metadata={},
            error="failed",
        )
        result = update["analysis_results"]["technical"]
        self.assertFalse(result.success)
        self.assertEqual(result.error, "failed")


class ToolGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_unrestricted_compatibility_loader_fails_closed(self):
        with self.assertRaises(ValueError):
            await get_mcp_tools()

    async def test_gateway_retries_validates_and_records_evidence(self):
        tool = FakeMCPTool(
            [retryable_failure(), success({"code": "sh.600519", "name": "贵州茅台"})]
        )
        trace = InMemoryTraceRecorder()
        state = WorkflowState(
            task="分析贵州茅台",
            symbol="sh.600519",
            max_retries_per_step=1,
            data={"persistent_cache_enabled": False},
        )
        gateway = AuditedToolGateway(
            agent_name="fundamental_agent",
            state=state,
            tools=[tool],
            trace_recorder=trace,
            execution_logger=FakeExecutionLogger(),
        )

        result = await gateway.wrap_tools()[0].ainvoke({"code": "sh.600519"})

        self.assertEqual(result["code"], "sh.600519")
        self.assertEqual(len(tool.calls), 2)
        self.assertEqual(len(gateway.events), 2)
        self.assertEqual(len(gateway.evidence), 1)
        self.assertIn("react_tool_retry", [event["stage"] for event in trace.events])

    async def test_gateway_reuses_common_collection_cache(self):
        arguments = {"code": "sh.600519"}
        tool = FakeMCPTool([])
        state = WorkflowState(
            task="分析贵州茅台",
            symbol="sh.600519",
            data={
                "persistent_cache_enabled": False,
                "collection_cache": {
                    collection_cache_key(tool.name, arguments): {
                        "tool": tool.name,
                        "arguments": arguments,
                        "data": {"code": "sh.600519", "name": "贵州茅台"},
                        "call_id": "collection-call",
                        "request_hash": "sha256:request",
                        "raw_data_hash": "sha256:data",
                        "quality_score": 1.0,
                    }
                }
            },
        )
        gateway = AuditedToolGateway(
            agent_name="fundamental_agent",
            state=state,
            tools=[tool],
            trace_recorder=InMemoryTraceRecorder(),
            execution_logger=FakeExecutionLogger(),
        )

        result = await gateway.wrap_tools()[0].ainvoke(arguments)

        self.assertEqual(result["code"], "sh.600519")
        self.assertEqual(tool.calls, [])
        self.assertTrue(gateway.events[-1]["cache_hit"])

    async def test_value_agent_can_query_explicitly_allowed_peer(self):
        tool = FakeMCPTool(
            [success({"code": "sh.601633", "name": "长城汽车"})]
        )
        state = WorkflowState(
            task="比较同行估值",
            symbol="sz.002594",
            allowed_symbols=["sh.601633"],
            data={"persistent_cache_enabled": False},
        )
        gateway = AuditedToolGateway(
            agent_name="value_agent",
            state=state,
            tools=[tool],
            trace_recorder=InMemoryTraceRecorder(),
            execution_logger=FakeExecutionLogger(),
        )

        result = await gateway.wrap_tools()[0].ainvoke({"code": "sh.601633"})

        self.assertEqual(result["code"], "sh.601633")
        self.assertEqual(gateway.events[-1]["scope"], "peer")
        self.assertEqual(len(tool.calls), 1)

    async def test_unlisted_peer_is_rejected_before_tool_execution(self):
        tool = FakeMCPTool(
            [success({"code": "sz.000625", "name": "长安汽车"})]
        )
        state = WorkflowState(
            task="比较同行估值",
            symbol="sz.002594",
            allowed_symbols=["sh.601633"],
            data={"persistent_cache_enabled": False},
        )
        gateway = AuditedToolGateway(
            agent_name="value_agent",
            state=state,
            tools=[tool],
            trace_recorder=InMemoryTraceRecorder(),
            execution_logger=FakeExecutionLogger(),
        )

        response = await gateway.wrap_tools()[0].ainvoke({"code": "sz.000625"})

        self.assertEqual(tool.calls, [])
        self.assertIn("rejected before execution", str(response))
        self.assertEqual(
            gateway.events[-1]["issues"][0]["code"], "PEER_SYMBOL_NOT_ALLOWED"
        )

    async def test_non_value_agent_cannot_query_peer(self):
        tool = FakeMCPTool(
            [success({"code": "sh.601633", "name": "长城汽车"})]
        )
        state = WorkflowState(
            task="基本面分析",
            symbol="sz.002594",
            allowed_symbols=["sh.601633"],
            data={"persistent_cache_enabled": False},
        )
        gateway = AuditedToolGateway(
            agent_name="fundamental_agent",
            state=state,
            tools=[tool],
            trace_recorder=InMemoryTraceRecorder(),
            execution_logger=FakeExecutionLogger(),
        )

        await gateway.wrap_tools()[0].ainvoke({"code": "sh.601633"})

        self.assertEqual(tool.calls, [])
        self.assertEqual(
            gateway.events[-1]["issues"][0]["code"], "SYMBOL_SCOPE_VIOLATION"
        )


class CanonicalWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_data_only_graph_returns_structured_result_without_summary(self):
        async def fake_agent(state):
            content = (
                "基于复权日线数据，价格趋势与成交量结构显示当前处于震荡阶段。"
                "短期均线尚未形成明确多头排列，支撑与阻力需要结合后续成交量确认。"
                "以上结论仅对应分析基准日前已验证的行情数据，不外推未发生的走势。"
            )
            return {
                "data": {"technical_analysis": content},
                "analysis_results": {
                    "technical": AnalysisResult(
                        agent_id="technical_agent",
                        analysis_type="technical",
                        content=content,
                        evidence=[{
                            "tool": "get_historical_k_data",
                            "call_id": "call-1",
                            "raw_data_hash": "sha256:data",
                            "quality_score": 1.0,
                        }],
                        tool_calls=[{
                            "tool": "get_historical_k_data",
                            "call_id": "call-1",
                            "ok": True,
                            "scope": "target",
                        }],
                    )
                },
            }

        original = workflow.ANALYSIS_NODES["technical"]
        workflow.ANALYSIS_NODES["technical"] = ("technical_analyst", fake_agent)
        trace_dir = Path("logs") / f"test_unified_workflow_{uuid4().hex}"
        try:
            state = await workflow.run_financial_workflow(
                WorkflowState(task="技术分析", data={"trace_dir": str(trace_dir)}),
                ["technical"],
                data_only=True,
            )
        finally:
            workflow.ANALYSIS_NODES["technical"] = original
            shutil.rmtree(trace_dir, ignore_errors=True)

        self.assertTrue(state.analysis_results["technical"].success)
        self.assertNotIn("final_report", state.data)
        self.assertEqual(state.status.value, "completed")

    async def test_failed_quality_gate_produces_no_report(self):
        async def placeholder_agent(state):
            return {
                "analysis_results": {
                    "event": AnalysisResult(
                        agent_id="event_agent",
                        analysis_type="event",
                        content="Sorry, need more steps to process this request.",
                    )
                }
            }

        def valid_agent(analysis_type, tool_names):
            async def run(state):
                content = (
                    f"{analysis_type}分析严格基于已验证数据，给出趋势、风险和观察条件。"
                    "所有结论都限定在分析基准日前，且不会把其他证券数据写成目标证券事实。"
                ) * 2
                return {
                    "analysis_results": {
                        analysis_type: AnalysisResult(
                            agent_id=f"{analysis_type}_agent",
                            analysis_type=analysis_type,
                            content=content,
                            evidence=[{
                                "tool": tool_name,
                                "call_id": f"call-{analysis_type}-{index}",
                                "raw_data_hash": f"sha256:{analysis_type}-{index}",
                                "quality_score": 1.0,
                                "record_count": 5 if analysis_type == "event" else 1,
                            } for index, tool_name in enumerate(tool_names)],
                            tool_calls=[{
                                "tool": tool_name,
                                "call_id": f"call-{analysis_type}-{index}",
                                "ok": True,
                                "scope": "target",
                            } for index, tool_name in enumerate(tool_names)],
                        )
                    }
                }
            return run

        original = {
            key: workflow.ANALYSIS_NODES[key] for key in workflow.ANALYSIS_NODES
        }
        workflow.ANALYSIS_NODES["event"] = ("event_analyst", placeholder_agent)
        workflow.ANALYSIS_NODES["technical"] = (
            "technical_analyst",
            valid_agent("technical", ["get_historical_k_data"]),
        )
        workflow.ANALYSIS_NODES["fundamental"] = (
            "fundamental_analyst",
            valid_agent("fundamental", ["get_stock_basic_info", "get_profit_data"]),
        )
        async def placeholder_value_agent(state):
            return {
                "analysis_results": {
                    "value": AnalysisResult(
                        agent_id="value_agent",
                        analysis_type="value",
                        content="Sorry, need more steps to process this request.",
                    )
                }
            }

        workflow.ANALYSIS_NODES["value"] = (
            "value_analyst",
            placeholder_value_agent,
        )
        trace_dir = Path("logs") / f"test_quality_stop_{uuid4().hex}"
        risk = AsyncMock()
        decision = AsyncMock()
        try:
            with (
                patch.object(workflow, "risk_review_agent", risk),
                patch.object(workflow, "decision_agent", decision),
            ):
                state = await workflow.run_financial_workflow(
                    WorkflowState(
                        task="综合分析",
                        data={"trace_dir": str(trace_dir)},
                    ),
                    ["fundamental", "technical", "value", "event"],
                    data_only=False,
                )
        finally:
            workflow.ANALYSIS_NODES.update(original)
            shutil.rmtree(trace_dir, ignore_errors=True)

        self.assertFalse(state.output["quality_gate"]["passed"])
        self.assertNotIn("final_report", state.data)
        self.assertEqual(state.status.value, "failed")
        risk.assert_not_awaited()
        decision.assert_not_awaited()

    async def test_passed_gate_runs_risk_before_decision(self):
        def specialist(analysis_type, tool_names):
            async def run(state):
                content = (
                    f"{analysis_type}分析严格基于已验证数据，给出可复核的趋势、风险和观察条件。"
                    "结论不使用分析基准日之后的信息，也不把其他证券数据写成目标证券事实。"
                ) * 2
                return {
                    "analysis_results": {
                        analysis_type: AnalysisResult(
                            agent_id=f"{analysis_type}_agent",
                            analysis_type=analysis_type,
                            content=content,
                            evidence=[{
                                "tool": tool_name,
                                "call_id": f"call-{analysis_type}-{index}",
                                "raw_data_hash": f"sha256:{analysis_type}-{index}",
                                "quality_score": 1.0,
                                "record_count": 5 if analysis_type == "event" else 1,
                            } for index, tool_name in enumerate(tool_names)],
                            tool_calls=[{
                                "tool": tool_name,
                                "call_id": f"call-{analysis_type}-{index}",
                                "ok": True,
                                "scope": "target",
                            } for index, tool_name in enumerate(tool_names)],
                        )
                    }
                }
            return run

        async def fake_risk(state):
            self.assertIn("evidence_registry", state.data)
            return {
                "data": {"risk_review": "风险审查已完成。" * 30},
                "analysis_results": {
                    "risk_review": AnalysisResult(
                        agent_id="risk_review_agent",
                        analysis_type="risk_review",
                        content="风险审查已完成。" * 30,
                        success=True,
                    )
                },
            }

        async def fake_decision(state):
            self.assertIn("risk_review", state.analysis_results)
            return {
                "data": {
                    "final_report": "# 有证据约束的决策报告\n" + "有效内容。" * 80,
                    "report_path": "reports/test.md",
                    "decision_status": "completed",
                }
            }

        originals = {
            key: workflow.ANALYSIS_NODES[key] for key in workflow.ANALYSIS_NODES
        }
        workflow.ANALYSIS_NODES["fundamental"] = (
            "fundamental_analyst",
            specialist("fundamental", ["get_stock_basic_info", "get_profit_data"]),
        )
        workflow.ANALYSIS_NODES["technical"] = (
            "technical_analyst",
            specialist("technical", ["get_historical_k_data"]),
        )
        workflow.ANALYSIS_NODES["value"] = (
            "value_analyst",
            specialist("value", ["get_historical_k_data", "get_profit_data"]),
        )
        workflow.ANALYSIS_NODES["event"] = (
            "event_analyst",
            specialist(
                "event", ["get_official_announcements", "get_financial_news"]
            ),
        )
        trace_dir = Path("logs") / f"test_full_gate_{uuid4().hex}"
        try:
            with (
                patch.object(workflow, "risk_review_agent", fake_risk),
                patch.object(workflow, "decision_agent", fake_decision),
            ):
                state = await workflow.run_financial_workflow(
                    WorkflowState(
                        task="综合分析",
                        data={"trace_dir": str(trace_dir)},
                    ),
                    ["fundamental", "technical", "value", "event"],
                    data_only=False,
                )
        finally:
            workflow.ANALYSIS_NODES.update(originals)
            shutil.rmtree(trace_dir, ignore_errors=True)

        self.assertTrue(state.output["quality_gate"]["passed"])
        self.assertIn("risk_review", state.analysis_results)
        self.assertTrue(state.data["final_report"])
        self.assertEqual(state.status.value, "completed")


if __name__ == "__main__":
    unittest.main()
