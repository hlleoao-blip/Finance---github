import json
import shutil
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import src.agent_loop_cli as agent_loop_cli
from src.agent_loop.contracts import PlanStep
from src.agent_loop.executor import MCPToolExecutor
from src.agent_loop.state import AgentLoopState
from src.agent_loop.tracing import JSONLTraceRecorder
from src.agent_loop.validator import FinancialDataValidator
from src.agent_loop_cli import normalize_a_share_symbol
from src.utils.listing_verification import ListingCheckResult, ListingStatus


class FakeTool:
    def __init__(self, name, responses):
        self.name = name
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, arguments):
        self.calls.append(arguments)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def success(data):
    return {
        "ok": True,
        "data": {"content": data},
        "error": None,
        "meta": {"tool": "fake", "provider": "test-provider"},
    }


class ExecutorAndValidatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_executor_normalizes_result_and_adds_provenance(self):
        step = PlanStep(
            id="prices",
            objective="load prices",
            candidate_tools=["prices"],
            arguments={"code": "sh.600519"},
        )
        tool = FakeTool("prices", [success({"code": "sh.600519", "close": 10})])
        state = AgentLoopState(task="分析价格", symbol="sh.600519")

        result = await MCPToolExecutor([tool]).execute(step, state)

        self.assertTrue(result.ok)
        self.assertEqual(result.data["code"], "sh.600519")
        self.assertEqual(result.meta.run_id, state.run_id)
        self.assertTrue(result.meta.request_hash.startswith("sha256:"))

    async def test_missing_tool_fails_closed(self):
        step = PlanStep(
            id="profit", objective="load profit", candidate_tools=["missing"], arguments={}
        )
        result = await MCPToolExecutor([]).execute(step, AgentLoopState(task="盈利"))

        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "TOOL_UNAVAILABLE")

    async def test_invalid_ohlc_is_rejected(self):
        step = PlanStep(
            id="prices", objective="load prices", candidate_tools=["prices"], arguments={}
        )
        tool = FakeTool(
            "prices", [success([{"open": 20, "high": 12, "low": 9, "close": 11}])]
        )
        state = AgentLoopState(task="分析价格")
        result = await MCPToolExecutor([tool]).execute(step, state)

        report = await FinancialDataValidator().validate(step, result, state)

        self.assertFalse(report.passed)
        self.assertEqual(report.issues[0].code, "INVALID_OHLC_RELATION")

    async def test_data_after_as_of_date_is_rejected(self):
        step = PlanStep(
            id="prices",
            objective="load prices",
            candidate_tools=["prices"],
            arguments={},
        )
        tool = FakeTool(
            "prices",
            [success({"date": "2026-08-10", "close": 10})],
        )
        state = AgentLoopState(task="历史分析", as_of="2026-08-09")
        result = await MCPToolExecutor([tool]).execute(step, state)

        report = await FinancialDataValidator().validate(step, result, state)

        self.assertFalse(report.passed)
        self.assertIn("FUTURE_DATED_DATA", {issue.code for issue in report.issues})

    async def test_jsonl_trace_sequence_is_unique_and_contiguous(self):
        temp_dir = Path("logs") / f"test_agent_trace_{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        state = AgentLoopState(task="trace test")
        recorder = JSONLTraceRecorder(temp_dir)
        try:
            recorder.record(state, "first", {"value": 1})
            recorder.record(state, "second", {"value": 2})
            recorder.save_state(state)
            events = [
                json.loads(line)
                for line in (temp_dir / state.run_id / "trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual([event["sequence"] for event in events], [1, 2])


class CliCompatibilityTests(unittest.TestCase):
    def test_symbol_normalization(self):
        self.assertEqual(normalize_a_share_symbol(None, "分析600519"), "sh.600519")
        self.assertEqual(normalize_a_share_symbol("300750", "分析股票"), "sz.300750")


class CliPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_listing_never_starts_workflow(self):
        args = Namespace(
            command="帮我看一下字节跳动这支股票值得购买吗",
            symbol=None,
            company="字节跳动",
        )
        rejected = ListingCheckResult(
            status=ListingStatus.NOT_SUPPORTED,
            company_name="字节跳动",
            message="未在当前 A 股证券主数据中找到字节跳动。",
        )

        with (
            patch.object(
                agent_loop_cli,
                "verify_a_share_listing",
                AsyncMock(return_value=rejected),
            ),
            patch.object(
                agent_loop_cli,
                "run_financial_workflow",
                AsyncMock(),
            ) as workflow,
        ):
            with self.assertRaisesRegex(ValueError, "字节跳动"):
                await agent_loop_cli.run_from_args(args)

        workflow.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
