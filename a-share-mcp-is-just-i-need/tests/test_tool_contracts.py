import asyncio
import inspect
import json
import unittest

from mcp.server.fastmcp import FastMCP

from src.tools.contracts import (
    Quarter,
    StockCode,
    ToolEnvelope,
    ToolExecutionError,
    contract_tool,
)


class FakeApp:
    def __init__(self):
        self.registered = {}

    def tool(self):
        def register(func):
            self.registered[func.__name__] = func
            return func

        return register


class ToolContractTests(unittest.TestCase):
    def setUp(self):
        self.app = FakeApp()

        @contract_tool(self.app)
        def sample_tool(code: StockCode, quarter: Quarter) -> str:
            return f"{code}-Q{quarter}"

        @contract_tool(self.app)
        def failing_tool() -> str:
            raise ToolExecutionError(
                "UPSTREAM_UNAVAILABLE",
                "The upstream service is unavailable.",
                retryable=True,
            )

    def test_valid_call_uses_success_envelope(self):
        result = self.app.registered["sample_tool"](
            code="sh.600519",
            quarter=2,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["content"], "sh.600519-Q2")
        self.assertIsNone(result["error"])
        self.assertEqual(result["meta"]["tool"], "sample_tool")
        json.dumps(result, ensure_ascii=False)

    def test_invalid_market_and_quarter_use_validation_envelope(self):
        result = self.app.registered["sample_tool"](
            code="hk.00700",
            quarter=5,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "VALIDATION_ERROR")
        self.assertFalse(result["error"]["retryable"])
        self.assertGreaterEqual(len(result["error"]["details"]["issues"]), 2)

    def test_unknown_argument_is_rejected(self):
        result = self.app.registered["sample_tool"](
            code="sz.300750",
            quarter=1,
            unsupported=True,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "VALIDATION_ERROR")

    def test_declared_failure_uses_stable_error_envelope(self):
        result = self.app.registered["failing_tool"]()

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "UPSTREAM_UNAVAILABLE")
        self.assertTrue(result["error"]["retryable"])

    def test_flat_signature_is_preserved_for_mcp_schema_generation(self):
        signature = inspect.signature(self.app.registered["sample_tool"])

        self.assertEqual(list(signature.parameters), ["code", "quarter"])
        self.assertIs(signature.return_annotation, ToolEnvelope)

    def test_fastmcp_runtime_returns_envelope_for_invalid_arguments(self):
        app = FastMCP("contract-test")

        @contract_tool(app)
        def strict_tool(code: StockCode, quarter: Quarter) -> str:
            return f"{code}-Q{quarter}"

        result = asyncio.run(
            app._tool_manager.call_tool(
                "strict_tool",
                {"code": "hk.00700", "quarter": 8, "unexpected": True},
            )
        )
        schema = app._tool_manager.get_tool("strict_tool").parameters

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "VALIDATION_ERROR")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["quarter"]["maximum"], 4)


if __name__ == "__main__":
    unittest.main()
