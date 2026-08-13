import unittest
from types import SimpleNamespace

from src.tools.tool_policy import (
    AGENT_TOOL_ALLOWLISTS,
    ToolPolicyError,
    filter_tools_for_agent,
)


def tool(name):
    return SimpleNamespace(name=name)


class ToolPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tools = [
            tool("get_historical_k_data"),
            tool("get_profit_data"),
            tool("get_stock_analysis"),
            tool("crawl_news"),
            tool("get_mootdx_bars"),
            tool("get_tencent_quote"),
            tool("get_eastmoney_signals"),
            tool("get_official_announcements"),
            tool("get_financial_news"),
        ]

    def test_technical_agent_cannot_see_news_or_meta_analysis(self):
        selected = filter_tools_for_agent("technical_agent", self.tools)

        self.assertEqual(
            [item.name for item in selected],
            [
                "get_historical_k_data",
                "get_mootdx_bars",
                "get_tencent_quote",
                "get_eastmoney_signals",
            ],
        )

    def test_event_agent_can_only_see_event_source(self):
        selected = filter_tools_for_agent("event_agent", self.tools)

        self.assertEqual(
            [item.name for item in selected],
            ["get_official_announcements", "get_financial_news"],
        )

    def test_fundamental_agent_cannot_see_technical_or_news_tools(self):
        selected = filter_tools_for_agent("fundamental_agent", self.tools)

        self.assertEqual([item.name for item in selected], ["get_profit_data"])

    def test_every_runtime_agent_has_a_policy(self):
        self.assertEqual(
            set(AGENT_TOOL_ALLOWLISTS),
            {
                "fundamental_agent",
                "technical_agent",
                "value_agent",
                "event_agent",
            },
        )

    def test_unknown_agent_fails_closed(self):
        with self.assertRaises(ToolPolicyError):
            filter_tools_for_agent("unregistered_agent", self.tools)

    def test_empty_policy_result_fails_closed(self):
        with self.assertRaises(ToolPolicyError):
            filter_tools_for_agent("event_agent", [tool("get_profit_data")])


if __name__ == "__main__":
    unittest.main()
