"""Code-level MCP tool boundaries for each analysis agent."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


AGENT_TOOL_ALLOWLISTS: dict[str, frozenset[str]] = {
    "fundamental_agent": frozenset(
        {
            "get_latest_trading_date",
            "get_stock_basic_info",
            "get_stock_industry",
            "get_profit_data",
            "get_operation_data",
            "get_growth_data",
            "get_balance_data",
            "get_cash_flow_data",
            "get_dupont_data",
            "get_dividend_data",
            "get_performance_express_report",
            "get_forecast_report",
        }
    ),
    "technical_agent": frozenset(
        {
            "get_latest_trading_date",
            "get_market_analysis_timeframe",
            "get_stock_basic_info",
            "get_historical_k_data",
            "get_adjust_factor_data",
            "get_mootdx_bars",
            "get_tencent_quote",
            "get_eastmoney_signals",
        }
    ),
    "value_agent": frozenset(
        {
            "get_latest_trading_date",
            "get_market_analysis_timeframe",
            "get_stock_basic_info",
            "get_stock_industry",
            "get_historical_k_data",
            "get_profit_data",
            "get_growth_data",
            "get_balance_data",
            "get_cash_flow_data",
            "get_dupont_data",
            "get_dividend_data",
            "get_mootdx_bars",
            "get_tencent_quote",
            "get_eastmoney_signals",
        }
    ),
    "event_agent": frozenset(
        {
            "get_official_announcements",
            "get_financial_news",
        }
    ),
}


class ToolPolicyError(ValueError):
    """Raised when an agent has no declared or usable tool boundary."""


def filter_tools_for_agent(agent_name: str, tools: Iterable[Any]) -> list[Any]:
    """Return only tools explicitly allowed for ``agent_name``.

    This is a security/capability boundary: prompts cannot add tools back after
    this function has removed them.
    """
    if agent_name not in AGENT_TOOL_ALLOWLISTS:
        raise ToolPolicyError(f"No MCP tool policy declared for agent: {agent_name}")

    allowed_names = AGENT_TOOL_ALLOWLISTS[agent_name]
    selected = [tool for tool in tools if getattr(tool, "name", None) in allowed_names]
    if not selected:
        raise ToolPolicyError(
            f"No loaded MCP tools satisfy the policy for agent: {agent_name}"
        )
    return selected
