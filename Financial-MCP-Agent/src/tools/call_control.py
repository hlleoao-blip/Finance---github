"""Run-scoped category budgets and provider circuit breakers."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


BUDGET_CATEGORIES = ("required", "cross_validation", "supplemental")
DEFAULT_BUDGET_SHARES = {
    "required": 0.60,
    "cross_validation": 0.20,
    "supplemental": 0.20,
}

TOOL_PROVIDER_BY_NAME = {
    "get_stock_basic_info": "baostock",
    "get_stock_industry": "baostock",
    "get_latest_trading_date": "baostock",
    "get_historical_k_data": "baostock",
    "get_adjust_factor_data": "baostock",
    "get_profit_data": "baostock",
    "get_operation_data": "baostock",
    "get_growth_data": "baostock",
    "get_balance_data": "baostock",
    "get_cash_flow_data": "baostock",
    "get_dupont_data": "baostock",
    "get_dividend_data": "baostock",
    "get_performance_express_report": "baostock",
    "get_forecast_report": "baostock",
    "get_tencent_quote": "tencent",
    "get_mootdx_bars": "mootdx",
    "get_eastmoney_signals": "eastmoney",
    "get_official_announcements": "official_announcements",
    "get_financial_news": "cls_sina",
}


def allocate_category_budget(
    total: int,
    shares: Mapping[str, float] | None = None,
) -> dict[str, int]:
    """Allocate a total deterministically while preserving the 60/20/20 reserve."""
    total = max(0, int(total))
    configured = dict(DEFAULT_BUDGET_SHARES)
    if shares:
        configured.update(
            {
                key: max(0.0, float(value))
                for key, value in shares.items()
                if key in BUDGET_CATEGORIES
            }
        )
    share_total = sum(configured.values()) or 1.0
    required_exact = total * configured["required"] / share_total
    allocated = {
        "required": min(total, math.ceil(required_exact)),
        "cross_validation": 0,
        "supplemental": 0,
    }
    remainder = total - allocated["required"]
    secondary_total = configured["cross_validation"] + configured["supplemental"]
    if remainder and secondary_total:
        cross_exact = remainder * configured["cross_validation"] / secondary_total
        allocated["cross_validation"] = min(remainder, math.ceil(cross_exact))
        allocated["supplemental"] = remainder - allocated["cross_validation"]
    elif remainder:
        allocated["supplemental"] = remainder
    return allocated


def provider_for_tool(tool_name: str) -> str:
    return TOOL_PROVIDER_BY_NAME.get(tool_name, tool_name)


def is_circuit_failure(error_code: Any, message: str = "") -> bool:
    marker = f"{error_code or ''} {message}".upper()
    return any(
        token in marker
        for token in ("TIMEOUT", "PROXY", "CONNECTION", "NETWORK", "CONNECT_ERROR")
    )


@dataclass
class CategoryBudget:
    limits: dict[str, int]
    used: dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in BUDGET_CATEGORIES}
    )

    def consume(self, category: str) -> bool:
        category = category if category in self.limits else "supplemental"
        if self.used.get(category, 0) >= self.limits.get(category, 0):
            return False
        self.used[category] = self.used.get(category, 0) + 1
        return True

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            key: {
                "limit": self.limits.get(key, 0),
                "used": self.used.get(key, 0),
                "remaining": max(
                    0, self.limits.get(key, 0) - self.used.get(key, 0)
                ),
            }
            for key in BUDGET_CATEGORIES
        }


@dataclass
class RunCallControl:
    circuit_threshold: int = 2
    consecutive_provider_failures: dict[str, int] = field(default_factory=dict)
    open_providers: set[str] = field(default_factory=set)
    blocked_agents: set[str] = field(default_factory=set)
    provider_locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def provider_lock(self, provider: str) -> asyncio.Lock:
        if provider not in self.provider_locks:
            self.provider_locks[provider] = asyncio.Lock()
        return self.provider_locks[provider]

    def provider_is_open(self, provider: str) -> bool:
        return provider in self.open_providers

    def record_provider_result(
        self,
        provider: str,
        *,
        ok: bool,
        error_code: Any = None,
        message: str = "",
    ) -> None:
        if ok:
            self.consecutive_provider_failures[provider] = 0
            return
        if not is_circuit_failure(error_code, message):
            self.consecutive_provider_failures[provider] = 0
            return
        failures = self.consecutive_provider_failures.get(provider, 0) + 1
        self.consecutive_provider_failures[provider] = failures
        if failures >= self.circuit_threshold:
            self.open_providers.add(provider)

    def snapshot(self) -> dict[str, Any]:
        return {
            "threshold": self.circuit_threshold,
            "consecutive_provider_failures": dict(
                self.consecutive_provider_failures
            ),
            "open_providers": sorted(self.open_providers),
            "blocked_agents": sorted(self.blocked_agents),
        }


_RUN_CONTROLS: dict[str, RunCallControl] = {}


def get_run_call_control(run_id: str, *, circuit_threshold: int = 2) -> RunCallControl:
    control = _RUN_CONTROLS.get(run_id)
    if control is None:
        control = RunCallControl(circuit_threshold=max(1, int(circuit_threshold)))
        _RUN_CONTROLS[run_id] = control
    return control


def release_run_call_control(run_id: str) -> None:
    _RUN_CONTROLS.pop(run_id, None)
