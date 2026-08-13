"""Persistent validated-tool cache with TTL and stale-while-revalidate metadata."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4


CACHE_CONTRACT_VERSION = "financial-tool-cache-v1"


@dataclass(frozen=True)
class CachePolicy:
    ttl_seconds: float
    stale_seconds: float


@dataclass(frozen=True)
class CacheLookup:
    entry: dict[str, Any] | None
    status: str
    age_seconds: float | None = None


def cache_policy(tool_name: str, arguments: dict[str, Any]) -> CachePolicy:
    """Return freshness windows appropriate for the data category."""
    if tool_name in {"get_stock_basic_info", "get_stock_industry"}:
        return CachePolicy(30 * 86400, 180 * 86400)
    if tool_name in {
        "get_profit_data",
        "get_operation_data",
        "get_growth_data",
        "get_balance_data",
        "get_cash_flow_data",
        "get_dupont_data",
        "get_performance_express_report",
        "get_forecast_report",
    }:
        return CachePolicy(7 * 86400, 180 * 86400)
    if tool_name in {"get_historical_k_data", "get_mootdx_bars"}:
        end_date = str(arguments.get("end_date") or "")[:10]
        historical = bool(end_date and end_date < date.today().isoformat())
        return (
            CachePolicy(30 * 86400, 365 * 86400)
            if historical
            else CachePolicy(6 * 3600, 7 * 86400)
        )
    if tool_name in {"get_financial_news", "get_official_announcements"}:
        return CachePolicy(15 * 60, 6 * 3600)
    if tool_name in {"get_tencent_quote", "get_eastmoney_signals"}:
        return CachePolicy(60, 15 * 60)
    if tool_name == "get_latest_trading_date":
        return CachePolicy(10 * 60, 6 * 3600)
    return CachePolicy(30 * 60, 24 * 3600)


def persistent_cache_key(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    provider: str = "mcp",
    contract_version: str = CACHE_CONTRACT_VERSION,
) -> str:
    payload = json.dumps(
        {
            "tool": tool_name,
            "arguments": arguments,
            "provider": provider,
            "contract_version": contract_version,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PersistentToolCache:
    """Small file-per-entry cache; only validated tool results are stored."""

    def __init__(self, directory: str | Path, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled

    @classmethod
    def from_state(cls, state: Any) -> "PersistentToolCache":
        data = getattr(state, "data", {}) or {}
        enabled = bool(data.get("persistent_cache_enabled", True))
        directory = data.get("persistent_cache_dir") or os.getenv(
            "FINANCIAL_CACHE_DIR", ".cache/financial_mcp"
        )
        return cls(directory, enabled=enabled)

    def _path(
        self, tool_name: str, arguments: dict[str, Any], provider: str = "mcp"
    ) -> Path:
        key = persistent_cache_key(tool_name, arguments, provider=provider)
        return self.directory / f"{key}.json"

    def lookup(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        provider: str = "mcp",
        allow_stale: bool = True,
        now: float | None = None,
    ) -> CacheLookup:
        if not self.enabled:
            return CacheLookup(None, "disabled")
        path = self._path(tool_name, arguments, provider)
        if not path.exists():
            return CacheLookup(None, "miss")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return CacheLookup(None, "corrupt")
        if payload.get("contract_version") != CACHE_CONTRACT_VERSION:
            return CacheLookup(None, "version_miss")
        age = max(0.0, (now or time.time()) - float(payload.get("stored_at", 0)))
        policy = cache_policy(tool_name, arguments)
        if age <= policy.ttl_seconds:
            return CacheLookup(payload["entry"], "fresh", age)
        if allow_stale and age <= policy.stale_seconds:
            return CacheLookup(payload["entry"], "stale", age)
        return CacheLookup(None, "expired", age)

    def store(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        entry: dict[str, Any],
        *,
        provider: str = "mcp",
        stored_at: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "contract_version": CACHE_CONTRACT_VERSION,
            "tool": tool_name,
            "arguments": arguments,
            "provider_key": provider,
            "stored_at": stored_at or time.time(),
            "entry": entry,
        }
        target = self._path(tool_name, arguments, provider)
        temporary = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def find_historical_prefix(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        provider: str = "mcp",
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Find a cached range covering the requested start but ending earlier."""
        if not self.enabled or tool_name != "get_historical_k_data":
            return None
        requested_start = str(arguments.get("start_date") or "")
        requested_end = str(arguments.get("end_date") or "")
        if not requested_start or not requested_end or not self.directory.exists():
            return None
        ignored = {"start_date", "end_date"}
        requested_fixed = {k: v for k, v in arguments.items() if k not in ignored}
        candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for path in self.directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            cached_args = payload.get("arguments") or {}
            if payload.get("tool") != tool_name or payload.get("provider_key") != provider:
                continue
            age = max(0.0, time.time() - float(payload.get("stored_at", 0)))
            if age > cache_policy(tool_name, cached_args).stale_seconds:
                continue
            if {k: v for k, v in cached_args.items() if k not in ignored} != requested_fixed:
                continue
            cached_start = str(cached_args.get("start_date") or "")
            cached_end = str(cached_args.get("end_date") or "")
            if cached_start <= requested_start and cached_end < requested_end:
                candidates.append((cached_end, cached_args, payload.get("entry") or {}))
        if not candidates:
            return None
        _, cached_args, entry = max(candidates, key=lambda item: item[0])
        return cached_args, entry
