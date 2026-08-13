"""Independent public-market adapters used by the MCP tools.

The adapters deliberately return normalized Python objects rather than formatted
text.  This keeps symbol/date validation deterministic in the workflow gateway
and makes provider provenance visible to downstream reviewers.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

import requests


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


class PublicDataSourceError(RuntimeError):
    """Stable provider error which can be translated at the MCP boundary."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        retryable: bool = True,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.code = code


class HTTPSession(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...


def normalize_a_share_code(code: str) -> tuple[str, str, str]:
    """Return (canonical, digits, exchange) for a Baostock A-share code."""
    match = re.fullmatch(r"(sh|sz)\.(\d{6})", str(code).strip().lower())
    if not match:
        raise ValueError("code must use the sh.600519 or sz.000001 format")
    exchange, digits = match.groups()
    return f"{exchange}.{digits}", digits, exchange


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            return _clean_scalar(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _scaled(value: Any, factor: float = 100.0) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return round(float(value) / factor, 6)
    except (TypeError, ValueError):
        return None


def _response_json(response: Any, provider: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as error:
        raise PublicDataSourceError(
            provider, f"{provider} returned an invalid HTTP/JSON response"
        ) from error
    if not isinstance(payload, dict):
        raise PublicDataSourceError(provider, f"{provider} returned a non-object payload")
    return payload


@dataclass
class MootdxMarketSource:
    """Online TongDaXin bars through mootdx, imported lazily."""

    client: Any | None = None
    server: tuple[str, int] | None = None
    timeout: int = 3
    max_servers: int = 3

    FREQUENCIES = {
        "5": 0,
        "15": 1,
        "30": 2,
        "60": 3,
        "d": 9,
        "1": 8,
    }

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from mootdx.quotes import Quotes
            from mootdx.consts import HQ_HOSTS
        except ImportError as error:
            raise PublicDataSourceError(
                "mootdx",
                "mootdx is not installed in the MCP server environment",
                retryable=False,
            ) from error
        # mootdx 0.11.7 can leave BESTIP.HQ as an empty string after first-time
        # discovery fails, which makes Quotes.factory() crash while unpacking the
        # server.  Passing an explicit upstream host avoids that hidden global
        # configuration dependency while still using mootdx for transport/parsing.
        selected_server = self.server or (str(HQ_HOSTS[0][1]), int(HQ_HOSTS[0][2]))
        self.client = Quotes.factory(
            market="std",
            server=selected_server,
            multithread=True,
            heartbeat=False,
            timeout=self.timeout,
        )
        return self.client

    def bars(self, code: str, frequency: str = "d", count: int = 120) -> dict[str, Any]:
        canonical, digits, _exchange = normalize_a_share_code(code)
        if frequency not in self.FREQUENCIES:
            raise ValueError(f"unsupported mootdx frequency: {frequency}")
        injected_client = self.client is not None
        if injected_client or self.server is not None:
            candidates: list[tuple[str, int] | None] = [None]
        else:
            try:
                from mootdx.consts import HQ_HOSTS
            except ImportError as error:
                raise PublicDataSourceError(
                    "mootdx",
                    "mootdx is not installed in the MCP server environment",
                    retryable=False,
                ) from error
            candidates = [
                (str(host), int(port))
                for _name, host, port in HQ_HOSTS[: self.max_servers]
            ]

        frame = None
        last_error: Exception | None = None
        for candidate_server in candidates:
            client = None
            if candidate_server is not None:
                self.server = candidate_server
                self.client = None
            try:
                client = self._client()
                candidate = client.bars(
                    symbol=digits,
                    frequency=self.FREQUENCIES[frequency],
                    offset=int(count),
                )
                if candidate is not None and not getattr(candidate, "empty", True):
                    frame = candidate
                    break
            except Exception as error:
                last_error = error
                if not injected_client and client is not None and hasattr(client, "close"):
                    client.close()
                self.client = None
        if frame is None and last_error is not None:
            raise PublicDataSourceError("mootdx", "failed to retrieve TongDaXin bars") from last_error
        if frame is None or getattr(frame, "empty", True):
            raise PublicDataSourceError("mootdx", "TongDaXin returned no bars")

        normalized = frame.reset_index()
        records = [
            {str(key): _clean_scalar(value) for key, value in row.items()}
            for row in normalized.to_dict(orient="records")
        ]
        for row in records:
            row["code"] = canonical
            if "datetime" in row and row["datetime"] is not None:
                row["datetime"] = str(row["datetime"])
            if "date" in row and row["date"] is not None:
                row["date"] = str(row["date"])
        return {
            "symbol": canonical,
            "frequency": frequency,
            "adjustment": "provider_default",
            "records": records,
        }


@dataclass
class TencentMarketSource:
    """Tencent quote adapter used as an independent market snapshot."""

    session: HTTPSession | None = None
    timeout: float = 10.0

    QUOTE_URL = "https://qt.gtimg.cn/q={symbol}"

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def quote(self, code: str) -> dict[str, Any]:
        canonical, digits, exchange = normalize_a_share_code(code)
        try:
            response = self.session.get(
                self.QUOTE_URL.format(symbol=f"{exchange}{digits}"),
                headers={**DEFAULT_HEADERS, "Referer": "https://gu.qq.com/"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            response.encoding = "gbk"
            text = response.text
        except Exception as error:
            raise PublicDataSourceError("tencent", "failed to retrieve Tencent quote") from error

        match = re.search(r'="(?P<payload>.*?)";?\s*$', text.strip(), re.S)
        if not match:
            raise PublicDataSourceError("tencent", "Tencent quote payload is malformed")
        fields = match.group("payload").split("~")
        if len(fields) < 49 or not fields[3]:
            raise PublicDataSourceError("tencent", "Tencent quote payload is incomplete")

        timestamp = fields[30]
        observed_at = None
        if re.fullmatch(r"\d{14}", timestamp or ""):
            observed_at = (
                f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}T"
                f"{timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}+08:00"
            )
        return {
            "symbol": canonical,
            "name": fields[1],
            "datetime": observed_at,
            "price": _scaled(fields[3], 1),
            "previous_close": _scaled(fields[4], 1),
            "open": _scaled(fields[5], 1),
            "volume_lots": _scaled(fields[6], 1),
            "change": _scaled(fields[31], 1),
            "change_pct": _scaled(fields[32], 1),
            "high": _scaled(fields[33], 1),
            "low": _scaled(fields[34], 1),
            "amount_wan": _scaled(fields[37], 1),
            "turnover_pct": _scaled(fields[38], 1),
            "pe_ttm": _scaled(fields[39], 1),
            "amplitude_pct": _scaled(fields[43], 1),
            "float_market_cap_yi": _scaled(fields[44], 1),
            "total_market_cap_yi": _scaled(fields[45], 1),
            "pb": _scaled(fields[46], 1),
            "limit_up": _scaled(fields[47], 1),
            "limit_down": _scaled(fields[48], 1),
        }


@dataclass
class EastmoneySignalSource:
    """Eastmoney quote/valuation and northbound-style capital-flow signals."""

    session: HTTPSession | None = None
    timeout: float = 10.0

    QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
    FLOW_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    @staticmethod
    def _secid(exchange: str, digits: str) -> str:
        return f"{1 if exchange == 'sh' else 0}.{digits}"

    def signals(self, code: str, flow_days: int = 10) -> dict[str, Any]:
        canonical, digits, exchange = normalize_a_share_code(code)
        secid = self._secid(exchange, digits)
        quote = _response_json(
            self.session.get(
                self.QUOTE_URL,
                params={
                    "secid": secid,
                    "fields": (
                        "f43,f44,f45,f46,f47,f48,f57,f58,f60,f116,f117,"
                        "f162,f167,f168,f170"
                    ),
                },
                headers={**DEFAULT_HEADERS, "Referer": "https://quote.eastmoney.com/"},
                timeout=self.timeout,
            ),
            "eastmoney",
        ).get("data")
        if not isinstance(quote, dict) or not quote:
            raise PublicDataSourceError("eastmoney", "Eastmoney returned no quote signals")

        flow_records: list[dict[str, Any]] = []
        try:
            flow_payload = _response_json(
                self.session.get(
                    self.FLOW_URL,
                    params={
                        "secid": secid,
                        "lmt": max(1, min(int(flow_days), 60)),
                        "klt": 101,
                        "fields1": "f1,f2,f3,f7",
                        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
                    },
                    headers={**DEFAULT_HEADERS, "Referer": "https://data.eastmoney.com/"},
                    timeout=self.timeout,
                ),
                "eastmoney",
            ).get("data") or {}
            for raw in flow_payload.get("klines") or []:
                values = str(raw).split(",")
                if len(values) < 6:
                    continue
                flow_records.append(
                    {
                        "date": values[0],
                        "symbol": canonical,
                        "main_net_inflow": _scaled(values[1], 1),
                        "small_net_inflow": _scaled(values[2], 1),
                        "medium_net_inflow": _scaled(values[3], 1),
                        "large_net_inflow": _scaled(values[4], 1),
                        "super_large_net_inflow": _scaled(values[5], 1),
                        "main_net_inflow_pct": _scaled(values[6], 1) if len(values) > 6 else None,
                    }
                )
        except PublicDataSourceError:
            # The quote/valuation snapshot remains usable if the optional flow
            # endpoint is temporarily unavailable.  Missing flow is explicit.
            flow_records = []

        current = _scaled(quote.get("f43"))
        previous_close = _scaled(quote.get("f60"))
        latest_flow = flow_records[-1] if flow_records else None
        return {
            "symbol": canonical,
            "name": quote.get("f58"),
            "snapshot": {
                "price": current,
                "previous_close": previous_close,
                "change_pct": _scaled(quote.get("f170")),
                "open": _scaled(quote.get("f46")),
                "high": _scaled(quote.get("f44")),
                "low": _scaled(quote.get("f45")),
                "volume": _scaled(quote.get("f47"), 1),
                "amount": _scaled(quote.get("f48"), 1),
                "turnover_pct": _scaled(quote.get("f168")),
                "pe_dynamic": _scaled(quote.get("f162")),
                "pb": _scaled(quote.get("f167")),
                "total_market_cap": _scaled(quote.get("f116"), 1),
                "float_market_cap": _scaled(quote.get("f117"), 1),
            },
            "signals": {
                "price_above_previous_close": (
                    current > previous_close
                    if current is not None and previous_close is not None
                    else None
                ),
                "latest_main_flow_direction": (
                    "inflow"
                    if latest_flow and (latest_flow.get("main_net_inflow") or 0) > 0
                    else "outflow"
                    if latest_flow and (latest_flow.get("main_net_inflow") or 0) < 0
                    else "flat_or_unavailable"
                ),
            },
            "capital_flow": flow_records,
        }
