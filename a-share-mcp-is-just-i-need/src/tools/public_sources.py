"""MCP registrations for independent market, signal, announcement and news sources."""

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from requests import exceptions as request_exceptions

from src.event_data_sources import FinancialNewsSource, OfficialAnnouncementSource
from src.market_data_sources import (
    EastmoneySignalSource,
    MootdxMarketSource,
    PublicDataSourceError,
    TencentMarketSource,
)
from src.tools.contracts import (
    CompanyName,
    ISODate,
    StockCode,
    ToolExecutionError,
    TopK,
    contract_tool,
    tool_success,
)


MootdxFrequency = Literal["1", "5", "15", "30", "60", "d"]


def _translate_provider_error(error: Exception) -> ToolExecutionError:
    causes: list[BaseException] = []
    observed: BaseException | None = error
    while observed is not None and observed not in causes:
        causes.append(observed)
        observed = observed.__cause__ or observed.__context__
    if any(isinstance(item, request_exceptions.Timeout) for item in causes):
        provider = getattr(error, "provider", "public_source")
        return ToolExecutionError(
            "TOOL_TIMEOUT",
            str(error),
            retryable=True,
            details={"provider": provider},
        )
    if any(isinstance(item, request_exceptions.ProxyError) for item in causes):
        provider = getattr(error, "provider", "public_source")
        return ToolExecutionError(
            "UPSTREAM_ERROR",
            str(error),
            retryable=False,
            details={"provider": provider, "exception_type": "ProxyError"},
        )
    if isinstance(error, PublicDataSourceError):
        inferred_code = error.code
        if inferred_code is None and any(
            token in str(error).lower()
            for token in ("no data", "no quote", "no bars", "returned empty")
        ):
            inferred_code = "NO_DATA"
        return ToolExecutionError(
            inferred_code
            or ("UPSTREAM_UNAVAILABLE" if error.retryable else "PROVIDER_NOT_CONFIGURED"),
            str(error),
            retryable=error.retryable,
            details={"provider": error.provider},
        )
    if isinstance(error, ValueError):
        return ToolExecutionError("INVALID_ARGUMENT", str(error))
    return ToolExecutionError(
        "UPSTREAM_ERROR",
        "Public data provider failed unexpectedly.",
        retryable=True,
        details={"exception_type": type(error).__name__},
    )


def register_public_source_tools(
    app: FastMCP,
    *,
    mootdx_source: MootdxMarketSource | None = None,
    tencent_source: TencentMarketSource | None = None,
    eastmoney_source: EastmoneySignalSource | None = None,
    announcement_source: OfficialAnnouncementSource | None = None,
    news_source: FinancialNewsSource | None = None,
) -> None:
    """Register all non-Baostock public providers with injectable adapters."""
    mootdx = mootdx_source or MootdxMarketSource()
    tencent = tencent_source or TencentMarketSource()
    eastmoney = eastmoney_source or EastmoneySignalSource()
    announcements = announcement_source or OfficialAnnouncementSource()
    news = news_source or FinancialNewsSource()

    @contract_tool(app)
    def get_mootdx_bars(
        code: StockCode,
        frequency: MootdxFrequency = "d",
        count: int = 120,
    ) -> dict[str, Any]:
        """获取通达信(mootdx)在线行情K线，作为Baostock行情的独立校验源。

        count 必须在 5 到 800 之间；返回结构化 OHLCV 记录并保留目标证券代码。
        """
        if not 5 <= int(count) <= 800:
            raise ToolExecutionError(
                "INVALID_ARGUMENT", "count must be between 5 and 800"
            )
        try:
            value = mootdx.bars(code, frequency, count)
        except Exception as error:
            raise _translate_provider_error(error) from error
        return tool_success("get_mootdx_bars", value, provider="mootdx")

    @contract_tool(app)
    def get_tencent_quote(code: StockCode) -> dict[str, Any]:
        """获取腾讯行情的实时/最近交易快照，用于价格、估值与成交数据交叉验证。"""
        try:
            value = tencent.quote(code)
        except Exception as error:
            raise _translate_provider_error(error) from error
        return tool_success("get_tencent_quote", value, provider="tencent")

    @contract_tool(app)
    def get_eastmoney_signals(
        code: StockCode,
        flow_days: int = 10,
    ) -> dict[str, Any]:
        """获取东方财富行情、估值、换手与主力资金流特色信号。

        price_above_previous_close 与 latest_main_flow_direction 是透明规则派生信号，
        原始快照和逐日资金流一并返回，不能替代财报或官方公告证据。
        """
        if not 1 <= int(flow_days) <= 60:
            raise ToolExecutionError(
                "INVALID_ARGUMENT", "flow_days must be between 1 and 60"
            )
        try:
            value = eastmoney.signals(code, flow_days)
        except Exception as error:
            raise _translate_provider_error(error) from error
        return tool_success("get_eastmoney_signals", value, provider="eastmoney")

    @contract_tool(app)
    def get_official_announcements(
        code: StockCode,
        start_date: ISODate,
        end_date: ISODate,
        top_k: TopK = 20,
    ) -> dict[str, Any]:
        """查询巨潮资讯并与对应上交所/深交所公告交叉取数。

        返回公告日期、标题、原始链接和来源；某一官方源失败会显式记录，只有所有
        可用官方源均无有效记录时才失败。
        """
        if start_date > end_date:
            raise ToolExecutionError(
                "INVALID_ARGUMENT", "start_date must not be after end_date"
            )
        try:
            value = announcements.announcements(code, start_date, end_date, top_k)
        except Exception as error:
            raise _translate_provider_error(error) from error
        return tool_success(
            "get_official_announcements",
            value,
            provider="official_announcements",
            source_chain=value.get("source_chain"),
            source_failures=value.get("source_failures"),
        )

    @contract_tool(app)
    def get_financial_news(
        code: StockCode,
        company_name: CompanyName,
        start_date: ISODate,
        end_date: ISODate,
        top_k: TopK = 10,
    ) -> dict[str, Any]:
        """按公司查询财联社新闻；数量不足或不可用时才由新浪财经补齐。"""
        if start_date > end_date:
            raise ToolExecutionError(
                "INVALID_ARGUMENT", "start_date must not be after end_date"
            )
        try:
            value = news.news(code, company_name, start_date, end_date, top_k)
        except Exception as error:
            raise _translate_provider_error(error) from error
        return tool_success(
            "get_financial_news",
            value,
            provider="cls_sina",
            source_chain=value.get("source_chain"),
            source_failures=value.get("source_failures"),
        )
