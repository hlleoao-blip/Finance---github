import unittest

import pandas as pd
from mcp.server.fastmcp import FastMCP

from src.event_data_sources import FinancialNewsSource, OfficialAnnouncementSource
from src.market_data_sources import (
    EastmoneySignalSource,
    MootdxMarketSource,
    PublicDataSourceError,
    TencentMarketSource,
)
from src.tools.public_sources import register_public_source_tools


class FakeResponse:
    def __init__(self, *, payload=None, text="", status_code=200):
        self.payload = payload
        self.text = text
        self.status_code = status_code
        self.encoding = None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class URLSession:
    def __init__(self, responses):
        self.responses = responses

    def get(self, url, **kwargs):
        response = self.responses[url]
        return response() if callable(response) else response

    def post(self, url, **kwargs):
        response = self.responses[url]
        return response() if callable(response) else response


class FakeQuotes:
    def bars(self, **kwargs):
        self.arguments = kwargs
        return pd.DataFrame(
            [{"open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "vol": 1000}]
        )


class MarketSourceTests(unittest.TestCase):
    def test_mootdx_bars_preserve_canonical_symbol(self):
        client = FakeQuotes()
        result = MootdxMarketSource(client=client).bars("sh.600519", "d", 30)

        self.assertEqual(result["symbol"], "sh.600519")
        self.assertEqual(result["records"][0]["code"], "sh.600519")
        self.assertEqual(client.arguments["frequency"], 9)

    def test_tencent_quote_is_normalized(self):
        fields = [""] * 49
        fields[1] = "贵州茅台"
        fields[2] = "600519"
        fields[3] = "1500.00"
        fields[4] = "1490.00"
        fields[5] = "1495.00"
        fields[6] = "12345"
        fields[30] = "20260807150000"
        fields[31] = "10.00"
        fields[32] = "0.67"
        fields[33] = "1510.00"
        fields[34] = "1480.00"
        fields[37] = "987654"
        fields[38] = "0.20"
        fields[39] = "25.50"
        fields[43] = "2.01"
        fields[44] = "18000"
        fields[45] = "19000"
        fields[46] = "8.10"
        fields[47] = "1639.00"
        fields[48] = "1341.00"
        session = URLSession(
            {
                TencentMarketSource.QUOTE_URL.format(symbol="sh600519"): FakeResponse(
                    text='v_sh600519="' + "~".join(fields) + '";'
                )
            }
        )

        result = TencentMarketSource(session=session).quote("sh.600519")

        self.assertEqual(result["symbol"], "sh.600519")
        self.assertEqual(result["price"], 1500.0)
        self.assertEqual(result["datetime"], "2026-08-07T15:00:00+08:00")

    def test_eastmoney_signals_keep_raw_inputs_and_derived_direction(self):
        quote = {
            "data": {
                "f43": 150000,
                "f44": 151000,
                "f45": 148000,
                "f46": 149500,
                "f47": 12345,
                "f48": 999999,
                "f58": "贵州茅台",
                "f60": 149000,
                "f162": 2550,
                "f167": 810,
                "f168": 20,
                "f170": 67,
            }
        }
        flow = {"data": {"klines": ["2026-08-07,100,1,2,3,4,0.5"]}}
        session = URLSession(
            {
                EastmoneySignalSource.QUOTE_URL: FakeResponse(payload=quote),
                EastmoneySignalSource.FLOW_URL: FakeResponse(payload=flow),
            }
        )

        result = EastmoneySignalSource(session=session).signals("sh.600519", 10)

        self.assertEqual(result["snapshot"]["price"], 1500.0)
        self.assertEqual(result["signals"]["latest_main_flow_direction"], "inflow")
        self.assertEqual(result["capital_flow"][0]["symbol"], "sh.600519")


class StubAnnouncements(OfficialAnnouncementSource):
    def __init__(self):
        pass

    def _cninfo(self, code, digits, exchange, start_date, end_date, top_k):
        return [{
            "symbol": code,
            "date": "2026-08-07",
            "title": "半年度报告",
            "source": "cninfo",
            "url": "https://example/cninfo",
            "announcement_id": "1",
        }]

    def _sse(self, code, digits, start_date, end_date, top_k):
        return [{
            "symbol": code,
            "date": "2026-08-07",
            "title": "半年度报告",
            "source": "sse",
            "url": "https://example/sse",
            "announcement_id": "2",
        }]


class StubNews(FinancialNewsSource):
    def __init__(self):
        pass

    def _cls(self, query, top_k):
        return [{
            "date": "2026-08-07",
            "title": "贵州茅台发布经营数据",
            "summary": "财联社摘要",
            "source": "cls",
            "url": "https://example/cls",
        }]

    def _sina(self, query, top_k):
        return [{
            "date": "2026-08-06",
            "title": "贵州茅台行业需求保持稳定",
            "summary": "新浪摘要",
            "source": "sina",
            "url": "https://example/sina",
        }]


class EventSourceTests(unittest.TestCase):
    def test_announcements_cross_check_and_deduplicate(self):
        result = StubAnnouncements().announcements(
            "sh.600519", "2026-01-01", "2026-08-08", 20
        )

        self.assertEqual(result["source_chain"], ["cninfo", "sse"])
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["corroborated_by"][0]["source"], "sse")

    def test_news_uses_sina_only_to_fill_cls_shortfall(self):
        result = StubNews().news(
            "sh.600519", "贵州茅台", "2026-01-01", "2026-08-08", 2
        )

        self.assertEqual(result["source_chain"], ["cls", "sina"])
        self.assertEqual([item["source"] for item in result["items"]], ["cls", "sina"])
        self.assertTrue(all(item["symbol"] == "sh.600519" for item in result["items"]))
        self.assertTrue(all(item["content_type"] == "news" for item in result["items"]))

    def test_news_rejects_generic_and_undated_records_before_counting(self):
        class MixedNews(StubNews):
            def _cls(self, query, top_k):
                return [
                    {
                        "date": "2026-08-07",
                        "title": "A股市场全天震荡",
                        "summary": "大盘成交活跃",
                        "source": "cls",
                        "url": "https://example/generic",
                    },
                    {
                        "date": None,
                        "title": "贵州茅台发布经营数据",
                        "summary": "缺少日期",
                        "source": "cls",
                        "url": "https://example/undated",
                    },
                ]

            def _sina(self, query, top_k):
                return [{
                    "date": "2026-08-06",
                    "title": "贵州茅台披露渠道调整",
                    "summary": "目标公司新闻",
                    "source": "sina",
                    "url": "https://example/target",
                }]

        result = MixedNews().news(
            "sh.600519", "贵州茅台", "2026-01-01", "2026-08-08", 2
        )

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["source"], "sina")

    def test_news_falls_back_to_ticker_query_without_relaxing_validation(self):
        class TickerFallbackNews(StubNews):
            def __init__(self):
                self.queries = []

            def _cls(self, query, top_k):
                self.queries.append(query)
                if query == "600519":
                    return [{
                        "date": "2026-08-07",
                        "title": "600519发布经营数据",
                        "summary": "目标证券代码出现在正文中",
                        "source": "cls",
                        "url": "https://example/ticker",
                    }]
                return [{
                    "date": "2026-08-07",
                    "title": "A股市场全天震荡",
                    "summary": "通用新闻",
                    "source": "cls",
                    "url": "https://example/generic",
                }]

            def _sina(self, query, top_k):
                return []

        source = TickerFallbackNews()
        result = source.news(
            "sh.600519", "贵州茅台", "2026-01-01", "2026-08-08", 1
        )

        self.assertEqual([item["title"] for item in result["items"]], ["600519发布经营数据"])
        self.assertEqual(result["query_chain"]["cls"], ["贵州茅台", "600519"])
        self.assertEqual(source.queries, ["贵州茅台", "600519"])

    def test_zero_effective_news_is_invalid_data(self):
        class GenericOnlyNews(StubNews):
            def _cls(self, query, top_k):
                return [{
                    "date": "2026-08-07",
                    "title": "市场热点轮动",
                    "summary": "未提及目标公司",
                    "source": "cls",
                    "url": "https://example/generic",
                }]

            def _sina(self, query, top_k):
                return []

        with self.assertRaises(PublicDataSourceError) as raised:
            GenericOnlyNews().news(
                "sh.600519", "贵州茅台", "2026-01-01", "2026-08-08", 2
            )

        self.assertEqual(raised.exception.code, "INVALID_DATA")
        self.assertFalse(raised.exception.retryable)

    def test_identical_results_for_different_windows_mark_broken_filter(self):
        class RepeatingNews(StubNews):
            def _cls(self, query, top_k):
                return [{
                    "date": "2026-08-07",
                    "title": "贵州茅台发布经营数据",
                    "summary": "相同集合",
                    "source": "cls",
                    "url": "https://example/repeat",
                }]

            def _sina(self, query, top_k):
                return []

        source = RepeatingNews()
        source.news("sh.600519", "贵州茅台", "2026-01-01", "2026-08-08", 1)

        with self.assertRaises(PublicDataSourceError) as raised:
            source.news("sh.600519", "贵州茅台", "2026-02-01", "2026-08-08", 1)

        self.assertEqual(raised.exception.code, "PROVIDER_FILTER_BROKEN")


class FakeApp:
    def __init__(self):
        self.registered = {}

    def tool(self):
        def register(func):
            self.registered[func.__name__] = func
            return func

        return register


class ToolRegistrationTests(unittest.TestCase):
    def test_all_public_tools_are_registered(self):
        app = FakeApp()
        register_public_source_tools(app)

        self.assertEqual(
            set(app.registered),
            {
                "get_mootdx_bars",
                "get_tencent_quote",
                "get_eastmoney_signals",
                "get_official_announcements",
                "get_financial_news",
            },
        )

    def test_real_fastmcp_builds_all_public_tool_schemas(self):
        app = FastMCP("public-source-contract-test")
        register_public_source_tools(app)

        names = {
            "get_mootdx_bars",
            "get_tencent_quote",
            "get_eastmoney_signals",
            "get_official_announcements",
            "get_financial_news",
        }
        for name in names:
            schema = app._tool_manager.get_tool(name).parameters
            self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
