import unittest

import pandas as pd

from src.data_source_interface import NoDataFoundError
from src.tools.stock_market import register_stock_market_tools


class FakeApp:
    def __init__(self):
        self.registered = {}

    def tool(self):
        def register(func):
            self.registered[func.__name__] = func
            return func

        return register


class FakeListingDataSource:
    def __init__(self, frame=None, error=None):
        self.frame = frame
        self.error = error
        self.calls = []

    def resolve_stock_listing(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.frame


class StockListingToolTests(unittest.TestCase):
    def register(self, source):
        app = FakeApp()
        register_stock_market_tools(app, source)
        return app.registered["resolve_stock_listing"]

    def test_returns_structured_listed_candidate(self):
        source = FakeListingDataSource(
            pd.DataFrame(
                [
                    {
                        "code": "sh.600519",
                        "code_name": "贵州茅台",
                        "ipoDate": "2001-08-27",
                        "outDate": "",
                        "type": "1",
                        "status": "1",
                    }
                ]
            )
        )
        tool = self.register(source)

        result = tool(code="sh.600519")

        self.assertTrue(result["ok"])
        content = result["data"]["content"]
        self.assertEqual(content["candidates"][0]["listing_status"], "listed")
        self.assertEqual(content["candidates"][0]["company_name"], "贵州茅台")
        self.assertEqual(
            source.calls,
            [{"code": "sh.600519", "code_name": None}],
        )

    def test_no_data_is_a_successful_empty_match_set(self):
        source = FakeListingDataSource(
            error=NoDataFoundError("not found"),
        )
        tool = self.register(source)

        result = tool(company_name="字节跳动")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["content"]["candidates"], [])

    def test_requires_code_or_company_name(self):
        tool = self.register(FakeListingDataSource())

        result = tool()

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT")


if __name__ == "__main__":
    unittest.main()
