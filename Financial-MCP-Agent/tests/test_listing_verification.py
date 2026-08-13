import unittest

from src.utils.listing_verification import ListingStatus, verify_a_share_listing


class FakeResolverTool:
    name = "resolve_stock_listing"

    def __init__(self, response):
        self.response = response
        self.calls = []

    async def ainvoke(self, arguments):
        self.calls.append(arguments)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def tool_success(candidates):
    return {
        "ok": True,
        "data": {
            "content": {
                "market": "A-share",
                "candidates": candidates,
            }
        },
        "error": None,
        "meta": {"tool": "resolve_stock_listing"},
    }


def candidate(
    code="sh.600519",
    name="贵州茅台",
    status="listed",
):
    return {
        "stock_code": code,
        "company_name": name,
        "exchange": code.split(".", 1)[0],
        "listing_status": status,
    }


class ListingVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_listed_security_is_canonicalized_and_allowed(self):
        tool = FakeResolverTool(tool_success([candidate()]))

        result = await verify_a_share_listing(
            {"company_name": "茅台", "stock_code": "600519"},
            tools=[tool],
        )

        self.assertEqual(result.status, ListingStatus.LISTED)
        self.assertTrue(result.may_start_workflow)
        self.assertEqual(result.company_name, "贵州茅台")
        self.assertEqual(result.stock_code, "sh.600519")
        self.assertEqual(tool.calls, [{"code": "sh.600519"}])

    async def test_private_company_without_a_share_match_is_stopped(self):
        tool = FakeResolverTool(tool_success([]))

        result = await verify_a_share_listing(
            {"company_name": "字节跳动", "stock_code": None},
            tools=[tool],
        )

        self.assertEqual(result.status, ListingStatus.NOT_SUPPORTED)
        self.assertFalse(result.may_start_workflow)
        self.assertIn("未在当前 A 股证券主数据中找到", result.message)
        self.assertEqual(tool.calls, [{"company_name": "字节跳动"}])

    async def test_multiple_matches_require_an_exact_code(self):
        tool = FakeResolverTool(
            tool_success(
                [
                    candidate("sh.600001", "示例股份"),
                    candidate("sz.000001", "示例银行"),
                ]
            )
        )

        result = await verify_a_share_listing(
            {"company_name": "示例", "stock_code": None},
            tools=[tool],
        )

        self.assertEqual(result.status, ListingStatus.AMBIGUOUS)
        self.assertFalse(result.may_start_workflow)

    async def test_code_company_mismatch_is_stopped(self):
        tool = FakeResolverTool(tool_success([candidate()]))

        result = await verify_a_share_listing(
            {"company_name": "字节跳动", "stock_code": "600519"},
            tools=[tool],
        )

        self.assertEqual(result.status, ListingStatus.AMBIGUOUS)
        self.assertIn("与输入的", result.message)

    async def test_delisted_security_is_stopped(self):
        tool = FakeResolverTool(
            tool_success([candidate("sh.600001", "退市示例", "delisted")])
        )

        result = await verify_a_share_listing(
            {"company_name": "退市示例", "stock_code": "600001"},
            tools=[tool],
        )

        self.assertEqual(result.status, ListingStatus.DELISTED)
        self.assertFalse(result.may_start_workflow)

    async def test_provider_failure_fails_closed(self):
        tool = FakeResolverTool(RuntimeError("offline"))

        result = await verify_a_share_listing(
            {"company_name": "贵州茅台", "stock_code": "600519"},
            tools=[tool],
        )

        self.assertEqual(result.status, ListingStatus.UNKNOWN)
        self.assertFalse(result.may_start_workflow)

    async def test_hong_kong_symbol_is_reported_as_unsupported(self):
        result = await verify_a_share_listing(
            {"company_name": "腾讯控股", "stock_code": "00700", "llm_market": "港股"},
            tools=[],
        )

        self.assertEqual(result.status, ListingStatus.NOT_SUPPORTED)
        self.assertIn("港股", result.message)


if __name__ == "__main__":
    unittest.main()
