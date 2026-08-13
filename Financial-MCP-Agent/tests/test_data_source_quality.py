import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from src.tools import data_collection
from src.tools.data_collection import CollectionSpec, estimate_record_count
from src.tools.data_quality import validate_tool_records
from src.utils.state_definition import WorkflowState


class Tool:
    name = "get_financial_news"


class ResponseTool:
    description = "test tool"
    args_schema = None

    def __init__(self, name, responses):
        self.name = name
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, arguments):
        self.calls.append(arguments)
        return self.responses.pop(0)


def news_payload(*items):
    return {"items": list(items)}


def success_envelope(value):
    return {
        "ok": True,
        "data": {"content": value},
        "error": None,
        "meta": {"provider": "test"},
    }


def failure_envelope(code):
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": code, "retryable": True},
        "meta": {"provider": "test"},
    }


class NewsQualityTests(unittest.TestCase):
    def test_count_includes_only_dated_target_relevant_news(self):
        payload = news_payload(
            {
                "date": "2026-08-01",
                "title": "贵州茅台发布经营数据",
                "summary": "",
            },
            {
                "date": "2026-08-01",
                "title": "A股市场震荡",
                "summary": "通用市场新闻",
            },
            {
                "date": None,
                "title": "贵州茅台渠道调整",
                "summary": "",
            },
        )
        arguments = {
            "code": "sh.600519",
            "company_name": "贵州茅台",
            "start_date": "2026-07-01",
            "end_date": "2026-08-11",
        }

        validation = validate_tool_records("get_financial_news", payload, arguments)

        self.assertEqual(validation.record_count, 1)
        self.assertEqual(validation.rejected_count, 2)
        self.assertEqual(
            estimate_record_count(payload, "get_financial_news", arguments), 1
        )
        self.assertEqual(len(validation.data["items"]), 1)

    def test_entity_recognition_metadata_can_establish_relevance(self):
        payload = news_payload(
            {
                "date": "2026-08-01",
                "title": "公司发布经营数据",
                "summary": "",
                "recognized_entities": ["贵州茅台"],
            }
        )
        arguments = {
            "code": "sh.600519",
            "company_name": "贵州茅台",
            "start_date": "2026-07-01",
            "end_date": "2026-08-11",
        }

        self.assertEqual(
            estimate_record_count(payload, "get_financial_news", arguments), 1
        )


class CollectionStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_zero_effective_news_is_invalid_data_not_success(self):
        payload = news_payload(
            {
                "date": "2026-08-01",
                "title": "A股市场震荡",
                "summary": "通用市场新闻",
            }
        )

        async def execute(_executor, step, _state):
            return SimpleNamespace(
                ok=True,
                data=payload,
                error=None,
                quality=SimpleNamespace(status="unchecked", score=0.0, warnings=[]),
                meta=SimpleNamespace(
                    call_id=step.id,
                    request_hash="sha256:request",
                    raw_data_hash="sha256:data",
                    provider="cls",
                    upstream_meta={},
                ),
            )

        spec = CollectionSpec(
            "news",
            "新闻",
            "required",
            "required",
            "get_financial_news",
            lambda state: {
                "code": state.target_symbol,
                "company_name": state.company_name,
                "start_date": "2026-07-01",
                "end_date": "2026-08-11",
                "top_k": 10,
            },
        )
        state = WorkflowState(
            task="test",
            symbol="sh.600519",
            company_name="贵州茅台",
            as_of=date(2026, 8, 11),
            data={"persistent_cache_enabled": False},
        )
        with (
            patch.object(data_collection, "COMMON_COLLECTION_CALLS", (spec,)),
            patch.object(
                data_collection,
                "get_all_tools_for_orchestrator",
                return_value=[Tool()],
            ),
            patch.object(data_collection.MCPToolExecutor, "execute", execute),
        ):
            update = await data_collection.collect_common_target_data(state)

        item = update["data"]["collection_plan"][0]
        self.assertEqual(item["status"], "INVALID_DATA")
        self.assertEqual(item["code"], "INVALID_NEWS_DATA")
        self.assertEqual(update["data"]["collection_report"]["cached_calls"], 0)

    async def _collect_single(self, tool):
        spec = CollectionSpec(
            "single",
            "single",
            "required",
            "required",
            tool.name,
            lambda state: {"code": state.target_symbol},
        )
        state = WorkflowState(
            task="test",
            symbol="sh.600519",
            data={"persistent_cache_enabled": False},
        )
        with (
            patch.object(data_collection, "COMMON_COLLECTION_CALLS", (spec,)),
            patch.object(
                data_collection,
                "get_all_tools_for_orchestrator",
                return_value=[tool],
            ),
        ):
            return await data_collection.collect_common_target_data(state)

    async def test_timeout_retries_same_parameters_once(self):
        tool = ResponseTool(
            "test_source",
            [failure_envelope("TOOL_TIMEOUT"), success_envelope({"value": 1})],
        )

        update = await self._collect_single(tool)

        self.assertEqual(len(tool.calls), 2)
        self.assertEqual(tool.calls[0], tool.calls[1])
        item = update["data"]["collection_plan"][0]
        self.assertEqual(item["status"], "SUCCESS")
        self.assertEqual(item["attempts"], 2)

    async def test_no_data_does_not_retry_same_parameters(self):
        tool = ResponseTool(
            "test_source",
            [failure_envelope("NO_DATA"), success_envelope({"value": 1})],
        )

        update = await self._collect_single(tool)

        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(update["data"]["collection_plan"][0]["status"], "NO_DATA")

    async def test_upstream_error_switches_without_same_source_retry(self):
        tool = ResponseTool(
            "test_source",
            [failure_envelope("UPSTREAM_ERROR"), success_envelope({"value": 1})],
        )

        update = await self._collect_single(tool)

        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(
            update["data"]["collection_plan"][0]["status"], "UPSTREAM_ERROR"
        )
