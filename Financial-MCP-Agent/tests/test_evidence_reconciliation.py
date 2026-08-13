import unittest
from datetime import date

from src.utils.report_renderer import (
    reconcile_analysis_sections,
    reconcile_evidence,
    render_evidence_registry,
)
from src.utils.state_definition import AnalysisResult, DataFetchStatus


class EvidenceReconciliationTests(unittest.TestCase):
    def test_success_wins_across_agents_and_removes_dividend_contradiction(self):
        results = {
            "fundamental": AnalysisResult(
                agent_id="fundamental_agent",
                analysis_type="fundamental",
                content="分红未获取。",
            ),
            "value": AnalysisResult(
                agent_id="value_agent",
                analysis_type="value",
                content="分红数据可用。",
                tool_calls=[
                    {
                        "agent_name": "value_agent",
                        "tool": "get_dividend_data",
                        "call_id": "dividend-call",
                        "arguments": {"code": "sz.300750", "year": "2025"},
                        "provider": "baostock",
                        "scope": "target",
                        "ok": True,
                        "record_count": 1,
                        "status": "SUCCESS",
                        "business_key": "dividend:sz.300750:2025",
                        "evidence_value": {"cash_dividend": 4.55},
                    }
                ],
            ),
        }

        registry = reconcile_evidence(
            results, symbol="sz.300750", as_of=date(2026, 8, 11)
        )

        entry = registry.entries["dividend:sz.300750:2025"]
        self.assertEqual(entry.status, DataFetchStatus.SUCCESS)
        reconciled = reconcile_analysis_sections(
            {"fundamental": "分红未获取。"}, registry
        )
        self.assertNotIn("分红未获取", reconciled["fundamental"])
        self.assertIn("SUCCESS", reconciled["fundamental"])

    def test_not_attempted_is_explicit_and_does_not_override_profit_success(self):
        results = {
            "fundamental": AnalysisResult(
                agent_id="fundamental_agent",
                analysis_type="fundamental",
                content="有效财务分析。" * 20,
                tool_calls=[
                    {
                        "agent_name": "fundamental_agent",
                        "tool": "get_profit_data",
                        "call_id": "profit-call",
                        "arguments": {
                            "code": "sz.300750",
                            "year": "2026",
                            "quarter": 2,
                        },
                        "provider": "baostock",
                        "scope": "target",
                        "ok": True,
                        "record_count": 1,
                        "status": "SUCCESS",
                        "business_key": "profit:sz.300750:2026Q2",
                        "evidence_value": {"net_profit": 100},
                    }
                ],
            )
        }

        registry = reconcile_evidence(
            results, symbol="sz.300750", as_of=date(2026, 8, 11)
        )
        entry = registry.entries["profit:sz.300750:2026Q2"]

        self.assertEqual(entry.status, DataFetchStatus.SUCCESS)
        omissions = [
            record
            for record in entry.records
            if record.status == DataFetchStatus.NOT_ATTEMPTED
        ]
        self.assertEqual(
            {record.requirement for record in omissions},
            {"get_forecast_report", "get_performance_express_report"},
        )
        self.assertTrue(all(record.call_id is None for record in omissions))
        rendered = render_evidence_registry(registry)
        self.assertIn("无（未调用）", rendered)

    def test_conflicting_sources_retain_every_value_and_scope(self):
        calls = []
        for call_id, provider, value in (
            ("quote-1", "tencent", 210.1),
            ("quote-2", "eastmoney", 211.3),
        ):
            calls.append(
                {
                    "agent_name": "technical_agent",
                    "tool": "get_tencent_quote",
                    "call_id": call_id,
                    "arguments": {"code": "sz.300750", "date": "2026-08-11"},
                    "provider": provider,
                    "scope": "target",
                    "ok": True,
                    "record_count": 1,
                    "status": "SUCCESS",
                    "business_key": "quote:sz.300750:2026-08-11",
                    "evidence_value": {"close": value},
                }
            )
        registry = reconcile_evidence(
            {
                "technical": AnalysisResult(
                    agent_id="technical_agent",
                    analysis_type="technical",
                    content="行情证据分析。" * 20,
                    tool_calls=calls,
                )
            },
            symbol="sz.300750",
            as_of=date(2026, 8, 11),
        )

        entry = registry.entries["quote:sz.300750:2026-08-11"]
        self.assertTrue(entry.conflict)
        self.assertEqual(len(entry.values), 2)
        self.assertEqual({item["scope"] for item in entry.values}, {"target"})

    def test_budget_prose_is_replaced_by_traceable_structured_status(self):
        result = AnalysisResult(
            agent_id="fundamental_agent",
            analysis_type="fundamental",
            content="现金流因预算失败未获取。",
            tool_calls=[
                {
                    "agent_name": "fundamental_agent",
                    "tool": "get_cash_flow_data",
                    "call_id": "budget-call",
                    "arguments": {"code": "sz.300750", "year": "2025"},
                    "data_source": "gateway",
                    "error_code": "TOOL_CALL_BUDGET_EXCEEDED",
                    "ok": False,
                    "status": "BUDGET_BLOCKED",
                }
            ],
        )
        registry = reconcile_evidence(
            {"fundamental": result},
            symbol="sz.300750",
            as_of=date(2026, 8, 11),
        )
        reconciled = reconcile_analysis_sections(
            {"fundamental": result.content}, registry
        )["fundamental"]

        self.assertNotIn("因预算失败", reconciled)
        self.assertIn("BUDGET_BLOCKED", reconciled)
        self.assertIn("budget-call", reconciled)


if __name__ == "__main__":
    unittest.main()
