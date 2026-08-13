import unittest

from src.utils.quality_gate import evaluate_analysis_result, run_quality_gate
from src.utils.state_definition import AnalysisResult, WorkflowState


def evidence(tool="get_historical_k_data"):
    return [{
        "tool": tool,
        "call_id": "call-1",
        "raw_data_hash": "sha256:data",
        "quality_score": 1.0,
    }]


def tool_call(tool="get_historical_k_data", scope="target"):
    return [{"tool": tool, "call_id": "call-1", "ok": True, "scope": scope}]


class SpecialistQualityTests(unittest.TestCase):
    def test_placeholder_is_never_success_even_with_evidence(self):
        result = AnalysisResult(
            agent_id="technical_agent",
            analysis_type="technical",
            content="Sorry, need more steps to process this request.",
            evidence=evidence(),
            tool_calls=tool_call(),
        )

        checked = evaluate_analysis_result(result)

        self.assertFalse(checked.success)
        self.assertIn(
            "INCOMPLETE_PLACEHOLDER",
            {issue["code"] for issue in checked.quality_issues},
        )

    def test_substantive_content_without_evidence_is_not_success(self):
        result = AnalysisResult(
            agent_id="technical_agent",
            analysis_type="technical",
            content=(
                "技术走势分析包含趋势、成交量、均线、支撑和阻力等多个维度，"
                "但这一段故意没有任何工具证据，因此无论文字多完整都不能作为成功结果进入后续报告。"
            ) * 2,
        )

        checked = evaluate_analysis_result(result)

        self.assertFalse(checked.success)
        self.assertIn(
            "NO_VALID_EVIDENCE",
            {issue["code"] for issue in checked.quality_issues},
        )

    def test_quality_gate_requires_every_requested_specialist(self):
        state = WorkflowState(
            task="综合分析",
            analysis_results={
                "technical": AnalysisResult(
                    agent_id="technical_agent",
                    analysis_type="technical",
                    content=(
                        "基于已验证的日线行情、成交量和复权数据，当前价格处于区间震荡，"
                        "均线结构没有给出确定方向。支撑阻力仅作为观察条件，后续必须由成交量和收盘价共同确认。"
                    ) * 2,
                    evidence=evidence(),
                    tool_calls=tool_call(),
                )
            },
        )

        update = run_quality_gate(state, ["technical", "event"])

        self.assertFalse(update["data"]["quality_gate"]["passed"])
        self.assertIn("event", update["data"]["quality_gate"]["failures"])

    def test_event_with_one_valid_category_is_degraded_but_usable(self):
        result = AnalysisResult(
            agent_id="event_agent",
            analysis_type="event",
            content=(
                "事件分析基于已核验记录，逐项列出日期、来源、链接、催化与下行情景，"
                "并严格限定在分析基准日之前，不把同行事件写成目标证券事实。"
            ) * 3,
            evidence=[{
                "tool": "get_financial_news",
                "call_id": "news-call",
                "raw_data_hash": "sha256:news",
                "quality_score": 1.0,
                "record_count": 5,
            }],
            tool_calls=[{
                "tool": "get_financial_news",
                "call_id": "news-call",
                "ok": True,
                "scope": "target",
            }],
        )

        checked = evaluate_analysis_result(result)

        self.assertTrue(checked.success)
        self.assertEqual(checked.quality_status, "DEGRADED")
        self.assertEqual(checked.data_completeness, 0.5)
        self.assertIn(
            "MINIMUM_DATA_INCOMPLETE",
            {issue["code"] for issue in checked.quality_issues},
        )

    def test_three_of_four_specialists_allows_degraded_workflow(self):
        def result(analysis_type, tools):
            return AnalysisResult(
                agent_id=f"{analysis_type}_agent",
                analysis_type=analysis_type,
                content=(
                    f"{analysis_type}分析基于已验证证据，明确列出适用范围、风险和观察条件。"
                ) * 4,
                evidence=[{
                    "tool": tool,
                    "call_id": f"{analysis_type}-{index}",
                    "raw_data_hash": f"sha256:{analysis_type}-{index}",
                    "quality_score": 1.0,
                    "record_count": 5,
                } for index, tool in enumerate(tools)],
                tool_calls=[{
                    "tool": tool,
                    "call_id": f"{analysis_type}-{index}",
                    "ok": True,
                    "scope": "target",
                } for index, tool in enumerate(tools)],
            )

        state = WorkflowState(
            task="综合分析",
            analysis_results={
                "fundamental": result(
                    "fundamental", ["get_stock_basic_info", "get_profit_data"]
                ),
                "technical": result("technical", ["get_historical_k_data"]),
                "value": result(
                    "value", ["get_historical_k_data", "get_profit_data"]
                ),
                "event": AnalysisResult(
                    agent_id="event_agent",
                    analysis_type="event",
                    content="Sorry, need more steps to process this request.",
                ),
            },
        )

        update = run_quality_gate(
            state, ["fundamental", "technical", "value", "event"]
        )
        gate = update["data"]["quality_gate"]

        self.assertTrue(gate["passed"])
        self.assertFalse(gate["fully_passed"])
        self.assertEqual(gate["status"], "DEGRADED")
        self.assertEqual(gate["coverage"], 0.75)
        self.assertEqual(gate["failed_analysis_types"], ["event"])


if __name__ == "__main__":
    unittest.main()
