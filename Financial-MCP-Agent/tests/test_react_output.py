import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.react_output import extract_react_output, is_step_limit_placeholder
from src.utils.state_definition import WorkflowState


class FakeLogger:
    def __init__(self):
        self.records = []

    def warning(self, message, *args, **kwargs):
        self.records.append(("warning", message, args))

    def error(self, message, *args, **kwargs):
        self.records.append(("error", message, args))

    def info(self, message, *args, **kwargs):
        self.records.append(("info", message, args))


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def ainvoke(self, input_data, config=None):
        self.calls.append((input_data, config))
        return AIMessage(content=self.content)


class ReactOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_limit_placeholder_triggers_tool_free_synthesis(self):
        llm = FakeLLM(
            "基于已取得的财务指标，公司盈利能力仍然较强，但近期增速明显放缓。"
            "最近报告期数据完整覆盖盈利、成长、现金流和偿债能力，缺失指标已明确披露。"
            "综合判断应同时关注高毛利优势、增长放缓风险和当前估值安全边际。"
        )
        logger = FakeLogger()
        placeholder = "Sorry, need more steps to process this request."
        response = {
            "messages": [
                HumanMessage(content="分析目标公司"),
                AIMessage(
                    content="继续获取财务数据",
                    tool_calls=[{
                        "name": "get_profit_data",
                        "args": {"code": "sh.600519"},
                        "id": "call-1",
                        "type": "tool_call",
                    }],
                ),
                ToolMessage(content="ROE=10.5%", tool_call_id="call-1"),
                AIMessage(content=placeholder),
            ]
        }

        output = await extract_react_output(
            response,
            llm=llm,
            logger=logger,
            operation_name="test agent",
            analysis_name="基本面分析",
        )

        self.assertGreater(len(output), 80)
        self.assertEqual(len(llm.calls), 1)
        recovery_messages = llm.calls[0][0]
        self.assertIsInstance(recovery_messages[-1], HumanMessage)
        self.assertIn("不得再请求调用任何工具", recovery_messages[-1].content)
        self.assertNotIn(placeholder, [message.content for message in recovery_messages])

    async def test_substantive_final_output_does_not_call_recovery_model(self):
        content = "已完成分析。" * 30
        llm = FakeLLM("不应调用")

        output = await extract_react_output(
            {"messages": [HumanMessage(content="分析"), AIMessage(content=content)]},
            llm=llm,
            logger=FakeLogger(),
            operation_name="test agent",
            analysis_name="技术分析",
        )

        self.assertEqual(output, content)
        self.assertEqual(llm.calls, [])

    def test_default_iteration_budget_allows_more_than_five_tool_rounds(self):
        self.assertEqual(WorkflowState().max_iterations, 20)

    def test_only_short_step_limit_message_is_classified_as_placeholder(self):
        self.assertTrue(
            is_step_limit_placeholder("Sorry, need more steps to process this request.")
        )
        self.assertFalse(
            is_step_limit_placeholder(
                "本报告说明旧版系统曾出现 sorry, need more steps 的错误，"
                + "但当前内容是一份完整分析。" * 20
            )
        )


if __name__ == "__main__":
    unittest.main()
