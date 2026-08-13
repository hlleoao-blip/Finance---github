import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import AsyncMock, Mock, patch

from src import main as main_module
from src.ui import TerminalUI


class InteractiveSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_accepts_multiple_queries_until_exit(self):
        answers = iter(["", "分析贵州茅台 600519", "分析宁德时代 300750", "退出"])
        analyze = AsyncMock()

        with redirect_stdout(StringIO()):
            await main_module.run_interactive_session(
                show_analysis_output=True,
                input_func=lambda _prompt: next(answers),
                analyze_func=analyze,
            )

        self.assertEqual(analyze.await_count, 2)
        self.assertEqual(
            analyze.await_args_list[0].args,
            ("分析贵州茅台 600519",),
        )
        self.assertTrue(analyze.await_args_list[0].kwargs["show_analysis_output"])
        self.assertEqual(
            analyze.await_args_list[1].args,
            ("分析宁德时代 300750",),
        )

    async def test_eof_ends_session_without_analysis(self):
        analyze = AsyncMock()

        def end_input(_prompt):
            raise EOFError

        with redirect_stdout(StringIO()):
            await main_module.run_interactive_session(
                input_func=end_input,
                analyze_func=analyze,
            )

        analyze.assert_not_awaited()

    def test_exit_commands_are_case_insensitive(self):
        self.assertTrue(main_module.is_exit_command(" EXIT "))
        self.assertTrue(main_module.is_exit_command("退出"))
        self.assertTrue(main_module.is_exit_command("/QUIT"))
        self.assertFalse(main_module.is_exit_command("分析退出股份"))

    async def test_session_commands_do_not_trigger_analysis(self):
        answers = iter(
            [
                "/help",
                "/verbose on",
                "分析贵州茅台 600519",
                "/history",
                "/verbose off",
                "分析宁德时代 300750",
                "/quit",
            ]
        )
        analyze = AsyncMock()

        with redirect_stdout(StringIO()):
            await main_module.run_interactive_session(
                input_func=lambda _prompt: next(answers),
                analyze_func=analyze,
            )

        self.assertEqual(analyze.await_count, 2)
        self.assertTrue(analyze.await_args_list[0].kwargs["show_analysis_output"])
        self.assertFalse(analyze.await_args_list[1].kwargs["show_analysis_output"])


class TerminalUITests(unittest.TestCase):
    def test_plain_welcome_uses_requested_width(self):
        output = StringIO()
        ui = TerminalUI(stream=output, width=60, force_plain=True)

        ui.welcome()

        rendered = output.getvalue()
        self.assertIn("FinAgent · A股分析 / 交互模式", rendered)
        self.assertIn("/help 查看命令 · /quit 退出", rendered)
        self.assertIn("─" * 60, rendered)
        self.assertNotIn("═" * 78, rendered)

    def test_result_card_contains_target_elapsed_and_report(self):
        output = StringIO()
        ui = TerminalUI(stream=output, force_plain=True)

        ui.result(
            success=True,
            title="分析完成",
            company_name="贵州茅台",
            stock_code="600519",
            elapsed_seconds=138,
            report_path="reports/report.md",
        )

        rendered = output.getvalue()
        self.assertIn("[OK] 分析完成", rendered)
        self.assertIn("贵州茅台 600519", rendered)
        self.assertIn("02:18", rendered)
        self.assertIn("reports/report.md", rendered)

    def test_preflight_failure_uses_recognition_labels(self):
        output = StringIO()
        ui = TerminalUI(stream=output, force_plain=True)

        ui.result(
            success=False,
            title="无法确认证券",
            company_name="宇树科技",
            elapsed_seconds=5,
            detail="未找到可分析的 A 股匹配项。",
            target_label="识别结果",
            detail_label="未启动原因",
        )

        rendered = output.getvalue()
        self.assertIn("识别结果：宇树科技", rendered)
        self.assertIn("未启动原因：未找到可分析的 A 股匹配项。", rendered)
        self.assertNotIn("分析标的", rendered)

    def test_plain_prompt_uses_gbk_safe_separator(self):
        ui = TerminalUI(stream=StringIO(), force_plain=True)

        prompt = ui.read_query(lambda value: value)

        self.assertEqual(prompt, "分析 > ")
        prompt.encode("gbk")

    def test_no_color_environment_forces_plain_mode(self):
        with patch.dict("os.environ", {"NO_COLOR": "1"}):
            ui = TerminalUI(stream=StringIO())

        self.assertTrue(ui.plain)


class MainModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_command_mode_runs_once_and_closes_client(self):
        args = Namespace(command="分析贵州茅台 600519", show_analysis_output=False)
        parser = Mock()
        parser.parse_args.return_value = args

        with (
            patch.object(main_module, "build_parser", return_value=parser),
            patch.object(main_module, "run_query", AsyncMock()) as run_query,
            patch.object(
                main_module,
                "run_interactive_session",
                AsyncMock(),
            ) as session,
            patch.object(
                main_module,
                "close_mcp_client_sessions",
                AsyncMock(),
            ) as close_client,
        ):
            await main_module.main()

        run_query.assert_awaited_once_with(
            "分析贵州茅台 600519",
            show_analysis_output=False,
        )
        session.assert_not_awaited()
        close_client.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
