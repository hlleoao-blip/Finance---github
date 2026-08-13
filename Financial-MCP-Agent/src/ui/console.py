"""自适应、可降级的终端展示层。"""

from __future__ import annotations

from contextlib import contextmanager
import os
import shutil
import sys
from typing import Any, Callable, Iterator, TextIO

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - 由纯文本降级测试覆盖行为
    Console = Group = Panel = Text = None  # type: ignore[assignment]
    RICH_AVAILABLE = False


MIN_WIDTH = 28
MAX_WIDTH = 100


class TerminalUI:
    """集中管理 CLI 输出，避免业务逻辑散落 ANSI 转义和 ``print``。"""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        width: int | None = None,
        force_plain: bool | None = None,
    ) -> None:
        self.stream = stream or sys.stdout
        terminal_width = width or shutil.get_terminal_size((80, 24)).columns
        self.width = max(MIN_WIDTH, min(terminal_width, MAX_WIDTH))

        env_plain = bool(os.getenv("NO_COLOR")) or os.getenv("TERM") == "dumb"
        self.plain = env_plain if force_plain is None else force_plain
        self.plain = self.plain or not RICH_AVAILABLE

        self.console = None
        if not self.plain:
            self.console = Console(
                file=self.stream,
                width=self.width,
                highlight=False,
                soft_wrap=False,
            )

    def _write(self, value: str = "") -> None:
        print(value, file=self.stream)

    def _message(self, prefix: str, message: Any, style: str) -> None:
        if self.console:
            line = Text(prefix, style=style)
            line.append(f" {message}")
            self.console.print(line)
        else:
            self._write(f"{prefix} {message}")

    def welcome(self) -> None:
        """显示紧凑的欢迎卡片。"""
        if self.console:
            body = Text()
            body.append("输入公司名称和股票代码开始分析\n", style="white")
            body.append("示例：", style="dim")
            body.append("分析贵州茅台 600519\n", style="cyan")
            body.append("/help", style="cyan")
            body.append(" 查看命令 · ", style="dim")
            body.append("/quit", style="cyan")
            body.append(" 退出", style="dim")
            self.console.print()
            self.console.print(
                Panel(
                    body,
                    title="[bold cyan]FinAgent[/] · A股分析",
                    subtitle="交互模式",
                    border_style="cyan",
                    width=self.width,
                    padding=(1, 2),
                )
            )
            self.console.print()
            return

        rule = "─" * self.width
        self._write()
        self._write(rule)
        self._write("FinAgent · A股分析 / 交互模式")
        self._write("输入公司名称和股票代码开始分析")
        self._write("示例：分析贵州茅台 600519")
        self._write("/help 查看命令 · /quit 退出")
        self._write(rule)
        self._write()

    def read_query(self, input_func: Callable[[str], Any] = input) -> Any:
        """显示短提示符并读取一条查询。"""
        if self.console:
            prompt = Text("分析", style="bold cyan")
            prompt.append(" › ", style="cyan")
            self.console.print(prompt, end="")
            return input_func("")
        # 纯文本路径可能运行在 Windows GBK 管道中，使用 ASCII 箭头避免编码失败。
        return input_func("分析 > ")

    def success(self, message: Any) -> None:
        self._message("[OK]", message, "bold green")

    def error(self, message: Any) -> None:
        self._message("[ERROR]", message, "bold red")

    def warning(self, message: Any) -> None:
        self._message("[WARN]", message, "yellow")

    def info(self, message: Any) -> None:
        self._message("[INFO]", message, "cyan")

    def muted(self, message: Any) -> None:
        if self.console:
            self.console.print(str(message), style="dim")
        else:
            self._write(str(message))

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        """在真实终端显示瞬时 spinner，重定向时输出稳定的纯文本。"""
        if self.console and self.console.is_terminal:
            with self.console.status(message, spinner="dots", spinner_style="cyan"):
                yield
            return

        self.info(message)
        yield

    def analysis_task(
        self,
        *,
        company_name: str | None,
        stock_code: str | None,
        analysis_label: str,
    ) -> None:
        target = " ".join(part for part in (company_name, stock_code) if part)
        target = target or "待识别证券"

        if self.console:
            body = Text()
            body.append(target, style="bold white")
            body.append("\n分析范围  ", style="dim")
            body.append(analysis_label, style="cyan")
            self.console.print(
                Panel(
                    body,
                    title="分析任务",
                    border_style="blue",
                    width=self.width,
                    padding=(0, 2),
                )
            )
            return

        self._write()
        self._write(f"分析任务：{target}")
        self._write(f"分析范围：{analysis_label}")

    def result(
        self,
        *,
        success: bool,
        title: str,
        company_name: str | None = None,
        stock_code: str | None = None,
        elapsed_seconds: float | None = None,
        report_path: str | None = None,
        detail: str | None = None,
        target_label: str = "分析标的",
        detail_label: str = "说明",
    ) -> None:
        """显示一次任务的最终结果卡片。"""
        target = " ".join(part for part in (company_name, stock_code) if part)
        elapsed = self.format_elapsed(elapsed_seconds) if elapsed_seconds is not None else None

        if self.console:
            lines: list[Any] = []
            if target:
                target_line = Text(f"{target_label}  ", style="dim")
                target_line.append(target, style="bold white")
                lines.append(target_line)
            if elapsed:
                elapsed_line = Text("耗时      ", style="dim")
                elapsed_line.append(elapsed)
                lines.append(elapsed_line)
            if report_path:
                path_line = Text("报告      ", style="dim")
                path_line.append(report_path, style="cyan")
                lines.append(path_line)
            if detail:
                detail_line = Text(f"{detail_label}  ", style="dim")
                detail_line.append(detail, style="red" if not success else "white")
                lines.append(detail_line)

            self.console.print(
                Panel(
                    Group(*lines),
                    title=(
                        f"[bold green]✓ {title}[/]"
                        if success
                        else f"[bold red]× {title}[/]"
                    ),
                    border_style="green" if success else "red",
                    width=self.width,
                    padding=(0, 2),
                )
            )
            return

        prefix = "[OK]" if success else "[ERROR]"
        self._write(f"{prefix} {title}")
        if target:
            self._write(f"{target_label}：{target}")
        if elapsed:
            self._write(f"耗时：{elapsed}")
        if report_path:
            self._write(f"报告：{report_path}")
        if detail:
            self._write(f"{detail_label}：{detail}")

    def section(self, title: str) -> None:
        if self.console:
            self.console.rule(title, style="blue")
        else:
            self._write(f"\n--- {title} ---\n")

    def content(self, value: Any) -> None:
        if self.console:
            self.console.print(str(value), markup=False)
        else:
            self._write(str(value))

    def next_hint(self) -> None:
        self.muted("继续输入新的分析需求，或输入 /quit 结束。")
        self._write()

    def help(self) -> None:
        commands = (
            ("/help", "显示可用命令"),
            ("/history", "显示本次会话的分析记录"),
            ("/verbose on|off", "开启或关闭完整分析输出"),
            ("/clear", "清空终端"),
            ("/quit", "结束会话"),
        )
        if self.console:
            body = Text()
            for index, (command, description) in enumerate(commands):
                if index:
                    body.append("\n")
                body.append(f"{command:<18}", style="cyan")
                body.append(description)
            self.console.print(
                Panel(body, title="会话命令", border_style="blue", width=self.width)
            )
            return

        self._write("会话命令：")
        for command, description in commands:
            self._write(f"  {command:<18} {description}")

    def history(self, queries: list[str]) -> None:
        if not queries:
            self.muted("本次会话还没有分析记录。")
            return
        self.section("会话历史")
        for index, query in enumerate(queries, start=1):
            self.content(f"{index:>2}. {query}")

    def clear(self) -> None:
        if self.console:
            self.console.clear()
        else:
            self._write("\n" * 2)

    @staticmethod
    def format_elapsed(seconds: float | None) -> str:
        total = max(0, int(seconds or 0))
        minutes, remaining = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{remaining:02d}"
        return f"{minutes:02d}:{remaining:02d}"
