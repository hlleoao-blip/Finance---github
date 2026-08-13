"""
金融分析智能体系统主程序 (Financial Analysis AI Agent System Main Program)

本文件是金融分析智能体系统的核心入口点，实现了以下主要功能：

1. 多智能体工作流管理：使用LangGraph构建并行执行的智能体工作流
2. 命令行界面：提供用户友好的交互式命令行界面
3. 自然语言处理：自动识别和提取股票代码、公司名称
4. 日志系统：完整的执行日志记录和错误处理
5. 报告生成：生成综合性的金融分析报告

工作流程：
start_node → [fundamental_analyst, technical_analyst, value_analyst] → summarizer → END
"""

# ============================================================================
# 导入必要的模块和依赖
# ============================================================================

# 在导入其他模块之前设置环境变量，抑制无用输出
import os
import sys

# 提前加入项目根目录，保证从工作区根目录直接执行本文件时也能导入 src 包
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 设置环境变量来抑制transformers和其他库的冗余输出
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # 只显示错误信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer并行化警告
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 减少CUDA相关输出
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 减少内存分配信息

# 设置日志级别，抑制第三方库的INFO级别输出
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# 日志和状态管理相关导入
from src.utils.logging_config import setup_logger, SUCCESS_ICON
from src.utils.state_definition import WorkflowState
from src.utils.execution_logger import initialize_execution_logger, finalize_execution_logger, get_execution_logger
from src.utils.stock_extraction import extract_stock_info_hybrid
from src.utils.listing_verification import verify_a_share_listing
from src.utils.terminal_output import strip_terminal_emoji
from src.utils.analysis_intent import infer_single_analysis_type
from src.tools.mcp_client import close_mcp_client_sessions
from src.ui import TerminalUI

from src.workflow import run_financial_workflow

# 环境变量和系统相关导入
from dotenv import load_dotenv
import argparse
import asyncio
import time
from datetime import datetime

# ============================================================================
# 初始化和配置
# ============================================================================

# 设置日志记录器
logger = setup_logger(__name__)

# 加载环境变量（从.env文件）
load_dotenv(override=True)

# 调试：打印关键环境变量以验证配置
logger.info(f"Environment Variables Loaded:")
logger.info(
    f"  OPENAI_COMPATIBLE_MODEL: {os.getenv('OPENAI_COMPATIBLE_MODEL', 'Not Set')}")
logger.info(
    f"  OPENAI_COMPATIBLE_BASE_URL: {os.getenv('OPENAI_COMPATIBLE_BASE_URL', 'Not Set')}")
logger.info(
    f"  OPENAI_COMPATIBLE_API_KEY: {'*' * 20 if os.getenv('OPENAI_COMPATIBLE_API_KEY') else 'Not Set'}")

# 重新设置日志记录器（确保正确配置）
logger = setup_logger(__name__)


ANALYSIS_AGENT_CONFIGS = {
    "fundamental": {
        "label": "基本面分析",
        "node_name": "fundamental_analyst",
        "result_key": "fundamental_analysis",
        "error_key": "fundamental_analysis_error",
    },
    "technical": {
        "label": "技术面分析",
        "node_name": "technical_analyst",
        "result_key": "technical_analysis",
        "error_key": "technical_analysis_error",
    },
    "value": {
        "label": "估值分析",
        "node_name": "value_analyst",
        "result_key": "value_analysis",
        "error_key": "value_analysis_error",
    },
    "event": {
        "label": "事件分析",
        "node_name": "event_analyst",
        "result_key": "event_analysis",
        "error_key": "event_analysis_error",
    },
}


async def run_query(user_query: str, *, show_analysis_output: bool = False):
    """
    执行一条金融分析查询。
    
    功能包括：
    1. 初始化执行日志系统
    2. 构建LangGraph工作流
    3. 处理用户输入
    4. 提取股票信息（代码、公司名称）
    5. 执行多智能体分析工作流
    6. 生成和保存分析报告
    7. 错误处理和日志记录
    """
    
    ui = TerminalUI()

    # 初始化执行日志系统
    execution_logger = initialize_execution_logger()
    main_agent_start_time = time.time()
    logger.info(
        f"{SUCCESS_ICON} 执行日志系统已初始化，日志目录: {execution_logger.execution_dir}")

    try:
        # ============================================================================
        # 1. 实现命令行界面 
        # ============================================================================
        
        user_query = user_query.strip()
        if not user_query:
            raise ValueError("输入不能为空。")

        requested_analysis_type = infer_single_analysis_type(user_query)
        requested_analysis_config = (
            ANALYSIS_AGENT_CONFIGS.get(requested_analysis_type)
            if requested_analysis_type
            else None
        )
        if requested_analysis_config:
            logger.info(
                f"Detected single-analysis intent: {requested_analysis_type}"
            )
        else:
            logger.info("No single-analysis intent detected; using full workflow.")

        # 记录用户查询到执行日志
        main_agent_start_time = time.time()
        execution_logger.log_agent_start("main", {"user_query": user_query})

        # ============================================================================
        # 3. 自然语言处理和股票信息提取
        # ============================================================================
        
        # 执行混合提取：规则优先，规则不完整时用 LLM 结构化提取兜底
        with ui.status("正在识别证券信息..."):
            extraction_result = await extract_stock_info_hybrid(user_query, logger)
        company_name = extraction_result.get("company_name")
        stock_code = extraction_result.get("stock_code")

        # 记录 LLM 兜底提取交互，便于审计和后续优化
        if extraction_result.get("llm_used") and extraction_result.get("llm_raw_response"):
            execution_logger.log_llm_interaction(
                agent_name="main",
                interaction_type="stock_info_extraction",
                input_messages=extraction_result.get("llm_prompt_messages", []),
                output_content=extraction_result.get("llm_raw_response", ""),
                model_config=extraction_result.get("llm_model_config", {}),
                execution_time=extraction_result.get("llm_execution_time", 0),
            )

        # 记录提取结果
        logger.info(
            "从查询中提取 - "
            f"公司名称: {company_name}, 股票代码: {stock_code}, "
            f"来源: {extraction_result.get('source')}, "
            f"LLM使用: {extraction_result.get('llm_used')}, "
            f"置信度: {extraction_result.get('confidence')}"
        )

        # 在启动任何分析 Agent 前，用 A 股证券主数据验证代码、名称和上市状态。
        with ui.status("正在核验 A 股上市状态..."):
            listing_check = await verify_a_share_listing(extraction_result)
        logger.info(
            "上市状态预检 - 状态: %s, 公司: %s, 代码: %s, 说明: %s",
            listing_check.status.value,
            listing_check.company_name,
            listing_check.stock_code,
            listing_check.message,
        )
        if not listing_check.may_start_workflow:
            ui.result(
                success=False,
                title="无法确认证券",
                company_name=listing_check.company_name,
                stock_code=listing_check.stock_code,
                elapsed_seconds=time.time() - main_agent_start_time,
                detail=listing_check.message,
                target_label="识别结果",
                detail_label="未启动原因",
            )
            execution_logger.log_agent_complete(
                "main",
                {
                    "user_query": user_query,
                    "workflow_started": False,
                    "listing_preflight": listing_check.model_dump(mode="json"),
                },
                time.time() - main_agent_start_time,
                True,
                None,
            )
            finalize_execution_logger(success=True, error=None)
            ui.muted(f"执行日志：{execution_logger.execution_dir}")
            return

        company_name = listing_check.company_name
        stock_code = listing_check.stock_code

        # ============================================================================
        # 4. 时间信息处理
        # ============================================================================
        
        # 获取当前时间信息
        current_datetime = datetime.now()
        current_date_cn = current_datetime.strftime("%Y年%m月%d日")
        current_date_en = current_datetime.strftime("%Y-%m-%d")
        current_weekday_cn = ["星期一", "星期二", "星期三", "星期四",
                              "星期五", "星期六", "星期日"][current_datetime.weekday()]
        current_time = current_datetime.strftime("%H:%M:%S")

        # 格式化完整的时间信息
        current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"

        logger.info(f"当前时间: {current_time_info}")

        # ============================================================================
        # 5. 准备初始状态数据
        # ============================================================================
        
        # 准备初始状态
        initial_data = {
            "query": user_query,
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat(),
            "show_analysis_output": show_analysis_output,
            "requested_analysis_type": requested_analysis_type,
            "stock_extraction": {
                "source": extraction_result.get("source"),
                "rule_company_name": extraction_result.get("rule_company_name"),
                "rule_stock_code": extraction_result.get("rule_stock_code"),
                "llm_used": extraction_result.get("llm_used"),
                "llm_error": extraction_result.get("llm_error"),
                "llm_company_name": extraction_result.get("llm_company_name"),
                "llm_stock_code": extraction_result.get("llm_stock_code"),
                "llm_market": extraction_result.get("llm_market"),
                "confidence": extraction_result.get("confidence"),
                "analysis_intent": extraction_result.get("analysis_intent", []),
                "needs_clarification": extraction_result.get("needs_clarification"),
                "clarification_reason": extraction_result.get("clarification_reason"),
            },
            "listing_preflight": listing_check.model_dump(mode="json"),
        }
        
        # 添加公司名称（如果提取到）
        if company_name:
            initial_data["company_name"] = company_name
            
        # 预检结果已经返回规范化且验证过的 A 股代码。
        if stock_code:
            initial_data["stock_code"] = stock_code

        # 创建LangGraph工作流的初始状态
        initial_state = WorkflowState(
            task=user_query,
            symbol=initial_data.get("stock_code"),
            company_name=initial_data.get("company_name"),
            messages=[],  # Langchain约定：消息列表
            data=initial_data,  # 应用特定数据，包含提取的信息
            metadata={}  # 其他运行时特定信息
        )

        # ============================================================================
        # 6. 执行工作流
        # ============================================================================
        
        # 显示分析开始信息
        analysis_label = (
            requested_analysis_config["label"]
            if requested_analysis_config
            else "全面分析 · 基本面 / 技术面 / 估值 / 事件 / 风险决策"
        )
        ui.analysis_task(
            company_name=company_name,
            stock_code=stock_code,
            analysis_label=analysis_label,
        )
        logger.info(
            f"Starting financial analysis workflow for query: '{user_query}'")

        if requested_analysis_config:
            with ui.status(f"正在执行{requested_analysis_config['label']}..."):
                final_state = await run_financial_workflow(
                    initial_state,
                    [requested_analysis_type],
                    data_only=True,
                )
            logger.info(
                f"Single agent execution completed successfully: "
                f"{requested_analysis_config['node_name']}"
            )
        else:
            # 调用工作流 - 这是阻塞调用，会等待所有智能体完成
            with ui.status("正在并行执行多智能体分析并生成综合研判..."):
                final_state = await run_financial_workflow(initial_state)
            logger.info("Workflow execution completed successfully")

        # ============================================================================
        # 7. 结果处理和报告生成
        # ============================================================================
        
        # 提取并打印最终报告
        if requested_analysis_config and final_state and final_state.get("data"):
            final_data = final_state["data"]
            result_key = requested_analysis_config["result_key"]
            error_key = requested_analysis_config["error_key"]
            analysis_output = final_data.get(result_key, "")
            analysis_error = final_data.get(error_key)
            workflow_success = bool(analysis_output) and not analysis_error

            ui.result(
                success=workflow_success,
                title=(
                    f"{requested_analysis_config['label']}完成"
                    if workflow_success
                    else f"{requested_analysis_config['label']}失败"
                ),
                company_name=company_name,
                stock_code=initial_data.get("stock_code"),
                elapsed_seconds=time.time() - main_agent_start_time,
                detail=str(analysis_error) if analysis_error else None,
            )

            if analysis_output:
                ui.section(f"{requested_analysis_config['label']}结果")
                ui.content(strip_terminal_emoji(analysis_output))

            execution_logger.log_agent_complete("main", {
                "user_query": user_query,
                "company_name": company_name,
                "stock_code": initial_data.get("stock_code"),
                "requested_analysis_type": requested_analysis_type,
                "analysis_result_key": result_key,
                "analysis_output_length": len(analysis_output),
                "analysis_error": analysis_error,
            }, time.time() - main_agent_start_time, workflow_success, analysis_error)
        elif final_state and final_state.get("data") and "final_report" in final_state["data"]:
            final_data = final_state["data"]
            report_path = final_data.get("report_path")
            decision_error = final_data.get("decision_error")
            is_error_report = bool(
                report_path and os.path.basename(report_path).startswith("error_report_")
            )
            workflow_success = (
                final_state.status.value == "completed"
                and not decision_error
                and not is_error_report
            )

            ui.result(
                success=workflow_success,
                title="分析完成" if workflow_success else "综合报告生成失败",
                company_name=company_name,
                stock_code=initial_data.get("stock_code"),
                elapsed_seconds=time.time() - main_agent_start_time,
                report_path=report_path,
                detail=str(decision_error) if decision_error else None,
            )

            if show_analysis_output:
                ui.section("最终分析报告")
                ui.content(strip_terminal_emoji(final_data["final_report"]))

            # 显示报告文件路径（如果可用）
            if report_path:
                logger.info(
                    f"Report saved to: {report_path}")

                # 记录最终报告到执行日志
                execution_logger.log_final_report(
                    final_data["final_report"],
                    report_path
                )

            execution_logger.log_agent_complete("main", {
                "user_query": user_query,
                "company_name": company_name,
                "stock_code": initial_data.get("stock_code"),
                "report_path": report_path,
                "decision_error": decision_error,
                "is_error_report": is_error_report,
                "final_report_length": len(final_data.get("final_report", ""))
            }, time.time() - main_agent_start_time, workflow_success, decision_error)
        else:
            workflow_success = False
            expected_result = (
                requested_analysis_config["result_key"]
                if requested_analysis_config
                else "final_report"
            )
            final_data = final_state.data if final_state else {}
            gate = final_data.get("quality_gate") or {}
            failure_detail = (
                final_data.get("decision_error")
                or final_data.get("risk_review_error")
                or (
                    f"质量门未通过: {gate.get('failures')}"
                    if gate and not gate.get("passed")
                    else f"无法从工作流中检索结果: {expected_result}"
                )
            )
            ui.result(
                success=False,
                title="分析已停止",
                company_name=company_name,
                stock_code=initial_data.get("stock_code"),
                elapsed_seconds=time.time() - main_agent_start_time,
                detail=str(failure_detail),
            )
            logger.error(
                "Workflow stopped without %s: %s", expected_result, failure_detail
            )
            execution_logger.log_agent_complete("main", {
                "user_query": user_query,
                "company_name": company_name,
                "stock_code": initial_data.get("stock_code"),
                "requested_analysis_type": requested_analysis_type,
                "stop_reason": final_state.stop_reason.value if final_state and final_state.stop_reason else None,
                "failure_detail": failure_detail,
            }, time.time() - main_agent_start_time, False, failure_detail)

        # 完成执行日志记录
        finalize_execution_logger(
            success=workflow_success,
            error=None if workflow_success else "Final report generation failed"
        )
        ui.muted(f"执行日志：{execution_logger.execution_dir}")

    except Exception as e:
        # ============================================================================
        # 8. 错误处理
        # ============================================================================
        
        ui.result(
            success=False,
            title="工作流执行失败",
            elapsed_seconds=time.time() - main_agent_start_time,
            detail=str(e),
        )
        logger.error(f"Error during workflow execution: {e}", exc_info=True)

        execution_logger.log_agent_complete("main", {
            "error": str(e)
        }, time.time() - main_agent_start_time, False, str(e))

        # 记录错误并完成执行日志
        finalize_execution_logger(success=False, error=str(e))
        ui.muted(f"错误日志：{get_execution_logger().execution_dir}")


EXIT_COMMANDS = frozenset(
    {"退出", "结束", "再见", "exit", "quit", "q", "bye", "/quit", "/exit"}
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Financial Agent CLI")
    parser.add_argument(
        "--command",
        type=str,
        help="单次金融分析查询；省略时进入持续交互模式。",
    )
    parser.add_argument(
        "--show-analysis-output",
        action="store_true",
        help="在终端显示完整分析结果。默认只显示摘要和报告路径。",
    )
    return parser


def is_exit_command(value: str) -> bool:
    return value.strip().casefold() in EXIT_COMMANDS


def print_session_welcome(ui: TerminalUI | None = None) -> None:
    (ui or TerminalUI()).welcome()


def handle_session_command(
    query: str,
    *,
    ui: TerminalUI,
    history: list[str],
    verbose_output: bool,
) -> tuple[bool, bool]:
    """处理斜杠命令，返回 ``(是否已处理, verbose 状态)``。"""
    normalized = query.strip().casefold()
    if not normalized.startswith("/"):
        return False, verbose_output

    if normalized in {"/help", "/?"}:
        ui.help()
        return True, verbose_output
    if normalized == "/history":
        ui.history(history)
        return True, verbose_output
    if normalized == "/clear":
        ui.clear()
        return True, verbose_output
    if normalized == "/verbose":
        ui.info(f"完整分析输出当前为：{'开启' if verbose_output else '关闭'}")
        return True, verbose_output
    if normalized in {"/verbose on", "/verbose true"}:
        ui.success("已开启完整分析输出。")
        return True, True
    if normalized in {"/verbose off", "/verbose false"}:
        ui.success("已关闭完整分析输出。")
        return True, False

    ui.error(f"未知命令：{query}。输入 /help 查看可用命令。")
    return True, verbose_output


async def run_interactive_session(
    *,
    show_analysis_output: bool = False,
    input_func=input,
    analyze_func=None,
) -> None:
    """Continuously accept independent queries until the user exits."""
    ui = TerminalUI()
    print_session_welcome(ui)
    analyze = analyze_func or run_query
    history: list[str] = []
    verbose_output = show_analysis_output

    while True:
        try:
            raw_query = ui.read_query(input_func)
        except (EOFError, KeyboardInterrupt):
            ui.muted("\n会话已结束。")
            return

        query = str(raw_query).strip()
        if not query:
            ui.error("输入不能为空。示例：分析贵州茅台 600519")
            continue
        if is_exit_command(query):
            ui.success("会话已结束，感谢使用。")
            return

        handled, verbose_output = handle_session_command(
            query,
            ui=ui,
            history=history,
            verbose_output=verbose_output,
        )
        if handled:
            continue

        history.append(query)
        await analyze(query, show_analysis_output=verbose_output)
        ui.next_hint()


async def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command:
            await run_query(
                args.command,
                show_analysis_output=args.show_analysis_output,
            )
        else:
            await run_interactive_session(
                show_analysis_output=args.show_analysis_output,
            )
    finally:
        await close_mcp_client_sessions()


# ============================================================================
# 程序入口点
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())
