"""Compatibility CLI backed by the canonical LangGraph workflow.

The former standalone Planner/Executor loop is retained as reusable infrastructure,
but this entry point no longer starts a second production orchestration engine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.mcp_client import close_mcp_client_sessions
from src.utils.analysis_intent import infer_single_analysis_type
from src.utils.listing_verification import (
    normalize_a_share_symbol as normalize_verified_symbol,
    verify_a_share_listing,
)
from src.utils.state_definition import WorkflowState
from src.workflow import run_financial_workflow


def normalize_a_share_symbol(value: str | None, task: str) -> str | None:
    candidate = value
    if not candidate:
        match = re.search(r"(?<!\d)(\d{6})(?!\d)", task)
        candidate = match.group(1) if match else None
    if not candidate:
        return None
    return normalize_verified_symbol(candidate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Financial MCP Agent Workflow")
    parser.add_argument("--command", required=True, help="金融分析任务")
    parser.add_argument("--symbol", help="A 股代码，如 600519 或 sh.600519")
    parser.add_argument("--company", help="公司名称，用于新闻查询和证据展示")
    parser.add_argument(
        "--allowed-symbol",
        action="append",
        default=[],
        help="估值 Agent 可查询的同行 A 股代码；可重复指定",
    )
    parser.add_argument(
        "--benchmark-symbol",
        action="append",
        default=[],
        help="允许查询的基准证券代码；可重复指定",
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--tool-timeout", type=float, default=30.0)
    parser.add_argument("--trace-dir", default="logs/workflow")
    parser.add_argument(
        "--data-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过风险审查与决策 Agent，只输出质量门后的专业分析、证据与追踪索引",
    )
    parser.add_argument(
        "--show-data", action="store_true", help="显示完整分析内容和工具调用明细"
    )
    return parser


async def run_from_args(args: argparse.Namespace) -> WorkflowState:
    symbol = normalize_a_share_symbol(args.symbol, args.command)
    listing_check = await verify_a_share_listing(
        {
            "company_name": args.company,
            "stock_code": symbol,
            "llm_market": "A股",
            "needs_clarification": False,
        }
    )
    if not listing_check.may_start_workflow:
        raise ValueError(listing_check.message)
    symbol = listing_check.stock_code
    company_name = listing_check.company_name
    allowed_symbols = [
        normalize_verified_symbol(item)
        for item in getattr(args, "allowed_symbol", [])
        if normalize_verified_symbol(item) != symbol
    ]
    benchmark_symbols = [
        normalize_verified_symbol(item)
        for item in getattr(args, "benchmark_symbol", [])
        if normalize_verified_symbol(item) != symbol
    ]
    now = datetime.now()
    state = WorkflowState(
        task=args.command,
        symbol=symbol,
        target_symbol=symbol,
        allowed_symbols=allowed_symbols,
        benchmark_symbols=benchmark_symbols,
        company_name=company_name,
        as_of=args.as_of,
        max_iterations=args.max_iterations,
        max_retries_per_step=args.max_retries,
        data={
            "query": args.command,
            "stock_code": symbol,
            "target_symbol": symbol,
            "allowed_symbols": allowed_symbols,
            "benchmark_symbols": benchmark_symbols,
            "company_name": company_name,
            "current_date": args.as_of.isoformat(),
            "current_time_info": now.isoformat(timespec="seconds"),
            "tool_timeout": args.tool_timeout,
            "trace_dir": args.trace_dir,
            "data_only": args.data_only,
            "listing_preflight": listing_check.model_dump(mode="json"),
        },
    )
    single_analysis = infer_single_analysis_type(args.command)
    selected = [single_analysis] if single_analysis else None
    return await run_financial_workflow(
        state,
        selected,
        data_only=args.data_only,
    )


def _terminal_output(state: WorkflowState, *, show_data: bool) -> dict:
    output = dict(state.output)
    if show_data:
        return output
    summaries = {}
    for key, result in output.get("analysis_results", {}).items():
        summaries[key] = {
            "agent_id": result["agent_id"],
            "success": result["success"],
            "confidence": result["confidence"],
            "quality_passed": result.get("quality_passed", False),
            "quality_score": result.get("quality_score", 0.0),
            "data_completeness": result.get("data_completeness", 0.0),
            "quality_issues": result.get("quality_issues", []),
            "evidence_count": len(result.get("evidence", [])),
            "tool_call_count": len(result.get("tool_calls", [])),
            "error": result.get("error"),
        }
    output["analysis_results"] = summaries
    return output


async def main() -> int:
    args = build_parser().parse_args()
    try:
        state = await run_from_args(args)
        print(
            json.dumps(
                _terminal_output(state, show_data=args.show_data),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        print(f"Trace: {Path(args.trace_dir).resolve() / state.run_id / 'trace.jsonl'}")
        return 0 if state.status.value == "completed" else 1
    except ValueError as error:
        print(f"参数错误: {error}", file=sys.stderr)
        return 2
    finally:
        await close_mcp_client_sessions()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
