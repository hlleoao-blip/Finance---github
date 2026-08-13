"""
ValueAnalysis Agent: Performs valuation analysis of a stock using ReAct Agent framework.
估值分析 Agent：使用ReAct Agent框架对股票进行估值分析
"""
import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # 恢复OpenAI接口调用
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.prebuilt import create_react_agent
import time

from src.utils.state_definition import AgentState, AnalysisResult, failed_analysis_update
from src.tools.mcp_client import get_tools_for_agent
from src.tools.react_tooling import AuditedToolGateway, serialize_react_messages
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.retry_utils import ainvoke_with_retry
from src.utils.terminal_output import print_analysis_output
from src.utils.quality_gate import build_analysis_result
from src.agents.react_output import extract_react_output, react_iteration_limit
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv(override=True)

logger = setup_logger(__name__)


async def value_agent(state: AgentState) -> AgentState:
    """
    使用ReAct框架进行估值分析，直接集成MCP工具
    
    Args:
        state: 包含用户查询的当前 Agent状态

    Returns:
        更新后的AgentState，包含估值分析结果
    """
    logger.info(
        f"{WAIT_ICON} ValueAgent: Starting valuation analysis using ReAct framework.")

    # 获取执行日志记录器，用于记录 Agent的执行过程
    execution_logger = get_execution_logger()
    agent_name = "value_agent"

    # 从状态中提取当前数据、消息和元数据
    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})
    user_query = current_data.get("query")

    # 记录 Agent开始执行，包含关键信息
    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "stock_code": current_data.get("stock_code"),
        "company_name": current_data.get("company_name"),
        "input_data_keys": list(current_data.keys())
    })

    # 验证用户查询是否存在
    if not user_query:
        logger.error(
            f"{ERROR_ICON} ValueAgent: User query is missing in state data.")
        current_data["value_analysis_error"] = "User query is missing."

        # 记录 Agent执行失败
        execution_logger.log_agent_complete(
            agent_name, current_data, 0, False, "User query is missing")

        return failed_analysis_update(agent_id=agent_name, analysis_type="value", data=current_data, messages=current_messages, metadata=current_metadata, error=current_data["value_analysis_error"])

    # 记录 Agent开始时间，用于计算执行时长
    agent_start_time = time.time()

    try:
        # 使用API调用
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

        # 验证必要的环境变量是否存在
        if not all([api_key, base_url, model_name]):
            logger.error(f"{ERROR_ICON} ValueAgent: Missing OpenAI environment variables.")
            current_data["value_analysis_error"] = "Missing OpenAI environment variables."
            execution_logger.log_agent_complete(agent_name, current_data, time.time() - agent_start_time, False, "Missing OpenAI environment variables")
            return failed_analysis_update(agent_id=agent_name, analysis_type="value", data=current_data, messages=current_messages, metadata=current_metadata, error=current_data["value_analysis_error"])

        logger.info(f"{WAIT_ICON} ValueAgent: Creating ChatOpenAI with model {model_name}")
        # 创建LLM实例，设置合适的参数
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,  # 较低的温度确保分析的一致性
            max_tokens=6000   # 增加token数量用于详细分析
        )

        # 2. 获取MCP工具集
        logger.info(f"{WAIT_ICON} ValueAgent: Fetching MCP tools...")
        try:
            raw_mcp_tools = await get_tools_for_agent(agent_name)
            if not raw_mcp_tools:
                logger.error(
                    f"{ERROR_ICON} ValueAgent: No MCP tools available.")
                current_data["value_analysis_error"] = "No MCP tools available."

                # 记录 Agent执行失败
                execution_logger.log_agent_complete(agent_name, current_data, time.time(
                ) - agent_start_time, False, "No MCP tools available")

                return failed_analysis_update(agent_id=agent_name, analysis_type="value", data=current_data, messages=current_messages, metadata=current_metadata, error=current_data["value_analysis_error"])

            gateway = AuditedToolGateway(
                agent_name=agent_name, state=state, tools=raw_mcp_tools
            )
            mcp_tools = gateway.wrap_tools()
            logger.info(
                f"{SUCCESS_ICON} ValueAgent: Successfully loaded {len(mcp_tools)} policy-scoped tools.")

            # 打印可用工具列表，便于调试
            tool_names = [tool.name for tool in mcp_tools]
            logger.info(f"Available tools: {tool_names}")

            # 3. 创建ReAct Agent - 只传入LLM和工具
            logger.info(f"{WAIT_ICON} ValueAgent: Creating ReAct agent...")
            agent = create_react_agent(llm, mcp_tools)

            # 4. 准备输入数据，构建详细的分析请求
            stock_code = current_data.get('stock_code', 'Unknown')
            company_name = current_data.get('company_name', 'Unknown')
            current_time_info = current_data.get('current_time_info', '未知时间')
            current_date = current_data.get('current_date', '未知日期')
            allowed_peers = list(state.allowed_symbols)
            peer_contract = (
                "、".join(allowed_peers) if allowed_peers else "未配置；禁止查询任何其他证券"
            )
            shared_plan = [
                {
                    key: item.get(key)
                    for key in ("requirement", "tool", "arguments", "status", "code")
                    if item.get(key) is not None
                }
                for item in (current_data.get("collection_plan") or [])
                if item.get("tool") in tool_names
            ]

            # 构建详细的估值分析请求，包含多个分析维度
            agent_input = f"""请分析{company_name}（股票代码：{stock_code}）的估值情况。

当前时间：{current_time_info}
当前日期：{current_date}
目标证券：{state.target_symbol or stock_code}
允许的同行证券：{peer_contract}

集中取数清单（目标证券只能按成功项的参数原样读取共享缓存）：
{json.dumps(shared_plan, ensure_ascii=False, default=str)}

请进行以下估值分析：
1. 获取公司基本信息（市值、股价等），可用腾讯快照交叉验证最新价
2. 获取并分析主要估值指标（市盈率、市净率、市销率等），东方财富估值/资金流信号只能作为辅助证据
3. 将估值指标与行业平均水平进行对比分析
4. 分析历史估值水平变化趋势
5. 获取并分析股息数据和股息收益率
6. 计算和分析内在价值
7. 提供估值总结和投资建议

目标证券数据已经按“必需→交叉验证→补充”集中采集。按清单成功项的参数原样调用以登记共享缓存证据；同一财务、行情或分红数据只读一次。清单失败或跳过项不得重试，也不得自行改日期范围规避集中计划。只有明确列入允许列表的同行证券可在预算内补查。
调用上述财务工具时，year 必须传 JSON 字符串（例如 "2025"），quarter 传整数（例如 4）。

重要限制：请专注于估值指标和财务数据分析，不要使用crawl_news工具获取新闻信息。估值分析应该基于财务指标、估值比率和历史数据，而不是新闻事件。除目标证券外，只能查询上面明确列出的同行证券；未列出的证券一律不得查询。

请使用可用的工具获取实际数据进行分析，而不是基于假设。无法获取的数据必须明确标注为缺失，禁止用模型记忆、“公开行业认知”或未经工具核验的历史区间补写。请保持回答简洁，避免冗长的描述性文字"""

            logger.info(f"Agent input: {agent_input}")

            # 5. 调用ReAct Agent - 使用正确的messages格式
            logger.info(f"{WAIT_ICON} ValueAgent: Calling ReAct agent...")
            start_time = time.time()

            # LangGraph ReAct Agent需要messages格式的输入
            input_data = {
                "messages": [HumanMessage(content=agent_input)]
            }

            # 调用 Agent执行分析
            response = await ainvoke_with_retry(
                agent,
                input_data,
                logger=logger,
                operation_name="ValueAgent ReAct execution",
                config={"recursion_limit": react_iteration_limit(state, agent_name)},
            )
            execution_logger.log_react_trajectory(
                agent_name, serialize_react_messages(response)
            )

            # 6. 提取分析结果；迭代用尽时基于已收集证据强制总结
            final_output = await extract_react_output(
                response,
                llm=llm,
                logger=logger,
                operation_name="ValueAgent ReAct execution",
                analysis_name="估值分析",
            )
            end_time = time.time()
            execution_time = end_time - start_time

            logger.info(
                f"ReAct agent execution completed in {execution_time:.2f} seconds")

            logger.info(
                f"Final extracted analysis length: {len(final_output)} characters")
            print_analysis_output(
                "VALUEAGENT",
                final_output,
                current_data.get("show_analysis_output", False)
            )
            # 7. 记录LLM交互，用于后续分析和优化
            model_config = {
                "model": model_name,
                "temperature": 0.3,
                "max_tokens": 6000,
                "api_base": base_url
            }
            
            execution_logger.log_llm_interaction(
                agent_name=agent_name,
                interaction_type="react_agent",
                input_messages=[{"role": "user", "content": agent_input}],
                output_content=final_output,
                model_config=model_config,
                execution_time=execution_time
            )

            # 8. 更新状态，保存分析结果和元数据
            current_data["value_analysis"] = final_output
            current_metadata["value_agent_executed"] = True
            current_metadata["value_agent_timestamp"] = str(time.time())
            current_metadata["value_agent_execution_time"] = f"{execution_time:.2f} seconds"
            analysis_result = build_analysis_result(
                agent_id=agent_name,
                analysis_type="value",
                content=final_output,
                state=state,
                evidence=gateway.evidence,
                tool_calls=gateway.events,
            )
            if not analysis_result.success:
                current_data["value_analysis_error"] = analysis_result.error
                logger.error(
                    "%s ValueAgent: Output rejected by deterministic quality checks: %s",
                    ERROR_ICON,
                    analysis_result.error,
                )
            else:
                logger.info(
                    f"{SUCCESS_ICON} ValueAgent: Successfully completed valuation analysis.")

            # 9. 添加消息记录，保持对话历史
            new_message = {
                "role": "assistant",
                "content": "估值分析已完成" if analysis_result.success else "估值分析未通过质量门",
            }

            # 记录 Agent执行成功
            total_execution_time = time.time() - agent_start_time
            execution_logger.log_agent_complete(agent_name, {
                "value_analysis_length": len(final_output),
                "analysis_preview": final_output[:500] if len(final_output) > 500 else final_output,
                "llm_execution_time": execution_time,
                "total_execution_time": total_execution_time
            }, total_execution_time, analysis_result.success, analysis_result.error)

            return {
                "data": current_data,
                "messages": [new_message],
                "metadata": current_metadata,
                "analysis_results": {"value": analysis_result},
            }

        except Exception as e:
            logger.error(
                f"{ERROR_ICON} ValueAgent: Error in MCP or agent execution: {e}", exc_info=True)
            current_data["value_analysis_error"] = f"Error in MCP or agent execution: {e}"
            current_data["value_analysis"] = f"估值分析过程中出现错误: {str(e)}"
            current_metadata["value_agent_error"] = str(e)

            # 记录 Agent执行失败
            execution_logger.log_agent_complete(
                agent_name, current_data, time.time() - agent_start_time, False, str(e))

            return failed_analysis_update(agent_id=agent_name, analysis_type="value", data=current_data, messages=current_messages, metadata=current_metadata, error=current_data["value_analysis_error"])

    except Exception as e:
        logger.error(
            f"{ERROR_ICON} ValueAgent: Error during execution: {e}", exc_info=True)
        current_data["value_analysis_error"] = f"Error during execution: {e}"
        current_metadata["value_agent_error"] = str(e)

        # 记录 Agent执行失败
        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - agent_start_time, False, str(e))

        return failed_analysis_update(agent_id=agent_name, analysis_type="value", data=current_data, messages=current_messages, metadata=current_metadata, error=current_data["value_analysis_error"])


# 本地测试函数
async def test_value_agent():
    """估值分析 Agent的测试函数"""
    from src.utils.state_definition import AgentState
    from datetime import datetime

    # 准备测试数据，包含当前时间信息
    current_datetime = datetime.now()
    current_date_cn = current_datetime.strftime("%Y年%m月%d日")
    current_date_en = current_datetime.strftime("%Y-%m-%d")
    current_weekday_cn = ["星期一", "星期二", "星期三", "星期四",
                          "星期五", "星期六", "星期日"][current_datetime.weekday()]
    current_time = current_datetime.strftime("%H:%M:%S")
    current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"

    # 创建测试状态，模拟真实的用户查询
    test_state = AgentState(
        messages=[],
        data={
            "query": "分析嘉友国际的估值",
            "stock_code": "sh.603871",
            "company_name": "嘉友国际",
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        },
        metadata={}
    )

    # 运行 Agent并输出结果
    result = await value_agent(test_state)
    print("Valuation Analysis Result:")
    print(result.get("data", {}).get("value_analysis", "No analysis found"))

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_value_agent())
