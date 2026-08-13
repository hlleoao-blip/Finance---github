"""
Summary Agent: Consolidates analyses from other agents into a final report.
汇总 Agent：将其他 Agent的分析结果整合成最终报告
"""
import os
import json
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI  # 恢复OpenAI导入
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.retry_utils import ainvoke_with_retry
from src.utils.quality_gate import is_substantive_content
from src.utils.report_renderer import (
    compact_analysis_payload,
    parse_decision_narrative,
    reconcile_analysis_sections,
    reconcile_evidence,
    render_financial_report,
)
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv(override=True)

logger = setup_logger(__name__)


INVESTMENT_HORIZON_PROFILES = {
    "short_term": {
        "label": "短线/短期交易",
        "description": "适用于日内到1个月左右的交易决策，优先关注择时、价格趋势、资金情绪和突发事件。",
        "weights": {
            "technical": 0.40,
            "event": 0.30,
            "fundamental": 0.15,
            "value": 0.15,
        },
    },
    "medium_long_term": {
        "label": "中长期投资",
        "description": "适用于3个月以上的持有或配置决策，优先关注基本面质量、估值安全边际和长期催化。",
        "weights": {
            "fundamental": 0.35,
            "value": 0.35,
            "technical": 0.15,
            "event": 0.15,
        },
    },
    "long_term": {
        "label": "长期价值投资",
        "description": "适用于1年以上的持有决策，重点关注商业模式、盈利质量、估值区间和复利能力。",
        "weights": {
            "fundamental": 0.40,
            "value": 0.40,
            "technical": 0.10,
            "event": 0.10,
        },
    },
    "balanced": {
        "label": "未指定周期/综合判断",
        "description": "用户未明确说明投资周期时采用均衡偏中长期的权重，避免短期噪声主导最终结论。",
        "weights": {
            "fundamental": 0.30,
            "value": 0.30,
            "technical": 0.25,
            "event": 0.15,
        },
    },
}


def infer_investment_horizon(user_query: str) -> str:
    """Infer the investment horizon from explicit words in the user query."""
    normalized_query = user_query.lower()

    medium_long_keywords = [
        "中长期", "中长线", "中期", "半年", "6个月", "六个月",
        "三个月", "3个月", "12个月", "一年", "配置",
    ]
    short_term_keywords = [
        "短线", "短期", "超短", "日内", "t+0", "今天", "今日",
        "明天", "一周", "1周", "几天", "波段", "交易", "择时", "买卖点",
    ]
    long_term_keywords = [
        "长期", "长线", "价值投资", "一年以上", "三年", "3年",
        "五年", "5年", "养老", "复利",
    ]

    if any(keyword in normalized_query for keyword in medium_long_keywords):
        return "medium_long_term"
    if any(keyword in normalized_query for keyword in short_term_keywords):
        return "short_term"
    if any(keyword in normalized_query for keyword in long_term_keywords):
        return "long_term"
    return "balanced"


def format_weight_profile(profile: Dict[str, Any]) -> str:
    """Format the selected weighting profile for the summary prompt."""
    weights = profile["weights"]
    labels = {
        "fundamental": "基本面分析",
        "technical": "技术分析",
        "value": "估值分析",
        "event": "事件分析",
    }
    ordered_keys = ["fundamental", "technical", "value", "event"]
    lines = [
        f"- 投资周期：{profile['label']}",
        f"- 权重逻辑：{profile['description']}",
    ]
    lines.extend(
        f"- {labels[key]}：{weights[key] * 100:.0f}%"
        for key in ordered_keys
    )
    return "\n".join(lines)


def deterministic_decision_update(
    state: AgentState,
    *,
    reason: str,
) -> Dict[str, Any]:
    """Render an evidence-preserving report when the decision LLM is unavailable."""
    current_data = dict(state.data)
    company_name = str(current_data.get("company_name") or state.company_name or "未知公司")
    stock_code = str(current_data.get("stock_code") or state.symbol or "未知证券")
    registry = reconcile_evidence(
        state.analysis_results,
        current_data=current_data,
        symbol=stock_code,
        as_of=current_data.get("current_date") or state.as_of,
    )
    structured_payload = compact_analysis_payload(
        state.analysis_results,
        evidence_registry=registry,
    )
    analyses: dict[str, str] = {}
    labels = {
        "fundamental": "基本面",
        "technical": "技术面",
        "value": "估值",
        "event": "事件面",
    }
    for analysis_type, label in labels.items():
        result = state.analysis_results.get(analysis_type)
        analyses[analysis_type] = (
            result.content
            if result is not None and result.success
            else f"{label}模块未通过质量门，本降级报告不引用其正文结论。"
        )
    gate = current_data.get("quality_gate") or {}
    risk_result = state.analysis_results.get("risk_review")
    risk_review = (
        risk_result.content
        if risk_result is not None and risk_result.success
        else "风险审查不可用；最终报告仅保留专业模块原始结论。"
    )
    narrative = {
        "executive_summary": (
            "最终决策模型未能在时限内完成，系统已生成确定性降级报告。"
            f"质量门状态为 {gate.get('status', 'UNKNOWN')}，"
            f"可用模块覆盖率为 {float(gate.get('coverage') or 0.0):.0%}。"
            "以下内容仅整理已验证的专业分析，不新增事实、评级、目标价或概率。"
        ),
        "integrated_assessment": (
            "自动降级模式不执行主观加权或跨模块补全。各模块结论按原证据边界并列展示；"
            "若模块之间存在口径冲突，以统一证据登记表为准并保留全部口径。"
        ),
        "investment_recommendation": (
            "本次不形成新增投资评级。请先补齐降级证据并复核风险审查列出的触发条件；"
            "在证据恢复前，不应依据本报告生成目标价、概率或高置信度交易指令。"
        ),
    }
    final_report = render_financial_report(
        company_name=company_name,
        stock_code=stock_code,
        narrative=narrative,
        analyses=analyses,
        risk_review=risk_review,
        decision_permissions={
            "rating": False,
            "target_price": False,
            "probability": False,
        },
        structured_payload=structured_payload,
        current_time_info=str(current_data.get("current_time_info") or state.as_of),
        evidence_registry=registry,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_company = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", company_name)
    safe_code = re.sub(r"[^0-9A-Za-z_-]+", "", stock_code)
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reports",
    )
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(
        reports_dir,
        f"degraded_report_{safe_company}_{safe_code}_{timestamp}.md",
    )
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(final_report)
    current_data.update(
        {
            "evidence_registry": registry.model_dump(mode="json"),
            "final_report": final_report,
            "report_path": report_path,
            "decision_status": "completed",
            "decision_fallback": {"used": True, "reason": reason},
            "decision_warning": reason,
        }
    )
    current_data.pop("decision_error", None)
    return {"data": current_data}


def truncate_report_at_baseline_time(report_content: str, current_time_info: str) -> str:
    """
    使用正则表达式截断报告，在"分析基准时间"那一行之后停止
    
    Args:
        report_content: 完整的报告内容
        current_time_info: 当前时间信息
    
    Returns:
        截断后的报告内容
    """
    # 构建多种可能的"分析基准时间"模式
    baseline_patterns = [
        rf'分析基准时间[：:]\s*{re.escape(current_time_info)}',
        rf'分析基准时间[：:]\s*{re.escape(current_time_info)}\s*$',
        rf'基准时间[：:]\s*{re.escape(current_time_info)}',
        rf'时间基准[：:]\s*{re.escape(current_time_info)}',
        rf'分析时间[：:]\s*{re.escape(current_time_info)}',
        rf'报告时间[：:]\s*{re.escape(current_time_info)}',
        rf'生成时间[：:]\s*{re.escape(current_time_info)}',
        rf'更新时间[：:]\s*{re.escape(current_time_info)}',
        rf'数据时间[：:]\s*{re.escape(current_time_info)}',
        rf'分析基准[：:]\s*{re.escape(current_time_info)}'
    ]
    
    # 尝试匹配各种模式
    for pattern in baseline_patterns:
        match = re.search(pattern, report_content, re.MULTILINE | re.IGNORECASE)
        if match:
            # 找到匹配位置，截断到该行的末尾
            end_pos = match.end()
            
            # 查找该行的结束位置（换行符）
            line_end = report_content.find('\n', end_pos)
            if line_end == -1:
                # 如果没有换行符，说明是最后一行，直接截断
                truncated_content = report_content[:end_pos].strip()
            else:
                # 截断到该行结束
                truncated_content = report_content[:line_end].strip()
            
            logger.info(f"截断报告在'分析基准时间'行之后，截断位置: {end_pos}")
            return truncated_content
    
    # 如果没有找到匹配的模式，尝试查找包含时间信息的行
    time_patterns = [
        rf'.*{re.escape(current_time_info)}.*',
        rf'.*{re.escape(current_time_info.split()[0])}.*',  # 只匹配日期部分
        rf'.*{re.escape(current_time_info.split()[1])}.*'   # 只匹配时间部分
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, report_content, re.MULTILINE | re.IGNORECASE)
        if match:
            end_pos = match.end()
            line_end = report_content.find('\n', end_pos)
            if line_end == -1:
                truncated_content = report_content[:end_pos].strip()
            else:
                truncated_content = report_content[:line_end].strip()
            
            logger.info(f"截断报告在时间信息行之后，截断位置: {end_pos}")
            return truncated_content
    
    # 如果都没有找到，返回原始内容
    logger.warning("未找到'分析基准时间'模式，返回原始报告内容")
    return report_content


def load_finr1_model(model_path="/root/code/Finance/FinR1"):
    """加载FinR1模型"""
    logger.info(f"{WAIT_ICON} Loading FinR1 model from {model_path}...")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        model.eval()
        logger.info(f"{SUCCESS_ICON} FinR1 model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"{ERROR_ICON} Failed to load FinR1 model: {e}")
        raise e


def generate_report_with_finr1(model, tokenizer, prompt, max_new_tokens=5000):
    """使用FinR1模型生成报告"""
    
    try:
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成预测
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的报告部分（移除输入提示）
        # 方法1：尝试通过字符串匹配移除输入提示
        if prompt in generated_text:
            report = generated_text[len(prompt):].strip()
        else:
            # 方法2：如果字符串匹配失败，尝试通过token长度来提取
            input_length = len(tokenizer.encode(prompt, return_tensors="pt")[0])
            output_length = len(outputs[0])
            
            if output_length > input_length:
                # 只保留新生成的部分
                new_tokens = outputs[0][input_length:]
                report = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            else:
                # 如果无法确定，返回完整文本但尝试清理
                report = generated_text.strip()
        
        return report
    
    except Exception as e:
        logger.error(f"{ERROR_ICON} Error generating report with FinR1: {e}")
        raise e


def get_model_choice():
    """获取模型选择，默认选择API"""
    # 可以通过环境变量控制模型选择
    model_choice = os.getenv("USE_LOCAL_MODEL", "api").lower()
    return model_choice


async def decision_agent(state: AgentState) -> Dict[str, Any]:
    """
    整合基本面、技术面和估值分析的结果
    使用LLM生成最终的综合性报告
    """
    logger.info(f"{WAIT_ICON} DecisionAgent: Starting evidence-gated decision synthesis.")

    # 获取执行日志记录器，用于记录 Agent的执行过程
    execution_logger = get_execution_logger()
    agent_name = "decision_agent"

    # 从状态中提取当前数据、消息和用户查询
    current_data = state.get("data", {})
    messages = state.get("messages", [])
    structured_results = state.get("analysis_results", {})
    stock_code = current_data.get("stock_code", state.symbol or "Unknown Stock")
    evidence_registry = reconcile_evidence(
        structured_results,
        current_data=current_data,
        symbol=stock_code,
        as_of=current_data.get("current_date") or state.as_of,
    )
    state.evidence_registry = evidence_registry
    current_data["evidence_registry"] = evidence_registry.model_dump(mode="json")
    user_query = current_data.get("query", "")
    investment_horizon_key = infer_investment_horizon(user_query)
    investment_horizon_profile = INVESTMENT_HORIZON_PROFILES[investment_horizon_key]
    investment_weight_prompt = format_weight_profile(investment_horizon_profile)
    current_data["investment_horizon"] = investment_horizon_key
    current_data["investment_horizon_label"] = investment_horizon_profile["label"]
    current_data["analysis_weights"] = investment_horizon_profile["weights"]

    quality_gate = current_data.get("quality_gate") or {}
    risk_result = structured_results.get("risk_review")
    if not quality_gate.get("passed") or not risk_result or not risk_result.success:
        error = "Decision blocked: quality gate or risk review did not pass."
        current_data["decision_error"] = error
        execution_logger.log_agent_start(
            agent_name,
            {
                "quality_gate": quality_gate,
                "risk_review_success": bool(risk_result and risk_result.success),
            },
        )
        execution_logger.log_agent_complete(agent_name, current_data, 0.0, False, error)
        return {"data": current_data}

    risk_review = reconcile_analysis_sections(
        {"risk_review": risk_result.content}, evidence_registry
    )["risk_review"]
    decision_permissions = (current_data.get("risk_audit") or {}).get(
        "decision_permissions", {}
    )

    # 记录 Agent开始执行，包含可用的分析类型
    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "investment_horizon": investment_horizon_key,
        "investment_horizon_label": investment_horizon_profile["label"],
        "analysis_weights": investment_horizon_profile["weights"],
        "available_analyses": {
            "fundamental": "fundamental" in structured_results or "fundamental_analysis" in current_data,
            "technical": "technical" in structured_results or "technical_analysis" in current_data,
            "value": "value" in structured_results or "value_analysis" in current_data,
            "event": "event" in structured_results or "event_analysis" in current_data
        },
        "input_data_keys": list(current_data.keys())
    })

    # 记录 Agent开始时间，用于计算执行时长
    agent_start_time = time.time()

    # 获取之前 Agent的分析结果
    def result_content(analysis_type: str, legacy_key: str) -> str:
        result = structured_results.get(analysis_type)
        if result is not None:
            success = result.success if hasattr(result, "success") else result.get("success", False)
            if not success:
                return "该模块未通过质量门，最终报告不得引用其正文结论。"
            return result.content if hasattr(result, "content") else result.get("content", "Not available")
        return current_data.get(legacy_key, "Not available")

    fundamental_analysis = result_content("fundamental", "fundamental_analysis")
    technical_analysis = result_content("technical", "technical_analysis")
    value_analysis = result_content("value", "value_analysis")
    event_analysis = result_content("event", "event_analysis")

    # 处理各个分析的错误信息
    errors = []
    if "fundamental_analysis_error" in current_data:
        errors.append(
            f"Fundamental Analysis Error: {current_data['fundamental_analysis_error']}")
    if "technical_analysis_error" in current_data:
        errors.append(
            f"Technical Analysis Error: {current_data['technical_analysis_error']}")
    if "value_analysis_error" in current_data:
        errors.append(
            f"Value Analysis Error: {current_data['value_analysis_error']}")
    if "event_analysis_error" in current_data:
        errors.append(
            f"Event Analysis Error: {current_data['event_analysis_error']}")

    # 基本股票标识信息
    company_name = current_data.get("company_name", "Unknown Company")

    try:
        # 获取模型选择
        model_choice = get_model_choice()
        logger.info(f"{WAIT_ICON} DecisionAgent: Using model choice: {model_choice}")

        # 获取当前时间信息，用于报告中的时间标注
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")

        structured_payload = compact_analysis_payload(
            structured_results, evidence_registry=evidence_registry
        )
        analysis_sections = reconcile_analysis_sections({
            "fundamental": fundamental_analysis,
            "technical": technical_analysis,
            "value": value_analysis,
            "event": event_analysis,
        }, evidence_registry)
        system_prompt = (
            "你是面向普通投资者的金融决策分析师。只生成三个决策性段落，不复述完整专业报告。"
            "只能使用输入JSON，不得补写数字。严格服从风险审查和决策权限。"
            "统一证据登记表是跨Agent取数状态的唯一依据：任一业务键为SUCCESS时不得称其未获取；"
            "NOT_ATTEMPTED不得覆盖SUCCESS；多源冲突必须并列陈述全部口径，不得自行选值；"
            "这些内部规则只用于约束推理，输出不得出现业务键、call_id、工具函数名、错误码、"
            "PASS/DEGRADED/INVALID_DATA等系统状态；数据局限应改写成自然、易懂的中文。"
            "质量门为DEGRADED时必须显式披露缺失模块或证据类别，降低结论强度，"
            "且不得用其他模块、模型记忆或常识填补缺失事实。"
            "只返回一个JSON对象，字段必须为 executive_summary、integrated_assessment、"
            "investment_recommendation；字段值使用中文Markdown，可含列表但不要含二级标题。"
        )
        user_prompt = json.dumps(
            {
                "company": company_name,
                "stock_code": stock_code,
                "as_of": current_date,
                "original_query": user_query,
                "investment_profile": {
                    "horizon": investment_horizon_key,
                    "weights": investment_horizon_profile["weights"],
                },
                "analyses": structured_payload,
                "risk_review": risk_review[:3500],
                "decision_permissions": decision_permissions,
                "quality_gate": quality_gate,
                "analysis_issues": errors,
            },
            ensure_ascii=False,
            default=str,
        )

        # 根据模型选择决定使用哪种方式生成报告
        if model_choice == "local":
            # 使用本地FinR1模型
            logger.info(f"{WAIT_ICON} SummaryAgent: Using local FinR1 model...")
            
            # 记录模型配置信息
            model_config = {
                "model": "FinR1",
                "temperature": 0.5,
                "max_tokens": 2200,
                "model_path": "/root/code/Finance/FinR1",
                "investment_horizon": investment_horizon_key,
                "analysis_weights": investment_horizon_profile["weights"]
            }

            # 加载FinR1模型
            model, tokenizer = load_finr1_model()

            # 组合完整的提示词
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # 记录LLM交互开始时间
            llm_start_time = time.time()

            # 使用FinR1模型生成最终报告
            decision_output = generate_report_with_finr1(
                model, tokenizer, full_prompt, max_new_tokens=2200
            )

            # 记录LLM交互执行时间
            llm_execution_time = time.time() - llm_start_time

        else:
            # 默认使用API接口
            logger.info(f"{WAIT_ICON} SummaryAgent: Using OpenAI API...")
            
            # 创建OpenAI模型（使用直接API调用，而不是ReAct框架进行汇总）
            api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
            base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

            # 验证必要的环境变量是否存在
            if not all([api_key, base_url, model_name]):
                logger.error(
                    f"{ERROR_ICON} SummaryAgent: Missing OpenAI environment variables.")
                current_data["decision_error"] = "Missing OpenAI environment variables."

                # 记录 Agent执行失败
                execution_logger.log_agent_complete(agent_name, current_data, time.time(
                ) - agent_start_time, False, "Missing OpenAI environment variables")

                return {"data": current_data}

            # 记录模型配置信息
            model_config = {
                "model": model_name,
                "temperature": 0.5,
                "max_tokens": 2200,
                "api_base": base_url,
                "investment_horizon": investment_horizon_key,
                "analysis_weights": investment_horizon_profile["weights"]
            }

            # 准备汇总提示词消息列表
            summary_prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 使用ChatOpenAI模型
            logger.info(f"{WAIT_ICON} SummaryAgent: Creating ChatOpenAI with model {model_name}")
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.5,  # 提高温度以增加创造性和更自然的表达
                max_tokens=2200
            )

            # 记录LLM交互开始时间
            llm_start_time = time.time()

            # 调用LLM生成最终报告
            llm_message = await ainvoke_with_retry(
                llm,
                summary_prompt_messages,
                logger=logger,
                operation_name="SummaryAgent final report generation"
            )
            decision_output = llm_message.content

            # 记录LLM交互执行时间
            llm_execution_time = time.time() - llm_start_time

        if not is_substantive_content(str(decision_output), minimum_length=80):
            raise ValueError("Decision model returned incomplete structured narrative.")
        narrative = parse_decision_narrative(decision_output)
        narrative = reconcile_analysis_sections(narrative, evidence_registry)
        final_report = render_financial_report(
            company_name=company_name,
            stock_code=stock_code,
            narrative=narrative,
            analyses=analysis_sections,
            risk_review=risk_review,
            decision_permissions=decision_permissions,
            structured_payload=structured_payload,
            current_time_info=current_time_info,
            evidence_registry=evidence_registry,
        )

        # 记录LLM交互详情，用于后续分析和优化
        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="summary_generation",
            input_messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            output_content=str(decision_output),
            model_config=model_config,
            execution_time=llm_execution_time
        )

        # 移除任何可能出现的markdown代码块标记
        final_report = final_report.replace(
            "```markdown", "").replace("```", "").strip()
        if not is_substantive_content(final_report, minimum_length=300):
            raise ValueError("Decision model returned incomplete or non-substantive content.")
        
        # 使用正则表达式截断"分析基准时间"那一行之后的内容
        final_report = truncate_report_at_baseline_time(final_report, current_time_info)

        logger.info(
            f"{SUCCESS_ICON} SummaryAgent: Final report generated for {company_name} ({stock_code}).")
        logger.debug(f"Final report preview: {final_report[:300]}...")

        # 将报告保存到Markdown文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 处理公司名称和股票代码，确保文件名有意义
        if stock_code == "Unknown Stock" or stock_code == "Extracted from analysis":
            # 从用户查询中提取更有意义的名称
            query_based_name = user_query.replace(
                " ", "_").replace("分析", "").strip()
            if not query_based_name:
                query_based_name = "financial_analysis"
            safe_file_prefix = f"report_{query_based_name}"
        else:
            # 正常情况下使用公司名称和股票代码
            safe_company_name = company_name.replace(" ", "_").replace(".", "")
            if safe_company_name == "Unknown_Company" or safe_company_name == "Extracted_from_analysis":
                safe_company_name = user_query.replace(
                    " ", "_").replace("分析", "").strip()
                if not safe_company_name:
                    safe_company_name = "company"

            # 清理股票代码（移除可能的前缀）
            clean_stock_code = stock_code.replace("sh.", "").replace("sz.", "")
            safe_file_prefix = f"report_{safe_company_name}_{clean_stock_code}"

        report_filename = f"{safe_file_prefix}_{timestamp}.md"

        # 确保reports目录存在
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, report_filename)

        # 将报告写入文件
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        logger.info(
            f"{SUCCESS_ICON} SummaryAgent: Report saved to {report_path}")

        # 返回更新后的状态，包含最终报告
        current_data["final_report"] = final_report
        current_data["report_path"] = report_path
        current_data["decision_status"] = "completed"

        # 记录 Agent执行成功
        total_execution_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {
            "final_report_length": len(final_report),
            "report_path": report_path,
            "report_preview": final_report,
            "llm_execution_time": llm_execution_time,
            "total_execution_time": total_execution_time
        }, total_execution_time, True)

        return {"data": current_data}

    except Exception as e:
        logger.error(
            f"{ERROR_ICON} SummaryAgent: Error generating final report: {e}", exc_info=True)
        fallback_reason = f"Decision generation failed: {e}"

        # Preserve all validated specialist work even when the decision LLM fails.
        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - agent_start_time, True, fallback_reason)

        return deterministic_decision_update(state, reason=fallback_reason)


# Backward-compatible import name; the production graph uses ``decision_agent``.
summary_agent = decision_agent


# 本地测试函数
async def test_summary_agent():
    """汇总 Agent的测试函数"""
    from src.utils.state_definition import AgentState

    # 用于测试的示例状态，包含模拟分析结果
    test_state = AgentState(
        messages=[],
        data={
            "query": "分析嘉友国际",
            "stock_code": "603871",
            "company_name": "嘉友国际",
            "fundamental_analysis": "嘉友国际基本面分析：公司主营业务为跨境物流、供应链贸易以及供应链增值服务。财务状况良好，负债率较低，现金流充裕。近年来业绩稳步增长，毛利率保持在行业较高水平。",
            "technical_analysis": "嘉友国际技术分析：短期内股价处于上升通道，突破了200日均线。RSI指标显示股票尚未达到超买区域。MACD指标呈现多头形态，成交量有所放大，支持价格继续上行。",
            "value_analysis": "嘉友国际估值分析：当前市盈率为15倍，低于行业平均水平。市净率为1.8倍，处于合理区间。与同行业公司相比，嘉友国际的估值较为合理，具有一定的投资价值。",
            "event_analysis": "嘉友国际事件分析：近期公司发布了业绩预告和合作公告，相关事实均需按公告日期与来源复核，未验证内容不进入最终结论。"
        },
        metadata={}
    )

    # 运行 Agent并输出结果
    result = await summary_agent(test_state)
    print("Summary Report:")
    print(result.get("data", {}).get("final_report", "No report generated"))
    print(
        f"Report saved to: {result.get('data', {}).get('report_path', 'Not saved')}")

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_summary_agent())
