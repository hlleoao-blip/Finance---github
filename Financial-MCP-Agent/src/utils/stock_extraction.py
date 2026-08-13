"""
股票信息混合提取工具。

流程：
1. 先用规则提取明确的公司名和股票代码，保证常见格式低成本、确定性强。
2. 当规则结果不完整时，再调用 LLM 对原始用户问题做结构化 JSON 提取。
3. 对 LLM 输出进行本地解析、清洗和置信度校验，避免把自然语言回答直接当事实。
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from src.utils.retry_utils import ainvoke_with_retry


COMPANY_STOP_WORDS = [
    "的",
    "这个",
    "这只",
    "一下",
    "看看",
    "了解",
    "分析",
    "帮我",
    "我想",
    "给我",
    "财务状况",
    "投资价值",
    "基本面情况",
    "这只股票",
    "这个股票",
]

# 规则提取有时会把口语化的查询动作一并捕获，例如把“看宇树科技”当成公司名。
# 这里只清理名称开头的指令词，避免误删公司名中间的正常汉字。
COMPANY_QUERY_PREFIX = re.compile(
    r"^(?:(?:请|麻烦)?(?:帮我|给我)?(?:看看?|查(?:看|询)?(?:一下)?|研究(?:一下)?))"
)

LLM_COMPANY_CONFIDENCE_THRESHOLD = 0.5
LLM_STOCK_CODE_CONFIDENCE_THRESHOLD = 0.75


def clean_company_name(company_name: Optional[str]) -> Optional[str]:
    """清理规则或 LLM 提取出的公司名称。"""
    if not company_name:
        return None

    cleaned = str(company_name).strip().strip("\"'`，。；;：:")
    for word in COMPANY_STOP_WORDS:
        cleaned = cleaned.replace(word, "").strip()
    cleaned = COMPANY_QUERY_PREFIX.sub("", cleaned).strip()

    if len(cleaned) < 2:
        return None
    return cleaned


def clean_stock_code(stock_code: Optional[str]) -> Optional[str]:
    """只接受 5-6 位股票代码，兼容 sh.600519、600519.SH、00700.HK 等格式。"""
    if not stock_code:
        return None

    match = re.search(r"(?<!\d)(\d{5,6})(?!\d)", str(stock_code))
    if not match:
        return None
    return match.group(1)


def rule_extract_stock_info(query: str) -> Tuple[Optional[str], Optional[str]]:
    """用确定性规则提取股票代码和公司名称。"""
    stock_code = None
    company_name = None

    # 模式1: 包含"请帮我分析一下"的复杂查询，如"请帮我分析一下嘉友国际(603871)这只股票的投资价值如何"
    pattern1 = r"请帮我分析一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]"
    match1 = re.search(pattern1, query)
    if match1:
        company_name = match1.group(1).strip()
        stock_code = match1.group(2)
        return clean_company_name(company_name), clean_stock_code(stock_code)

    # 模式2: 包含"分析一下"的复杂查询，如"分析一下嘉友国际(603871)的财务状况"
    pattern2 = r"分析一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]"
    match2 = re.search(pattern2, query)
    if match2:
        company_name = match2.group(1).strip()
        stock_code = match2.group(2)
        return clean_company_name(company_name), clean_stock_code(stock_code)

    # 模式3: 股票代码在括号内，如"分析嘉友国际(603871)"
    pattern3 = r"分析\s*([^（(]+?)\s*[（(](\d{5,6})[)）]"
    match3 = re.search(pattern3, query)
    if match3:
        company_name = match3.group(1).strip()
        stock_code = match3.group(2)
        return clean_company_name(company_name), clean_stock_code(stock_code)

    # 模式4: 股票代码在括号内，如"分析(603871)嘉友国际"
    pattern4 = r"分析\s*[（(](\d{5,6})[)）]\s*([^）)]+)"
    match4 = re.search(pattern4, query)
    if match4:
        stock_code = match4.group(1)
        company_name = match4.group(2).strip()
        return clean_company_name(company_name), clean_stock_code(stock_code)

    # 模式5: 包含"帮我看看"的查询，如"帮我看看(000001)平安银行这只股票"
    pattern5 = r"帮我看看\s*[（(](\d{5,6})[)）]\s*([^）)]+?)(?:\s*这只|\s*这个)?\s*股票"
    match5 = re.search(pattern5, query)
    if match5:
        stock_code = match5.group(1)
        company_name = match5.group(2).strip()
        return clean_company_name(company_name), clean_stock_code(stock_code)

    # 模式6: 包含"我想了解一下"的查询，如"我想了解一下比亚迪(002594)的投资价值"
    pattern6 = r"我想了解一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]"
    match6 = re.search(pattern6, query)
    if match6:
        company_name = match6.group(1).strip()
        stock_code = match6.group(2)
        return clean_company_name(company_name), clean_stock_code(stock_code)

    # 模式7: 包含"帮我看看"的复杂查询，如"帮我看看茅台(600519)这只股票值得投资吗"
    pattern7 = r"帮我看看\s*([^（(]+?)\s*[（(](\d{5,6})[)）]"
    match7 = re.search(pattern7, query)
    if match7:
        company_name = match7.group(1).strip()
        stock_code = match7.group(2)
        return clean_company_name(company_name), clean_stock_code(stock_code)

    # 模式8: 直接公司名+括号格式，如"平安银行(000001)值得买吗"
    pattern8 = r"^([^（(]+?)\s*[（(](\d{5,6})[)）]"
    match8 = re.search(pattern8, query)
    if match8:
        company_name = match8.group(1).strip()
        stock_code = match8.group(2)
        return clean_company_name(company_name), clean_stock_code(stock_code)

    # 模式9: 包含"分析一下"的查询，如"分析一下宁德时代的财务状况"
    pattern9 = r"分析一下\s*([^0-9（）()\s]+?)(?:\s*的|\s|$)"
    match9 = re.search(pattern9, query)
    if match9:
        company_name = match9.group(1).strip()

    # 模式10: 包含"分析"关键词，如"分析嘉友国际"
    pattern10 = r"分析\s*([^0-9（）()\s]+)"
    match10 = re.search(pattern10, query)
    if match10 and not company_name:
        company_name = match10.group(1).strip()

    # 模式11: 包含"股票"关键词的查询，如"嘉友国际这只股票怎么样"
    pattern11 = r"([^0-9（）()\s]+)\s*(?:这只|这个|的)?\s*股票"
    match11 = re.search(pattern11, query)
    if match11 and not company_name:
        company_name = match11.group(1).strip()

    # 模式12: 包含"投资价值"的查询，如"了解一下腾讯的投资价值"
    pattern12 = r"了解一下\s*([^0-9（）()\s]+?)(?:\s*的|\s|$)"
    match12 = re.search(pattern12, query)
    if match12 and not company_name:
        company_name = match12.group(1).strip()

    # 模式13: 包含"给我分析一下"的查询，如"给我分析一下宁德时代的财务状况"
    pattern13 = r"给我分析一下\s*([^0-9（）()\s]+?)(?:\s*的|\s|$)"
    match13 = re.search(pattern13, query)
    if match13 and not company_name:
        company_name = match13.group(1).strip()

    # 模式14: 包含"的"字的查询，如"嘉友国际的财务表现如何"
    pattern14 = r"([^0-9（）()\s]+?)\s*的\s*(?:财务表现|盈利能力|现金流状况|资产负债情况|技术面|股价走势|技术指标|技术面表现|估值水平|市盈率|市净率|估值|投资风险|风险因素|风险评估|投资价值|股票|基本面情况|基本面|财务状况)"
    match14 = re.search(pattern14, query)
    if match14 and not company_name:
        company_name = match14.group(1).strip()

    # 模式15: 包含"在...中"的查询（无"的"字），如"比亚迪在新能源汽车行业的表现"
    pattern15 = r"([^0-9（）()\s]+?)\s*在\s*[^0-9（）()\s]*\s*中"
    match15 = re.search(pattern15, query)
    if match15 and not company_name:
        company_name = match15.group(1).strip()

    # 模式16: 包含"在...中"的查询，如"嘉友国际在行业中的地位"
    pattern16 = r"([^0-9（）()\s]+?)\s*在\s*[^0-9（）()\s]*\s*中\s*的"
    match16 = re.search(pattern16, query)
    if match16 and not company_name:
        company_name = match16.group(1).strip()

    # 模式17: 包含"面临"的查询，如"比亚迪面临的主要风险"
    pattern17 = r"([^0-9（）()\s]+?)\s*面临"
    match17 = re.search(pattern17, query)
    if match17 and not company_name:
        company_name = match17.group(1).strip()

    # 模式18: 直接包含5-6位数字股票代码
    pattern18 = r"\b(\d{5,6})\b"
    match18 = re.search(pattern18, query)
    if match18:
        stock_code = match18.group(1)

    # 模式19: 包含"值得买"的查询，如"603871 这个股票值得买吗"
    pattern19 = r"(\d{5,6})\s*(?:这个|这只)?\s*股票\s*值得买"
    match19 = re.search(pattern19, query)
    if match19 and not stock_code:
        stock_code = match19.group(1)

    # 模式20: 包含"这个股票最近表现"的查询，如"603871这个股票最近表现怎么样，值得投资吗"
    pattern20 = r"(\d{5,6})\s*这个\s*股票\s*最近表现"
    match20 = re.search(pattern20, query)
    if match20 and not stock_code:
        stock_code = match20.group(1)

    return clean_company_name(company_name), clean_stock_code(stock_code)


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _extract_json_object(text: str) -> Dict[str, Any]:
    """从 LLM 输出中提取第一个 JSON 对象。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM output does not contain a JSON object")
    return json.loads(match.group(0))


def _build_llm_messages(query: str) -> List[Dict[str, str]]:
    system_prompt = """
你是一个金融查询结构化信息抽取器。你的任务是从用户的原始问题中抽取股票/上市公司信息。

只输出一个 JSON 对象，不要输出解释、Markdown 或代码块。JSON 字段必须是：
{
  "company_name": string | null,
  "stock_code": string | null,
  "market": string | null,
  "analysis_intent": string[],
  "confidence": number,
  "needs_clarification": boolean,
  "clarification_reason": string | null
}

规则：
- company_name 使用最可能的上市公司中文简称或常用简称。
- stock_code 只在用户明确给出，或你对该简称对应股票高度确定时填写；不确定时填 null。
- stock_code 保留 5-6 位数字即可，不要添加交易所前缀。
- analysis_intent 可包含：基本面、技术面、估值、新闻、风险、投资价值、行业地位、综合分析。
- confidence 表示你对 company_name 和 stock_code 整体抽取结果的置信度，范围 0 到 1。
- 如果存在多个可能公司、市场或代码，needs_clarification 为 true，并把不确定原因写入 clarification_reason。
- 不要编造不存在的信息。
""".strip()

    user_prompt = f"用户原始问题：{query}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def _llm_extract_stock_info(query: str, logger) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

    if not all([api_key, base_url, model_name]):
        return {
            "llm_used": False,
            "llm_error": "Missing OPENAI_COMPATIBLE_* environment variables",
        }

    messages = _build_llm_messages(query)
    model_config = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 800,
        "api_base": base_url,
    }

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=800,
    )

    start_time = time.time()
    llm_message = await ainvoke_with_retry(
        llm,
        messages,
        logger=logger,
        operation_name="MainAgent stock info extraction",
        max_attempts=3,
        initial_delay=1.0,
        max_delay=8.0,
    )
    execution_time = time.time() - start_time
    raw_response = getattr(llm_message, "content", str(llm_message))
    if not isinstance(raw_response, str):
        raw_response = str(raw_response)
    parsed = _extract_json_object(raw_response)

    confidence = _coerce_confidence(parsed.get("confidence"))
    analysis_intent = parsed.get("analysis_intent") or []
    if isinstance(analysis_intent, str):
        analysis_intent = [analysis_intent]
    elif not isinstance(analysis_intent, list):
        analysis_intent = []

    return {
        "llm_used": True,
        "llm_error": None,
        "llm_raw_response": raw_response,
        "llm_prompt_messages": messages,
        "llm_model_config": model_config,
        "llm_execution_time": execution_time,
        "llm_company_name": clean_company_name(parsed.get("company_name")),
        "llm_stock_code": clean_stock_code(parsed.get("stock_code")),
        "llm_market": parsed.get("market"),
        "llm_analysis_intent": analysis_intent,
        "llm_confidence": confidence,
        "needs_clarification": bool(parsed.get("needs_clarification")),
        "clarification_reason": parsed.get("clarification_reason"),
    }


async def extract_stock_info_hybrid(query: str, logger) -> Dict[str, Any]:
    """
    混合提取链：规则优先，规则不完整时调用 LLM 兜底。

    返回值包含最终 company_name、stock_code，以及提取来源和 LLM 调试信息。
    """
    rule_company_name, rule_stock_code = rule_extract_stock_info(query)
    result: Dict[str, Any] = {
        "company_name": rule_company_name,
        "stock_code": rule_stock_code,
        "source": "rules" if (rule_company_name or rule_stock_code) else "none",
        "rule_company_name": rule_company_name,
        "rule_stock_code": rule_stock_code,
        "llm_used": False,
        "llm_error": None,
        "analysis_intent": [],
        "confidence": 1.0 if (rule_company_name and rule_stock_code) else 0.0,
        "needs_clarification": False,
        "clarification_reason": None,
    }

    if rule_company_name and rule_stock_code:
        return result

    try:
        llm_result = await _llm_extract_stock_info(query, logger)
    except Exception as error:
        logger.warning(f"LLM 股票信息提取失败，继续使用规则结果: {error}")
        result["llm_used"] = True
        result["llm_error"] = str(error)
        return result

    result.update(llm_result)
    if not llm_result.get("llm_used"):
        result["llm_error"] = llm_result.get("llm_error")
        return result

    llm_confidence = llm_result.get("llm_confidence", 0.0)
    llm_company_name = llm_result.get("llm_company_name")
    llm_stock_code = llm_result.get("llm_stock_code")

    if not result["company_name"] and llm_company_name and llm_confidence >= LLM_COMPANY_CONFIDENCE_THRESHOLD:
        result["company_name"] = llm_company_name

    if not result["stock_code"] and llm_stock_code and llm_confidence >= LLM_STOCK_CODE_CONFIDENCE_THRESHOLD:
        result["stock_code"] = llm_stock_code

    result["analysis_intent"] = llm_result.get("llm_analysis_intent") or []
    result["confidence"] = llm_confidence

    has_rule_value = bool(rule_company_name or rule_stock_code)
    has_llm_value = bool(
        (result["company_name"] and result["company_name"] != rule_company_name)
        or (result["stock_code"] and result["stock_code"] != rule_stock_code)
    )
    if has_rule_value and has_llm_value:
        result["source"] = "rules+llm"
    elif has_llm_value:
        result["source"] = "llm"

    return result
