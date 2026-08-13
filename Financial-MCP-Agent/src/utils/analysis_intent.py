"""单一分析类型意图识别工具。"""
import re


ANALYSIS_KEYWORDS = {
    "fundamental": [
        "基本面", "财务", "财报", "盈利能力", "成长能力", "偿债能力",
        "现金流", "资产负债", "利润表", "roe", "毛利率", "净利率"
    ],
    "technical": [
        "技术分析", "技术面", "k线", "均线", "macd", "rsi",
        "支撑位", "压力位", "阻力位", "价格趋势", "量价", "成交量"
    ],
    "value": [
        "估值", "市盈率", "市净率", "pe", "pb", "ps", "dcf",
        "股息率", "贵不贵", "便宜吗", "安全边际"
    ],
    "event": [
        "新闻", "消息面", "舆情", "情绪", "资讯", "公告", "政策",
        "解禁", "减持", "增持", "利好", "利空", "媒体报道", "重大事件"
    ],
}


def infer_single_analysis_type(user_query: str) -> str | None:
    """用户明确只问单一分析类型时，返回对应 agent 类型。"""
    normalized_query = user_query.lower()

    def keyword_matches(keyword: str) -> bool:
        normalized_keyword = keyword.lower()
        if normalized_keyword.isascii() and normalized_keyword.isalnum():
            pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"
            return re.search(pattern, normalized_query) is not None
        return normalized_keyword in normalized_query

    matched_types = [
        analysis_type
        for analysis_type, keywords in ANALYSIS_KEYWORDS.items()
        if any(keyword_matches(keyword) for keyword in keywords)
    ]
    if len(matched_types) == 1:
        return matched_types[0]
    return None
