"""
股票市场数据工具，用于MCP服务器
"""
import logging
from typing import List, Optional, Callable, Any

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource, NoDataFoundError, LoginError, DataSourceError
from src.formatting.markdown_formatter import format_df_to_markdown
from src.tools.contracts import (
    AdjustFlag,
    CompanyName,
    DividendYearType,
    Frequency,
    ISODate,
    StockCode,
    ToolExecutionError,
    Year,
    contract_tool,
)

logger = logging.getLogger(__name__)


def safe_data_fetch(
    func_name: str,
    data_source_func: Callable,
    *args,
    **kwargs
) -> str:
    """
    安全的数据获取函数，统一处理所有异常和错误情况
    
    参数:
        func_name: 函数名称，用于日志记录
        data_source_func: 数据源函数
        *args: 传递给数据源函数的参数
        **kwargs: 传递给数据源函数的关键字参数
        
    返回:
        Markdown格式的数据表格或错误消息
    """
    try:
        # 调用数据源函数
        df = data_source_func(*args, **kwargs)
        
        # 格式化结果
        logger.info(f"Successfully retrieved data for {func_name}, formatting to Markdown.")
        return format_df_to_markdown(df)
        
    except NoDataFoundError as e:
        logger.warning(f"NoDataFoundError for {func_name}: {e}")
        raise ToolExecutionError("NO_DATA", str(e)) from e
    except LoginError as e:
        logger.error(f"LoginError for {func_name}: {e}")
        raise ToolExecutionError(
            "DATA_SOURCE_LOGIN_ERROR",
            "Could not connect to the data source.",
            retryable=True,
        ) from e
    except DataSourceError as e:
        logger.error(f"DataSourceError for {func_name}: {e}")
        raise ToolExecutionError("DATA_SOURCE_ERROR", str(e), retryable=True) from e
    except ValueError as e:
        logger.warning(f"ValueError processing request for {func_name}: {e}")
        raise ToolExecutionError("INVALID_ARGUMENT", str(e)) from e
    except Exception as e:
        logger.exception(f"Unexpected Exception processing {func_name}: {e}")
        raise ToolExecutionError(
            "INTERNAL_ERROR",
            f"Unexpected failure while executing {func_name}.",
            details={"exception_type": type(e).__name__},
        ) from e


def register_stock_market_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    向MCP应用注册股票市场数据工具

    参数:
        app: FastMCP应用实例
        active_data_source: 活跃的金融数据源
    """

    @contract_tool(app)
    def resolve_stock_listing(
        code: Optional[StockCode] = None,
        company_name: Optional[CompanyName] = None,
    ) -> dict[str, Any]:
        """按代码或公司名解析中国A股证券，并返回结构化上市状态。

        至少提供一个参数。candidates 为空表示当前 A 股数据源没有匹配项，
        并不表示已经证明该公司在全球所有市场均未上市。
        """
        if not code and not company_name:
            raise ToolExecutionError(
                "INVALID_ARGUMENT",
                "code and company_name cannot both be empty.",
            )

        try:
            frame = active_data_source.resolve_stock_listing(
                code=code,
                code_name=company_name,
            )
        except NoDataFoundError:
            frame = None
        except LoginError as error:
            raise ToolExecutionError(
                "DATA_SOURCE_LOGIN_ERROR",
                "Could not connect to the listing data source.",
                retryable=True,
            ) from error
        except DataSourceError as error:
            raise ToolExecutionError(
                "DATA_SOURCE_ERROR",
                str(error),
                retryable=True,
            ) from error

        type_labels = {
            "1": "stock",
            "2": "index",
            "3": "other",
            "4": "convertible_bond",
            "5": "etf",
        }
        candidates = []
        if frame is not None:
            for _, row in frame.iterrows():
                security_type = str(row.get("type", ""))
                if security_type and security_type != "1":
                    continue
                raw_status = str(row.get("status", ""))
                if raw_status == "1":
                    listing_status = "listed"
                elif raw_status == "0":
                    listing_status = "delisted"
                else:
                    listing_status = "unknown"
                raw_code = str(row.get("code", ""))
                candidates.append(
                    {
                        "stock_code": raw_code,
                        "company_name": str(row.get("code_name", "")),
                        "exchange": raw_code.split(".", 1)[0],
                        "ipo_date": str(row.get("ipoDate", "")),
                        "out_date": str(row.get("outDate", "")),
                        "security_type": type_labels.get(
                            security_type,
                            security_type or "unknown",
                        ),
                        "listing_status": listing_status,
                    }
                )

        return {
            "market": "A-share",
            "query": {"stock_code": code, "company_name": company_name},
            "candidates": candidates,
        }

    @contract_tool(app)
    def get_historical_k_data(
        code: StockCode,
        start_date: ISODate,
        end_date: ISODate,
        frequency: Frequency = "d",
        adjust_flag: AdjustFlag = "3",
        fields: Optional[List[str]] = None,
    ) -> str:
        """
        获取中国A股股票的历史K线（OHLCV）数据

        参数:
            code: Baostock格式的股票代码（例如：'sh.600000', 'sz.000001'）
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'
            frequency: 数据频率。有效选项（来自Baostock）：
                         'd': 日线
                         'w': 周线
                         'm': 月线
                         '5': 5分钟
                         '15': 15分钟
                         '30': 30分钟
                         '60': 60分钟
                       默认为'd'
            adjust_flag: 价格/成交量调整标志。有效选项（来自Baostock）：
                           '1': 前复权
                           '2': 后复权
                           '3': 不复权
                         默认为'3'
            fields: 可选的具体数据字段列表（必须是有效的Baostock字段）
                    如果为None或空，将使用默认字段（例如：date, code, open, high, low, close, volume, amount, pctChg）

        返回:
            包含K线数据表的Markdown格式字符串，或错误消息
            如果结果集太大，表格可能会被截断
        """
        logger.info(
            f"Tool 'get_historical_k_data' called for {code} ({start_date}-{end_date}, freq={frequency}, adj={adjust_flag}, fields={fields})")
        
        # 验证频率和调整标志
        valid_freqs = ['d', 'w', 'm', '5', '15', '30', '60']
        valid_adjusts = ['1', '2', '3']
        if frequency not in valid_freqs:
            logger.warning(f"Invalid frequency requested: {frequency}")
            raise ToolExecutionError(
                "INVALID_ARGUMENT",
                "frequency is not supported.",
                details={"field": "frequency", "allowed": valid_freqs},
            )
        if adjust_flag not in valid_adjusts:
            logger.warning(f"Invalid adjust_flag requested: {adjust_flag}")
            raise ToolExecutionError(
                "INVALID_ARGUMENT",
                "adjust_flag is not supported.",
                details={"field": "adjust_flag", "allowed": valid_adjusts},
            )

        # 使用通用函数处理数据获取
        return safe_data_fetch(
            "get_historical_k_data",
            active_data_source.get_historical_k_data,
            code=code,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjust_flag=adjust_flag,
            fields=fields,
        )

    @contract_tool(app)
    def get_stock_basic_info(code: StockCode, fields: Optional[List[str]] = None) -> str:
        """
        获取给定中国A股股票的基本信息

        参数:
            code: Baostock格式的股票代码（例如：'sh.600000', 'sz.000001'）
            fields: 可选列表，用于从可用的基本信息中选择特定列
                    （例如：['code', 'code_name', 'industry', 'listingDate']）
                    如果为None或空，返回Baostock中所有可用的基本信息列

        返回:
            包含基本股票信息表的Markdown格式字符串，或错误消息
        """
        logger.info(
            f"Tool 'get_stock_basic_info' called for {code} (fields={fields})")
        
        # 使用通用函数处理数据获取
        return safe_data_fetch(
            "get_stock_basic_info",
            active_data_source.get_stock_basic_info,
            code=code,
            fields=fields,
        )

    @contract_tool(app)
    def get_dividend_data(
        code: StockCode,
        year: Year,
        year_type: DividendYearType = "report",
    ) -> str:
        """
        获取给定股票代码和年份的分红信息

        参数:
            code: Baostock格式的股票代码（例如：'sh.600000', 'sz.000001'）
            year: 查询年份（例如：'2023'）
            year_type: 年份类型。有效选项（来自Baostock）：
                         'report': 预案公告年份
                         'operate': 除权除息年份
                       默认为'report'

        返回:
            包含分红数据表的Markdown格式字符串，或错误消息
        """
        logger.info(
            f"Tool 'get_dividend_data' called for {code}, year={year}, year_type={year_type}")
        
        # 基本验证
        if year_type not in ['report', 'operate']:
            logger.warning(f"Invalid year_type requested: {year_type}")
            raise ToolExecutionError(
                "INVALID_ARGUMENT",
                "year_type is not supported.",
                details={"field": "year_type", "allowed": ["report", "operate"]},
            )
        if not year.isdigit() or len(year) != 4:
            logger.warning(f"Invalid year format requested: {year}")
            raise ToolExecutionError(
                "INVALID_ARGUMENT",
                "year must be a four-digit string.",
                details={"field": "year"},
            )

        # 使用通用函数处理数据获取
        return safe_data_fetch(
            "get_dividend_data",
            active_data_source.get_dividend_data,
            code=code,
            year=year,
            year_type=year_type,
        )

    @contract_tool(app)
    def get_adjust_factor_data(
        code: StockCode,
        start_date: ISODate,
        end_date: ISODate,
    ) -> str:
        """
        获取给定股票代码和日期范围的复权因子数据
        使用Baostock的"涨跌幅复权算法"因子。用于计算复权价格

        参数:
            code: Baostock格式的股票代码（例如：'sh.600000', 'sz.000001'）
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'

        返回:
            包含复权因子数据表的Markdown格式字符串，或错误消息
        """
        logger.info(
            f"Tool 'get_adjust_factor_data' called for {code} ({start_date} to {end_date})")
        
        # 使用通用函数处理数据获取
        return safe_data_fetch(
            "get_adjust_factor_data",
            active_data_source.get_adjust_factor_data,
            code=code,
            start_date=start_date,
            end_date=end_date,
        )
