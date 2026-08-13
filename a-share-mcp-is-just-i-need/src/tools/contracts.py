"""Shared contracts for every MCP tool exposed by this server."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Annotated, Any, Callable, Literal, Optional, TypeAlias
from typing_extensions import NotRequired, TypedDict

from pydantic import ConfigDict, Field, ValidationError, create_model, validate_call

from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase


StockCode: TypeAlias = Annotated[
    str,
    Field(
        pattern=r"^(?:sh|sz)\.\d{6}$",
        description="Baostock A-share code such as sh.600519 or sz.300750",
        examples=["sh.600519"],
    ),
]
ISODate: TypeAlias = Annotated[
    str,
    Field(
        pattern=r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$",
        description="Calendar date in YYYY-MM-DD format",
        examples=["2026-07-19"],
    ),
]
OptionalISODate: TypeAlias = Optional[ISODate]
Year: TypeAlias = Annotated[
    str,
    Field(pattern=r"^\d{4}$", description="Four-digit year", examples=["2025"]),
]
OptionalYear: TypeAlias = Optional[Year]
YearMonth: TypeAlias = Annotated[
    str,
    Field(
        pattern=r"^\d{4}-(?:0[1-9]|1[0-2])$",
        description="Calendar month in YYYY-MM format",
        examples=["2026-07"],
    ),
]
OptionalYearMonth: TypeAlias = Optional[YearMonth]
Quarter: TypeAlias = Annotated[int, Field(ge=1, le=4)]
TopK: TypeAlias = Annotated[int, Field(ge=1, le=50)]
SearchQuery: TypeAlias = Annotated[str, Field(min_length=1, max_length=200)]
CompanyName: TypeAlias = Annotated[
    str,
    Field(
        min_length=2,
        max_length=100,
        description="Chinese listed-company name or common security name",
        examples=["贵州茅台"],
    ),
]
Frequency: TypeAlias = Literal["d", "w", "m", "5", "15", "30", "60"]
AdjustFlag: TypeAlias = Literal["1", "2", "3"]
DividendYearType: TypeAlias = Literal["report", "operate"]
ReserveRatioYearType: TypeAlias = Literal["0", "1"]
AnalysisType: TypeAlias = Literal["fundamental", "technical", "comprehensive"]
MarketPeriod: TypeAlias = Literal["recent", "quarter", "half_year", "year"]


class ToolData(TypedDict):
    content: Any


class ToolErrorPayload(TypedDict):
    code: str
    message: str
    retryable: bool
    details: dict[str, Any]


class ToolMeta(TypedDict):
    tool: str
    provider: NotRequired[str]
    source_chain: NotRequired[list[str]]
    source_failures: NotRequired[list[dict[str, Any]]]


class ToolEnvelope(TypedDict):
    ok: bool
    data: Optional[ToolData]
    error: Optional[ToolErrorPayload]
    meta: ToolMeta


class ToolExecutionError(Exception):
    """A stable, machine-readable tool failure raised by implementations."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}


def tool_success(
    tool_name: str,
    value: Any,
    *,
    provider: str = "mcp",
    source_chain: Optional[list[str]] = None,
    source_failures: Optional[list[dict[str, Any]]] = None,
) -> ToolEnvelope:
    """Build the only success shape returned by MCP tools."""
    meta: ToolMeta = {"tool": tool_name, "provider": provider}
    if source_chain:
        meta["source_chain"] = source_chain
    if source_failures:
        meta["source_failures"] = source_failures
    return {
        "ok": True,
        "data": {"content": value},
        "error": None,
        "meta": meta,
    }


def tool_failure(
    tool_name: str,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: Optional[dict[str, Any]] = None,
) -> ToolEnvelope:
    """Build the only error shape returned by MCP tools."""
    return {
        "ok": False,
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": details or {},
        },
        "meta": {"tool": tool_name},
    }


def contract_tool(app):
    """Register a tool with strict Pydantic validation and a JSON envelope.

    The original flat function signature is retained, so generated tool schemas stay
    easy for models to call. Validation failures and implementation failures are
    normalized instead of leaking provider-specific strings.
    """

    def decorator(func: Callable[..., Any]):
        validated = validate_call(
            func,
            config=ConfigDict(
                arbitrary_types_allowed=True,
                str_strip_whitespace=True,
            ),
        )

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> ToolEnvelope:
            try:
                value = validated(*args, **kwargs)
                if inspect.isawaitable(value):
                    raise ToolExecutionError(
                        "ASYNC_TOOL_NOT_SUPPORTED",
                        "This synchronous MCP contract wrapper received an async result.",
                    )
                if isinstance(value, dict) and {"ok", "data", "error", "meta"} <= value.keys():
                    return value
                return tool_success(func.__name__, value)
            except ValidationError as error:
                return tool_failure(
                    func.__name__,
                    "VALIDATION_ERROR",
                    "Tool arguments do not satisfy the declared schema.",
                    details={
                        "issues": error.errors(
                            include_url=False,
                            include_context=False,
                            include_input=False,
                        )
                    },
                )
            except ToolExecutionError as error:
                return tool_failure(
                    func.__name__,
                    error.code,
                    error.message,
                    retryable=error.retryable,
                    details=error.details,
                )
            except Exception as error:
                return tool_failure(
                    func.__name__,
                    "INTERNAL_ERROR",
                    "The tool failed unexpectedly.",
                    details={"exception_type": type(error).__name__},
                )

        original_signature = inspect.signature(func)
        wrapped.__signature__ = original_signature.replace(  # type: ignore[attr-defined]
            return_annotation=ToolEnvelope
        )
        wrapped.__annotations__ = dict(func.__annotations__)
        wrapped.__annotations__["return"] = ToolEnvelope
        registered = app.tool()(wrapped)

        # FastMCP normally validates with its own generated model before calling
        # the function. Keep the strict schema advertised to the LLM, but route
        # runtime validation through ``wrapped`` so every validation failure uses
        # the same JSON envelope as other tool failures.
        tool_manager = getattr(app, "_tool_manager", None)
        registered_tool = (
            tool_manager.get_tool(func.__name__)
            if tool_manager is not None and hasattr(tool_manager, "get_tool")
            else None
        )
        if registered_tool is not None:
            registered_tool.parameters["additionalProperties"] = False

            class EnvelopeArgModelBase(ArgModelBase):
                model_config = ConfigDict(
                    arbitrary_types_allowed=True,
                    extra="allow",
                )

                def model_dump_one_level(self) -> dict[str, Any]:
                    values = super().model_dump_one_level()
                    values.update(self.__pydantic_extra__ or {})
                    return values

            permissive_fields: dict[str, Any] = {}
            for name, parameter in original_signature.parameters.items():
                default = (
                    ...
                    if parameter.default is inspect.Parameter.empty
                    else parameter.default
                )
                permissive_fields[name] = (Any, default)

            registered_tool.fn_metadata.arg_model = create_model(
                f"{func.__name__}EnvelopeArguments",
                __base__=EnvelopeArgModelBase,
                **permissive_fields,
            )

        return registered

    return decorator
