# Unified Financial MCP Agent Workflow

项目现在只有一套生产编排：LangGraph 先采集并缓存公共证券数据，再并行运行
四个专业 ReAct Agent。所有 MCP 调用统一经过权限、证券范围、日期、数据质量、
重试和追踪网关；最终报告必须先通过非 LLM 质量门和独立风险审查。

公共数据层现在使用多源分工：Baostock 提供证券身份、交易日历、财务与基础
历史行情；mootdx 与腾讯行情用于独立价格/K线核验；东方财富提供估值、换手和
资金流特色信号；事件层查询巨潮并与上交所或深交所公告交叉取数，再以财联社
新闻为主、仅在数量不足时由新浪财经补齐。

```text
                                                   +-> Fundamental ReAct Agent --+
Preflight -> fixed data plan -> shared cache ------+-> Technical ReAct Agent ----+-> deterministic quality gate
                                                   +-> Value ReAct Agent --------+              |
                                                   +-> Event ReAct Agent --------+              v
                                                                                   Risk Review -> Decision -> Report

Each ReAct tool call:
agent allowlist -> MCP executor -> deterministic validator -> bounded retry -> trace
```

## 核心保证

- `WorkflowState` 是 LangGraph 和工具执行基础设施共用的唯一 Pydantic 状态。
- 每个专业 Agent 都产出结构化 `AnalysisResult`，失败路径同样遵守该契约。
- 专业结果只有同时满足“实质内容、至少一条有效证据、无占位文本、最低数据完整度”才会成功。
- 四个必需专业结果未全部通过时，工作流在确定性质量门停止，不调用风险或决策模型，也不写错误报告。
- 风险审查 Agent 专门检查证据缺口、矛盾、未来数据、证券混淆、推断和下行情景。
- 决策 Agent 只能消费质量门通过的数据，并受风险审查给出的评级、目标价和概率权限约束。
- 普通调用者不能再通过无参数 `get_mcp_tools()` 获取全部工具。
- ReAct Agent 只接收其代码级白名单允许的工具。
- 技术/估值 Agent 可使用 `get_mootdx_bars`、`get_tencent_quote` 和
  `get_eastmoney_signals`；事件 Agent 只能使用 `get_official_announcements`
  与 `get_financial_news`，旧的通用网页新闻爬虫不再进入生产白名单。
- 事件质量门同时要求官方公告与财经新闻两类已验证证据，且合计不少于 5 条记录。
- 聚合来源的部分失败通过 `source_chain` / `source_failures` 写入上游元数据；
  全部来源均无可用记录时工具才返回结构化失败。
- 每个工具结果带 `run_id`、`trace_id`、`call_id`、请求哈希和原始数据哈希。
- 工具异常、空数据、未来日期、证券代码错配和 OHLC 异常使用确定性规则校验。
- 工具调用显式记录 `target_symbol`、`scope` 和 `allowed_symbols`。只有估值 Agent
  能在 `peer` 范围查询明确列出的同行；其他 Agent 和未列出的证券会在外部调用前被拒绝。
- 可恢复工具错误按 `max_retries_per_step` 有界重试。
- ReAct AI/Tool 消息轨迹、工具参数、校验结果及重试事件均可审计。
- `max_iterations` 作为每个专业 ReAct 子图的递归上限。
- 运行前会生成完整且确定性的 `collection_plan`：必需项包含最新行情、K线、最新/
  同比财务、公告和新闻；重要项包含分红、历史估值、杜邦与复权口径；可选项包含
  资金流、备用行情源和更早年度数据。专业 Agent 对目标证券只能按计划参数读取共享
  缓存，计划外调用或预取失败后的重复调用会被网关拒绝。
- 工具预算按 `required=60%`、`cross_validation=20%`、`supplemental=20%`
  分类保留（整数预算优先保证必需项不少于 60%），缓存命中不计入外部调用预算。
- Agent 一旦产生 `BUDGET_BLOCKED`，其余工具调用立即被禁用。同一供应商连续两次
  超时、代理或网络连接失败后打开运行级熔断；腾讯与 Baostock 的日期和涨跌幅一致
  时，集中计划直接跳过 mootdx 第三行情源。
- 临时止血默认值为：基本面 14、技术 6、估值 14、事件 6 次外部调用；这些上限
  只约束集中缓存之外的调用，不替代上述分类保留与熔断规则。

## 延迟优化

- 公共证券信息、交易日、行情、财务、分红、公告和新闻按确定性计划集中采集；
  独立供应商并发执行，同一供应商串行通过熔断门；
  默认并发度为 4，可通过状态中的 `collection_concurrency` 调整。
- 已通过契约和质量校验的工具结果写入 `.cache/financial_mcp`。缓存按数据类别采用
  不同 TTL，并支持 stale-while-revalidate；历史行情会复用已缓存区间，仅补取缺失日期。
- 日线数据只采集一次，MA、MACD、RSI、成交量比率和 20 日支撑阻力在本地确定性计算，
  技术 Agent 直接消费 `technical_snapshot`。
- 每个专业 Agent 都有独立的 ReAct 迭代上限、工具调用上限和整段执行超时；超时结果以
  `AGENT_TIMEOUT` 和 `data_incomplete=true` 进入质量门，不生成未经验证的报告。
- 风险审查先由代码检查未来日期、证券范围、证据缺口和指标冲突，LLM 仅分析语义冲突、
  因果跳跃和下行情景。
- 最终决策模型只生成执行摘要、综合评估和投资建议三个结构化字段，输出上限为
  2200 tokens；专业章节、风险复核和数据质量附录由代码模板拼装。

主要运行参数可通过 `WorkflowState.data` 覆盖：

```python
{
    "agent_timeout_seconds": {"technical_analyst": 120, "decision_maker": 60},
    "agent_tool_call_limits": {"technical_agent": 5},
    "collection_tool_call_budget": 27,
    "tool_budget_shares": {
        "required": 0.60,
        "cross_validation": 0.20,
        "supplemental": 0.20,
    },
    "provider_circuit_threshold": 2,
    "agent_iteration_limits": {"technical_agent": 10},
    "collection_concurrency": 4,
    "persistent_cache_enabled": True,
    "stale_while_revalidate": True,
}
```

## 环境与依赖

建议使用 **64 位 CPython 3.11** 和独立的 Conda 环境。当前锁定的依赖组合
以 Python 3.11 为推荐运行版本，尚未对 Python 3.14 做完整兼容性验证。
不要把项目依赖安装到 Conda `base` 环境。

本工作区的两个运行组件必须保持为同级目录：

```text
Finance/
├─ requirements.txt
├─ requirements-mootdx.txt
├─ Financial-MCP-Agent/          # Agent 主程序
└─ a-share-mcp-is-just-i-need/   # A 股 MCP server
```

在 `Finance` 工作区根目录执行：

```powershell
conda create -n finance-agent python=3.11 -y
conda activate finance-agent
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
python -m pip check
python -c "import langgraph, mcp, baostock, pandas, requests, bs4"
```

`mootdx` 只用于通达信 K 线备用校验。需要该数据源时，再执行：

```powershell
python -m pip install --no-deps -r .\requirements-mootdx.txt
python -c "from mootdx.quotes import Quotes; print('mootdx Quotes: OK')"
```

这里的 `--no-deps` 不能省略。`mootdx==0.11.7` 声明的旧版 `httpx`
和 `tenacity` 上限与当前 MCP/LangChain 运行时冲突；
`requirements-mootdx.txt` 已单独列出项目所使用的 `Quotes` 路径依赖。
由于不接受 mootdx 的过时约束，安装该可选依赖后再执行
`pip check` 会看到它针对 `httpx` 和 `tenacity` 的两条预期提示；
以上 `Quotes` 导入检查用于验证项目实际使用的路径。
不需要 mootdx 时可跳过这一步，其他行情源仍可正常运行；
该工具会明确返回 `PROVIDER_NOT_CONFIGURED`。

复制环境变量模板，再填写所使用的 OpenAI 兼容服务：

```powershell
Copy-Item .\Financial-MCP-Agent\.env.example .\Financial-MCP-Agent\.env
```

```dotenv
OPENAI_COMPATIBLE_API_KEY=your_openai_compatible_api_key
OPENAI_COMPATIBLE_BASE_URL=your_base_url
OPENAI_COMPATIBLE_MODEL=your_model_name
USE_LOCAL_MODEL=api
```

不要提交真实的 API Key。Agent 和 MCP server 使用同一个
`sys.executable`，因此安装依赖、运行主程序和执行测试时必须保持
同一 Conda 环境。运行前可用以下命令确认当前解释器：

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

安装完成后进入主程序目录：

```powershell
Set-Location .\Financial-MCP-Agent
```

## 统一入口

所有入口在启动分析 Agent 前都会调用 `resolve_stock_listing` 执行 A 股上市状态
预检。只有公司名称与代码匹配、候选唯一且状态为 `listed` 时才会继续。未找到、
多义、退市、数据源不可用或境外市场请求都会提前停止，避免对不存在或错误的
股票生成报告。当前预检只覆盖 A 股；“A 股未找到”不会被表述成“全球未上市”。

生成完整综合报告：

```powershell
python -m src.main --command "分析贵州茅台600519的基本面和技术走势"
```

持续交互模式（每次提问生成独立执行日志，并复用已加载的 MCP 工具）：

```powershell
python -m src.main
```

完成一次分析后可以直接继续输入下一条问题；输入 `退出`、`exit` 或 `quit`
结束会话。带 `--command` 时仍只执行一次，适合脚本调用。

兼容入口 `agent_loop_cli` 不再启动第二套 Planner/Executor 编排器，而是调用
同一个 LangGraph 工作流。默认启用 data-only 模式，只返回通过质量门的结构化专业分析和证据摘要：

```powershell
python -m src.agent_loop_cli `
  --command "分析贵州茅台600519的基本面" `
  --company "贵州茅台" `
  --data-only
```

显示完整分析内容和工具调用明细：

```powershell
python -m src.agent_loop_cli `
  --command "分析贵州茅台600519的基本面" `
  --data-only `
  --show-data
```

通过兼容入口生成综合报告：

```powershell
python -m src.agent_loop_cli `
  --command "全面分析贵州茅台600519" `
  --no-data-only
```

为估值 Agent 明确授权同行（可重复指定，未授权的同行仍会被拒绝）：

```powershell
python -m src.agent_loop_cli `
  --command "全面分析比亚迪002594并与同行比较" `
  --symbol sz.002594 `
  --allowed-symbol sh.601633 `
  --allowed-symbol sz.000625 `
  --no-data-only
```

审计文件默认位于：

```text
logs/workflow/<run_id>/trace.jsonl
logs/workflow/<run_id>/final_state.json
```

原 `src.agent_loop` 包不再包含独立 Planner/Loop/Replanner 编排器，只保留
统一工具网关实际复用的结果契约、执行器、验证器、追踪器和兼容状态别名。

## 测试

测试使用 Fake MCP Tools，不连接真实数据源：

```powershell
python -m unittest discover -s tests -v
```

MCP 数据源适配器测试位于相邻的 `a-share-mcp-is-just-i-need/tests`。mootdx
0.11.7 的安装顺序和兼容性注意事项见“环境与依赖”。
