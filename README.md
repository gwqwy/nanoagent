# nanoagent

[![CI](https://github.com/gwqwy/nanoagent/actions/workflows/ci.yml/badge.svg)](https://github.com/gwqwy/nanoagent/actions/workflows/ci.yml)

从零实现的轻量级 Python Agent 框架。设计参考 [Agno](https://github.com/agno-agi/agno)（轻量、agent 即库）、
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)（极简循环与 handoff）与
[LangGraph](https://github.com/langchain-ai/langgraph)（可观测性）的思想，但全部代码手写、依赖极少、
每个模块都可以单独读懂——适合学习 agent 原理，也可以直接拿来用。

## 功能一览

| 模块 | 说明 | 对标 |
|---|---|---|
| `llm.py` | OpenAI 兼容客户端（同步 `LLM` + 异步 `AsyncLLM`，非流式/流式），任何 base_url 均可 | OpenAI SDK |
| `anthropic_llm.py` | Anthropic Messages 协议适配器（`AnthropicLLM`/`AsyncAnthropicLLM`）：system 顶层化、tool_use/tool_result 转换、SSE 流解析，Agent 侧零改动 | Anthropic SDK / Claude |
| `tools.py` | `@tool` 装饰器，schema 从类型注解 + docstring 自动生成；支持协程工具 | OpenAI Agents SDK |
| `agent.py` | 核心 agent loop（LLM → 工具 → 回填 → 循环），`run/arun` 双轨；工具并行（线程池/asyncio.gather）；`response_model` 结构化输出 | 所有框架的同构核心 |
| `memory.py` | 多会话记忆、滑动窗口裁剪、JSON 持久化；`SummaryMemory` 超窗自动摘要 | Agno Memory |
| `tracing.py` | 每步 LLM/工具调用记录为 JSONL | LangSmith（极简版） || `rag/` | 分块 → embedding → 向量检索（numpy / faiss 双后端，接口一致可互换）→ 元数据过滤 → LLM 重排序（可选）→ 自动变工具 | LlamaIndex（极简版） |
| `mcp.py` | MCP 客户端：stdio / SSE / Streamable HTTP 三种传输接入任意 MCP 服务器，工具自动适配注册；`MCPManager` 管理服务器安装/连接 | Model Context Protocol |
| `mcp_server.py` | 反向集成：把 nanoagent Agent 暴露为 MCP 服务器（`serve_mcp`），被 Claude Code / dsh 等宿主调用 | Model Context Protocol |
| `observability.py` | trace JSONL → 单文件 HTML 报告 / OpenTelemetry spans 导出 | LangSmith（极简版）/ OTel |
| `multi.py` | Team（agent-as-tool 委托）、Pipeline（顺序流水线）、Workflow（计划→开发↔审查反馈回路→汇总，支持 checkpoint 续跑） | CrewAI / handoff / durable execution |
| `task.py` | TaskRunner：大目标自主拆解成任务清单、逐项连续执行、checkpoint 续跑 | AutoGPT / plan-and-execute |
| `guardrails.py` | 输入/输出安全护栏：规则护栏（关键词/长度/正则）+ LLM 判别护栏，fail-closed，四条执行路径全覆盖 | OpenAI Agents SDK guardrails |
| `skills.py` | Agent Skills：SKILL.md 渐进式披露（索引 → `use_skill` 按需加载全文） | DeepSeek Harness / Claude Code Skills |
| `plugins.py` | 插件系统：mini-Cordis 内核（ctx/effect/disposer）+ 多格式适配器（nanoagent / dsh / mcp-only / agent-skills，可注册扩展） | DeepSeek Harness「一切皆插件」 |
| `bootstrap.py` | 扫描 `.nanoagent/` 生态一键装配成 Agent，返回装配报告 | dsh profile |
| `manage.py` | 插件/技能/MCP 管理命令行（install/remove/list） | `dsh plugin` 子命令 |
| `media.py` | 多模态消息构造：图片源（URL/文件/bytes）→ OpenAI `image_url` content parts | OpenAI Vision / DeepSeek Vision |
| `coding.py` | 编程工具集 `CodingWorkspace`：read/write/edit/list/search/run_command/git 三件套/后台进程/子 agent，路径监狱 + 危险操作确认门 | Claude Code 等编程 agent |
| `evals.py` | 评估体系：用例集 + contains/regex/LLM-as-judge 三种判分，报告可存档做回归对比 | promptfoo（极简版） |
| `checkpoints.py` | 可插拔 checkpoint 后端：JSON 文件 / SQLite（stdlib，跨进程）/ 内存，Workflow 与 TaskRunner 通用 | durable execution |
| `cli.py` | 命令行 REPL 聊天，支持流式输出与 /image 多模态输入 | — |
| `server.py` | FastAPI 异步 HTTP 服务（`/chat`、`/sessions`、`/health`） | — |

## 快速开始

```cmd
:: 1. 安装（二选一）
::    a) 直接从 GitHub 安装
pip install "git+https://github.com/gwqwy/nanoagent.git"
::    b) 本地开发安装（项目根目录已有 .venv 可跳过创建）
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -e .[server]

:: 2. 配置模型服务（支持智谱 GLM / DeepSeek / Kimi / 通义 / Ollama 等任何 OpenAI 兼容服务）
copy .env.example .env
::  编辑 .env 填入 NANOAGENT_BASE_URL / NANOAGENT_API_KEY / NANOAGENT_MODEL

:: 3. 命令行聊天
.venv\Scripts\python -m nanoagent.cli

:: 4. 或运行示例
.venv\Scripts\python examples\02_custom_tools.py
```

## 30 秒上手

```python
from nanoagent import Agent, tool

@tool
def get_weather(city: str) -> str:
    """查询城市天气。

    Args:
        city: 城市中文名
    """
    return f"{city} 晴 25℃"

agent = Agent(name="助手", instructions="你是中文助手", tools=[get_weather])
print(agent.run("北京天气怎么样？").content)
```

`agent.run()` 背后的循环（也是所有 agent 框架的同构核心）：

```
messages = [system] + 记忆 + [user]
循环最多 max_iterations 次：
    响应 = LLM(messages, tools)
    若有 tool_calls：执行工具 → 结果以 role=tool 回填 → 继续
    否则：返回最终回答
```

## 核心用法

**多 agent 协作** —— leader 自主委托成员：

```python
from nanoagent import Team, Agent
team = Team(leader=Agent(name="组长", instructions="..."),
            members=[Agent(name="研究员", instructions="..."),
                     Agent(name="撰稿人", instructions="...")])
team.run("写一段关于城市养猫的短文")
```

**RAG 知识库**（大数据量可切 faiss 后端）：

```python
kb = KnowledgeBase(persist_path="kb.json")                  # 默认 numpy 后端
kb = KnowledgeBase(persist_path="kb.faiss", backend="faiss")  # 需 pip install nanoagent[faiss]
kb.add_file("docs/manual.md")
agent = Agent(name="问答", instructions="...", tools=[kb.as_tool()])
```

**摘要记忆** —— 长对话超窗时自动把旧消息压缩成摘要，Agent 侧零改动：

```python
from nanoagent import SummaryMemory
memory = SummaryMemory(max_messages=8, keep_recent=2, persist_path="chat.json")
agent = Agent(..., memory=memory)
```

**按 token 限窗与自动压缩**（对齐 Claude Code 的 compaction 思路）：

```python
from nanoagent import Memory, SummaryMemory

Memory(max_messages=50, max_tokens=20_000)          # 条数与 token 双维限窗
SummaryMemory(max_tokens=50_000, keep_recent=6)     # token 超限自动触发摘要压缩
```

token 用零依赖估算器（ASCII 约 4 字符/token，CJK 约 1 字符/token，多模态消息
按文本 + 每图上限估算）；`memory.tokens(session_id)` 可随时查看当前用量。

**Anthropic 协议接入**（Claude / DeepSeek /anthropic 端点）：

```python
from nanoagent import Agent, AnthropicLLM

llm = AnthropicLLM(base_url="https://api.deepseek.com/anthropic", model="deepseek-flash")
agent = Agent(llm=llm, tools=[...])   # 对话/工具/流式与 OpenAI 版完全一致
```

**并行工具** —— 同一响应内的多个工具调用并发执行（默认开启）：

```python
agent = Agent(..., tools=[search_a, search_b], parallel_tools=True, max_workers=4)
# 注意：并行模式下工具函数需线程安全；设 parallel_tools=False 可退回串行
```

**多 agent 开发工作流** —— 计划 → 开发 ↔ 审查（不过自动打回）→ 汇总：

```python
from nanoagent import Workflow
workflow = Workflow(计划员, 程序员, 审查员, 汇总员, max_rounds=3)
result = workflow.run("写一个 utils.py")     # 审查员给程序员 agent 也可以挂工具，
print(result.summary, result.passed)         # 让程序员真实写文件（见示例 09）
```

**异步（async-first，与同步 API 双轨并存）**：

```python
from nanoagent import Agent, AsyncLLM
agent = Agent(..., llm=AsyncLLM())                    # 或 Agent(model=...) 同步版
result = await agent.arun("...")                      # 工具并行走 asyncio.gather
async for ev in agent.arun_stream("..."):             # 流式同样支持
    ...
```

**MCP 工具生态**（`pip install nanoagent[mcp]`）：

```python
from nanoagent import MCPServer
server = await MCPServer.connect_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", dir])
agent = Agent(..., tools=await server.tools())        # 像 @tool 一样注册进 Agent
result = await agent.arun("...")                      # MCP 工具是异步工具，需 arun
await server.disconnect()
```

**结构化输出**（对齐 PydanticAI）：

```python
class Report(BaseModel):
    title: str
    risk: str

agent = Agent(..., response_model=Report)   # 解析失败自动回填模型重试一次
result = agent.run("...")
result.output.title                          # 强类型对象
```

**Workflow checkpoint 续跑**：

```python
workflow = Workflow(..., checkpoint_path="wf.json")   # 每阶段自动落盘
result = Workflow.resume("wf.json", 计划员, 程序员, 审查员, 汇总员)  # 中断后续跑
```

**连续完成任务（TaskRunner，plan-and-execute）** —— 与 Workflow 的区别：
Workflow 的阶段是预先编排死的固定 SOP；TaskRunner 的步骤清单由模型运行时自己拆，
适合开放式目标：

```python
from nanoagent import TaskRunner

runner = TaskRunner(agent, checkpoint_path="task.json")
result = runner.run("为项目补全测试并保证全部通过")   # 模型拆解 → 逐项执行 → 汇总
print(result.summary)                # 最终汇总
print(result.summary_lines())        # 每项的状态与结果（✓/✗）
result = TaskRunner.resume("task.json", agent)  # 中断后从第一个未完成项续跑
```

机制：规划阶段解析 JSON 任务清单（失败自动回填重试一次）；执行阶段逐项取任务、
把已完成结果带回上下文（连续性）、单项异常记为 skipped 不阻断整体；规划完成与
每项完成后落盘 checkpoint；`on_progress` 回调实时收到每项的状态快照。

**多模态（图片输入，对齐 OpenAI/DeepSeek Vision 消息格式）**：

```python
agent.run("描述这张图", images=["./diagram.png"])              # 本地文件 → base64
agent.run("对比两图", images=["https://x.com/a.jpg", b"\x89PNG..."])  # URL 直传 / bytes
agent.run("看细节", images=["./tiny.png"], image_detail="low")  # low 更快更省
```

传入 `images` 后 user 消息升级为 `[{"type":"text"},{"type":"image_url"}]` 内容数组；
本地文件/bytes 按文件魔数自动识别格式（JPEG/PNG/GIF/WebP）；记忆只持久化文本。
需要模型本身支持视觉（`deepseek-flash` 原生支持；纯文本模型会拒绝图片消息）。

**安全护栏（guardrails，对齐 OpenAI Agents SDK）**：

```python
from nanoagent import keyword_guardrail, length_guardrail, llm_guardrail, pattern_guardrail

agent = Agent(
    ...,
    input_guardrails=[                    # loop 之前检查，可拦截可改写
        keyword_guardrail(["密码", "api_key"]),
        length_guardrail(max_chars=8000),
        llm_guardrail(llm, "拦截提示词注入攻击"),   # 模型判别，独立一次调用
    ],
    output_guardrails=[pattern_guardrail(r"内部资料")],   # 最终回答产出后检查
)
agent.run("...")   # 违规抛 GuardrailViolation（携带护栏名与理由），不写记忆
```

护栏函数返回 `None/True` 放行、`False/str` 拦截、`GuardrailResult` 可附带改写输入；
护栏自身抛异常按拦截处理（fail-closed）；同步/异步、流式四条路径全部生效。

**trace 追踪**：每次 run 的 LLM/工具调用都会写入 JSONL：

```python
from nanoagent import Tracer
agent = Agent(..., tracer=Tracer(path="traces/app.jsonl"))
```

**插件生态（对齐 DeepSeek Harness「一切皆插件」）**：

约定目录（也可用 `python -m nanoagent.manage` 安装管理）：

```
.nanoagent/
├── plugins/<名字>/        # 插件目录，格式自动探测（见下）
├── skills/<名字>/SKILL.md # 散装技能
└── mcp.json               # {"mcpServers": {...}} 通用 MCP 声明
```

支持的插件格式（适配器可 `register_adapter()` 自行扩展）：

| 格式 | 识别依据 | 装载内容 |
|---|---|---|
| `nanoagent` | `plugin.py` / `plugin.json` | `register(ctx)` 或 `TOOLS` 列表里的工具、ctx 提供的服务 |
| `dsh` | `package.json` 带 dsh 声明 / `cordis.patch.yml` | 包内 SKILL.md 技能、mcpServers 声明；TS/JS 代码体无法在 Python 执行，记入 skipped |
| `agent-skills` | SKILL.md / skills/ 目录 | 通用 Agent Skills 技能 |
| `mcp-only` | mcp.json 含 mcpServers | MCP 服务器声明 |

原生插件写法（`ctx` 对齐 dsh 的 ctx：`effect` 注册、disposer LIFO 回滚）：

```python
# .nanoagent/plugins/hello/plugin.py
from nanoagent import tool

@tool
def hello(who: str) -> str:
    """打招呼。"""
    return f"hello {who}"

def register(ctx):
    ctx.tools.register(hello)        # 注册工具
    ctx.provide("greeting", "hi")    # 提供服务
    ctx.effect(lambda: (lambda: None))  # 可回滚资源
```

一键装配（MCP 连接是异步的）：

```python
from nanoagent import bootstrap_agent
agent, report = await bootstrap_agent(".nanoagent")
print(report.summary())     # 装了哪些插件/技能/MCP、跳过了什么、失败原因
await report.aclose()       # 收尾断开 MCP
```

命令行管理（对照 `dsh plugin`）：

```cmd
python -m nanoagent.manage install-plugin https://github.com/user/dsh-plugin-xxx
python -m nanoagent.manage install-skill  .\my-skill.md
python -m nanoagent.manage install-mcp    filesystem --command npx --args "-y @modelcontextprotocol/server-filesystem ."
python -m nanoagent.manage list
```

SKILL.md 技能走渐进式披露：只有名字+描述注入 system prompt，模型按需调
`use_skill` 工具加载全文，不撑爆上下文（见 `examples/11_plugins.py`）。

**编程 agent（CodingWorkspace）** —— 一套工具把 nanoagent 变成 coding agent：

```python
from nanoagent import Agent, CODING_INSTRUCTIONS, CodingWorkspace, TaskRunner

ws = CodingWorkspace("./workspace", auto_approve=True)   # 或传 confirm 回调人工确认
agent = Agent(name="程序员", tools=ws.as_tools(), instructions=CODING_INSTRUCTIONS)
agent.run("在 utils.py 里实现 slugify 并写好测试")
TaskRunner(agent).run("给项目补全测试并保证全绿")          # 配合 TaskRunner 连续开发
```

工具：`read_file`（带行号/区间）、`write_file`、`edit_file`（精确替换 + diff 回显）、
`file_diff`、`list_files`、`search_code`（正则）、`run_command`、`delete_file`。
安全设计：路径监狱（越界访问拒绝）；`run_command`/`delete_file` 默认拒绝执行
（fail-closed），需 `auto_approve=True` 或 `confirm` 回调放行；输出限幅防爆上下文。

**评估体系（evals）**——验证"任务做得多好"而非"机制对不对"：

```python
from nanoagent import EvalRunner

runner = EvalRunner(judge_llm=LLM())                       # judge 用例需要判别模型
runner.add_case(expect_contains=["25℃"])                   # 确定性关键词
runner.add_case(expect_regex=r"\d+ 分")                    # 正则
runner.add_case(judge_rubric="摘要是否覆盖全部要点")          # LLM-as-judge 打 0-100 分
report = runner.run(agent)
print(report.summary_text()); report.save("evals/report.json")
```

**可插拔 checkpoint 后端**——JSON 文件（默认）/ SQLite（跨进程，stdlib）/ 内存：

```python
from nanoagent import SQLiteBackend, TaskRunner

runner = TaskRunner(agent, checkpoint=SQLiteBackend("runs.db"))
result = TaskRunner.resume(SQLiteBackend("runs.db"), agent)   # 跨进程续跑
```

**限流重试与用量统计**：`LLM/AsyncLLM` 对 429/5xx/连接失败指数退避重试
（`max_retries`/`retry_backoff` 可配），`llm.total_usage` 累计 token 用量；
装有 `tiktoken`（`pip install nanoagent[tiktoken]`）时上下文估算用精确 tokenizer。

**远程 MCP 服务器**：mcp.json 支持 SSE/Streamable HTTP 传输：

```json
{"mcpServers": {"remote": {"url": "https://example.com/mcp", "type": "http"}}}
```

**流式 HTTP 端点**：`POST /chat/stream`（SSE），事件含 delta / tool_call / done。

**HTTP 服务**：

```cmd
.venv\Scripts\python -m nanoagent.server
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" ^
     -d "{\"session_id\": \"u1\", \"message\": \"你好\"}"
```

## 运行测试

测试全部使用 mock 模型，不需要 API key：

```cmd
.venv\Scripts\python -m unittest discover tests
```

## 目录结构

```
nanoagent/
├── nanoagent/          # 框架源码（见上表）
│   └── rag/            # chunking.py / store.py（numpy 后端+工厂）/ faiss_store.py / retriever.py
├── examples/           # 01 简单对话 → 16 编程 agent，渐进式示例
├── tests/              # 294 个单元测试（mock LLM，离线可跑；MCP 集成测试自动跳过）
├── .github/workflows/  # CI：3.10/3.12/3.14 三版本跑全量测试
├── pyproject.toml
└── .env.example        # 模型服务配置模板

## Roadmap

- milvus、pinecone 等云端向量库后端（实现 `add/search` 接口即可接入工厂）
- MCP resources/prompts 支持（tools 已覆盖）
- 更细粒度的多模态（Files API、audio 输入）

## 可选依赖组

```cmd
pip install nanoagent[server]      :: FastAPI HTTP 服务
pip install nanoagent[faiss]       :: faiss 向量后端
pip install nanoagent[mcp]         :: MCP 客户端与服务端
pip install nanoagent[tiktoken]    :: 上下文 token 精确估算
pip install nanoagent[otel]        :: OpenTelemetry 导出
pip install nanoagent[postgres]    :: Postgres checkpoint 后端
pip install nanoagent[browser]     :: 浏览器工具（需 playwright install chromium）
```

## 已知限制

- 向量检索 numpy 后端是暴力余弦，适合几千条以内；更大数据量请 `backend="faiss"`（精确检索，
  几十万条毫秒级）；百万级以上或需要更省内存时可再扩展 IVF 类近似索引。
- 并行工具的同步路径用线程池，受 Python GIL 限制，CPU 密集工具加速有限（IO 密集工具与
  异步协程工具收益明显）。
- RAG 依赖模型服务提供 `/embeddings` 接口（OpenAI 兼容），选择服务时注意确认。
- 摘要记忆每次压缩多一次 LLM 调用（低成本模型即可胜任）。
- numpy 与 faiss 后端的持久化格式互不兼容，切换后端后需重新入库。
- MCP 工具是异步工具，只能在 `arun/arun_stream` 中使用；checkpoint 与各 agent 的
  session 绑定，恢复时需传入与中断前相同的角色配置。
- dsh 插件的 TS/JS 代码体（Cordis 服务）无法在 Python 运行时执行，适配器只消费
  包内的声明层（SKILL.md 技能、mcpServers、元信息），代码体记入报告的 skipped。
- TaskRunner 的清单在规划时一次性生成（执行中不重规划）；单项失败记为 skipped，
  中途终止可用 `resume` 从第一个未完成项续跑。
- token 估算：装有 tiktoken 时精确计数，否则零依赖启发式；Anthropic 协议无
  embeddings 接口，RAG 场景继续用 OpenAI 兼容客户端（`KnowledgeBase(llm=...)` 可单独指定）。

## 许可证

本项目采用 MIT 协议，详见 [LICENSE](LICENSE)。
