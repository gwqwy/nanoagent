"""nanoagent：从零实现的轻量级 Python Agent 框架。

模块分层：
    config    配置加载（.env / 环境变量）
    llm       OpenAI 兼容模型客户端
    tools     @tool 装饰器与工具注册表
    agent     核心 agent loop
    memory    会话记忆与持久化
    tracing   JSONL 事件追踪
    rag       知识库（分块 / 向量检索）
    multi     多 agent 编排（Team / Pipeline / Workflow）
    task      TaskRunner：自主任务分解与逐项推进（可 checkpoint 续跑）
    guardrails 输入/输出安全护栏（规则护栏 + LLM 判别护栏）
    media    多模态消息构造（图片 → OpenAI image_url content parts）
    skills    Agent Skills（SKILL.md 渐进式披露）
    plugins   插件系统（mini-Cordis 内核 + 多格式适配器）
    mcp       MCP 客户端与服务器配置管理
    bootstrap 一键装配插件生态
    manage    插件管理命令行：python -m nanoagent.manage
    cli       命令行聊天入口：python -m nanoagent.cli
    server    HTTP 服务入口：python -m nanoagent.server
"""

from .agent import Agent, AgentResult, OutputValidationError
from .anthropic_llm import AnthropicLLM, AsyncAnthropicLLM
from .bootstrap import BootstrapReport, bootstrap_agent, bootstrap_agent_sync
from .browser import BrowserWorkspace
from .checkpoints import (
    CheckpointBackend,
    InMemoryBackend,
    JsonFileBackend,
    PostgresBackend,
    SQLiteBackend,
)
from .coding import CODING_INSTRUCTIONS, CodingWorkspace, SandboxViolation
from .evals import CaseReport, EvalCase, EvalReport, EvalRunner
from .mcp_server import create_mcp_server, serve_mcp
from .observability import export_traces_to_otel, trace_report, trace_summary
from .guardrails import (
    Guardrail,
    GuardrailResult,
    GuardrailViolation,
    Guardrails,
    keyword_guardrail,
    length_guardrail,
    llm_guardrail,
    make_guardrail,
    pattern_guardrail,
)
from .llm import AsyncLLM, LLM, LLMResponse, ToolCall
from .memory import Memory, SummaryMemory, estimate_tokens
from .media import build_user_content, image_part, make_png
from .mcp import MCPManager, MCPServer
from .multi import Pipeline, Team, Workflow, WorkflowResult, agent_as_tool
from .plugins import EventBus, Plugin, PluginContext, PluginManager, register_adapter
from .rag import KnowledgeBase, VectorStore, split_text
from .skills import Skill, SkillRegistry, parse_frontmatter
from .task import TaskPlanningError, TaskResult, TaskRunner, TodoItem
from .tools import Tool, ToolRegistry, tool
from .tracing import Tracer

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentResult",
    "OutputValidationError",
    "AsyncLLM",
    "LLM",
    "LLMResponse",
    "ToolCall",
    "AnthropicLLM",
    "AsyncAnthropicLLM",
    "MCPServer",
    "MCPManager",
    "Tool",
    "ToolRegistry",
    "tool",
    "Memory",
    "SummaryMemory",
    "estimate_tokens",
    "Tracer",
    "KnowledgeBase",
    "VectorStore",
    "split_text",
    "Pipeline",
    "Team",
    "Workflow",
    "WorkflowResult",
    "agent_as_tool",
    "Skill",
    "SkillRegistry",
    "parse_frontmatter",
    "Plugin",
    "PluginContext",
    "PluginManager",
    "EventBus",
    "register_adapter",
    "CodingWorkspace",
    "CODING_INSTRUCTIONS",
    "SandboxViolation",
    "EvalRunner",
    "EvalReport",
    "EvalCase",
    "CaseReport",
    "CheckpointBackend",
    "JsonFileBackend",
    "SQLiteBackend",
    "InMemoryBackend",
    "PostgresBackend",
    "BrowserWorkspace",
    "create_mcp_server",
    "serve_mcp",
    "trace_report",
    "trace_summary",
    "export_traces_to_otel",
    "BootstrapReport",
    "bootstrap_agent",
    "bootstrap_agent_sync",
    "TaskRunner",
    "TaskResult",
    "TodoItem",
    "TaskPlanningError",
    "Guardrail",
    "GuardrailResult",
    "GuardrailViolation",
    "Guardrails",
    "make_guardrail",
    "keyword_guardrail",
    "length_guardrail",
    "pattern_guardrail",
    "llm_guardrail",
    "image_part",
    "build_user_content",
    "make_png",
    "__version__",
]
