"""一键装配：扫描插件/技能/MCP 配置，组装成开箱即用的 Agent。

目录约定（对齐 dsh 的 profile 思路）：
    <config_dir>/plugins/<name>/   插件目录（格式自动探测，见 plugins.py）
    <config_dir>/skills/           散装 SKILL.md 技能
    <config_dir>/mcp.json          {"mcpServers": {...}} MCP 服务器声明

用法（MCP 连接是异步的，故 bootstrap 是协程）：
    agent, report = asyncio.run(bootstrap_agent(config_dir=".nanoagent"))
    print(report.summary())
    result = agent.run("用上插件给我的能力做点什么")

单个插件激活失败 / 单个 MCP 服务器连接失败都不阻断整体，
分别记录进 report.failed；dsh 插件无法执行的 TS 代码体记录进 report.skipped。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent import Agent
from .llm import LLM
from .mcp import MCPManager, MCPServer
from .memory import Memory
from .plugins import ACTIVE, PluginManager
from .skills import SkillRegistry
from .tracing import Tracer


@dataclass
class BootstrapReport:
    """bootstrap 的装配报告。"""

    plugins: List[dict] = field(default_factory=list)   # 每个插件的 summary()
    skills: List[str] = field(default_factory=list)     # 已注册技能名
    mcp_servers: List[str] = field(default_factory=list)  # 成功连接的 MCP 服务器
    tools: List[str] = field(default_factory=list)      # agent 最终可用的全部工具名
    failed: List[Dict[str, str]] = field(default_factory=list)  # [{name, error}]
    skipped: List[Dict[str, str]] = field(default_factory=list)  # [{plugin, reason}]
    config_dir: str = ""
    _mcp_connections: List[Any] = field(default_factory=list, repr=False)

    async def aclose(self) -> None:
        """断开 bootstrap 期间建立的全部 MCP 连接（收尾用）。"""
        for server in self._mcp_connections:
            try:
                await server.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._mcp_connections.clear()

    def summary(self) -> str:
        lines = [f"config_dir: {self.config_dir}"]
        for plugin in self.plugins:
            state = f"{plugin['name']}[{plugin['format']}:{plugin['state']}]"
            provided = ", ".join(
                f"{kind}={','.join(items)}"
                for kind, items in plugin["provided"].items() if items
            )
            lines.append(f"plugin {state} {provided}".rstrip())
            for note in plugin["skipped"]:
                lines.append(f"  skipped: {note}")
        if self.skills:
            lines.append(f"skills: {', '.join(self.skills)}")
        if self.mcp_servers:
            lines.append(f"mcp: {', '.join(self.mcp_servers)}")
        lines.append(f"tools: {', '.join(self.tools) or '无'}")
        for item in self.failed:
            lines.append(f"failed: {item['name']} —— {item['error']}")
        return "\n".join(lines)


async def bootstrap_agent(
    config_dir: str | Path = ".nanoagent",
    *,
    llm: Any = None,
    instructions: str = "",
    memory: Optional[Memory] = None,
    tracer: Optional[Tracer] = None,
    **agent_kwargs: Any,
) -> Tuple[Agent, BootstrapReport]:
    """扫描 config_dir 并组装 Agent，返回 (agent, report)。

    llm 缺省时按 .env / 环境变量构造 LLM()；其余 agent_kwargs
    （max_iterations、response_model 等）原样透传给 Agent。
    """
    root = Path(config_dir)
    report = BootstrapReport(config_dir=str(root))

    # 1) 插件：发现 + 激活（失败记录不阻断）
    manager = PluginManager()
    manager.scan(root / "plugins")
    for plugin in manager.activate_all():
        report.plugins.append(plugin.summary())
        if plugin.state != ACTIVE:
            report.failed.append({"name": plugin.name, "error": plugin.error})
        for note in plugin.skipped:
            report.skipped.append({"plugin": plugin.name, "reason": note})

    # 2) 散装技能目录
    skills: SkillRegistry = manager.skills
    skills.add_dir(root / "skills")
    report.skills = skills.names()

    # 3) MCP 服务器：逐个连接，失败的记录后继续
    mcp_config = root / "mcp.json"
    mcp_tools = []
    try:
        servers = MCPManager(mcp_config).list() if mcp_config.is_file() else {}
    except Exception as exc:  # noqa: BLE001 —— 配置文件损坏不应阻断装配
        servers = {}
        report.failed.append({"name": "mcp:config", "error": f"{type(exc).__name__}: {exc}"})
    for name, cfg in servers.items():
        try:
            single = MCPManager(str(mcp_config))
            single.config = {name: cfg}
            mcp_tools.extend(await single.connect_all(stop_on_error=True))
            report._mcp_connections.extend(single._servers.values())
            report.mcp_servers.append(name)
        except Exception as exc:  # noqa: BLE001
            report.failed.append(
                {"name": f"mcp:{name}", "error": f"{type(exc).__name__}: {exc}"}
            )

    # 4) 组装 Agent
    agent = Agent(
        instructions=instructions,
        llm=llm or LLM(),
        tools=manager.collect_tools() + mcp_tools,
        memory=memory,
        tracer=tracer,
        **agent_kwargs,
    )
    if len(skills):
        agent.enable_skills(skills)
    report.tools = agent.tools.names()
    return agent, report


def bootstrap_agent_sync(
    config_dir: str | Path = ".nanoagent", **kwargs: Any
) -> Tuple[Agent, BootstrapReport]:
    """bootstrap_agent 的同步包装：内部 asyncio.run。

    注意：若在已有事件循环内调用请直接用 await bootstrap_agent(...)。
    """
    return asyncio.run(bootstrap_agent(config_dir, **kwargs))
