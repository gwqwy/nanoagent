"""把 nanoagent Agent 暴露为 MCP 服务器（stdio 传输），双向互通的另一半。

暴露的工具（对 MCP 客户端如 Claude Code / dsh 可见）：
    chat(message)            与 agent 对话（含工具调用循环），返回最终回答
    list_agent_tools()       列出 agent 当前可用的内部工具

用法（脚本）：
    from nanoagent import Agent, tool, serve_mcp

    @tool
    def get_weather(city: str) -> str: ...

    agent = Agent(tools=[get_weather])
    serve_mcp(agent)          # 阻塞运行，stdio 传输；配合 MCP 客户端使用

或在 mcp.json 里这样接入其它宿主：
    {"mcpServers": {"nanoagent": {"command": "python", "args": ["my_mcp_entry.py"]}}}
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def create_mcp_server(agent: Any, name: str = "nanoagent", session_prefix: str = "mcp") -> Any:
    """把 Agent 包装成 MCP 服务器实例（mcp SDK 的 MCPServer）。"""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise ImportError(
            "MCP 服务端需要先安装官方 SDK: pip install nanoagent[mcp] （或 pip install mcp）"
        ) from exc

    server = MCPServer(name)
    counter = {"n": 0}

    @server.tool()
    def chat(message: str) -> str:
        """与 nanoagent agent 对话：执行完整 agent loop（含工具调用），返回最终回答。"""
        counter["n"] += 1
        result = agent.run(message, session_id=f"{session_prefix}-{counter['n']}")
        return result.content

    @server.tool()
    def list_agent_tools() -> str:
        """列出这个 agent 内部可用的工具清单。"""
        return ", ".join(agent.tools.names()) or "（无）"

    return server


def serve_mcp(agent: Any, name: str = "nanoagent") -> None:
    """以 stdio 传输阻塞运行 MCP 服务器（供 MCP 客户端子进程拉起）。"""
    create_mcp_server(agent, name).run()
