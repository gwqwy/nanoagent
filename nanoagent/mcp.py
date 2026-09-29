"""MCP 客户端：接入 Model Context Protocol 工具生态（可选依赖 nanoagent[mcp]）。

用法：
    server = await MCPServer.connect_stdio("python", ["-m", "some_mcp_server"])
    agent = Agent(..., tools=await server.tools())   # MCP 工具像普通 @tool 一样注册
    ...
    await server.disconnect()

内部通过官方 mcp SDK 的 stdio 传输与子进程服务器通信；
MCP 工具的 inputSchema 直接就是 JSON Schema，无需再生成。
注意：MCP 工具是异步工具，只能配合 Agent.arun/arun_stream 使用。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tools import Tool, sanitize_tool_name


def mcp_schema_to_nanoagent(mcp_tool: Any) -> Tool:
    """把 MCP 工具描述适配成 nanoagent Tool（纯函数，便于单测）。

    mcp_tool 需要有 name / description 与入参 schema 属性
    （mcp 2.x 为 input_schema，1.x 为 inputSchema），缺失时按无参工具处理。
    返回的 Tool.func 只是占位，实际调用需经 MCPServer 转发（见 _bind）。
    工具名经 sanitize_tool_name 净化（MCP 常见 `server.tool` 带点号名会被
    OpenAI 兼容服务 400 拒绝）；原始名经 list_tools 传给 _bind 用于转发。
    """
    schema_obj = getattr(mcp_tool, "input_schema", None) or getattr(mcp_tool, "inputSchema", None)
    schema = dict(schema_obj or {})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    raw_name = str(getattr(mcp_tool, "name", "") or "")
    return Tool(
        name=sanitize_tool_name(raw_name),
        description=(getattr(mcp_tool, "description", "") or f"MCP 工具 {raw_name}").strip(),
        parameters=schema,
        func=_unbound_stub,
    )


async def _unbound_stub(**kwargs):
    raise NotImplementedError("MCP 工具必须经 MCPServer 转发调用（用 server.tools() 获取）")


class MCPServer:
    """一个 MCP 服务器会话：支持 stdio / SSE / Streamable HTTP 三种传输。"""

    def __init__(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters  # noqa: F401
            from mcp.client.stdio import stdio_client  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MCP 支持需要先安装官方 SDK: pip install nanoagent[mcp] （或 pip install mcp）"
            ) from exc
        self._session: Optional[Any] = None
        self._transport_cm: Any = None
        self._session_cm: Any = None

    # ------------------------------------------------------------------
    async def _finish_connect(self, transport_cm: Any) -> None:
        """在传输之上建立会话并完成初始化握手（三种传输共用）。

        握手后段失败必须回滚已成功的段（审计 N-03）：stdio 传输的 __aenter__
        已经拉起了子进程，若 ClientSession/initialize 抛异常而不回滚，
        子进程将永远挂着（泄漏），进程退出前无人清理。
        """
        from mcp import ClientSession

        self._transport_cm = transport_cm
        read, write = await transport_cm.__aenter__()
        self._session_cm = None
        try:
            self._session_cm = ClientSession(read, write)
            self._session = await self._session_cm.__aenter__()
            await self._session.initialize()
        except BaseException:
            # 先关会话再关传输；单段失败不影响另一段的清理
            if self._session_cm is not None:
                try:
                    await self._session_cm.__aexit__(None, None, None)
                except BaseException:
                    pass
                self._session_cm = None
                self._session = None
            try:
                await transport_cm.__aexit__(None, None, None)
            except BaseException:
                pass
            self._transport_cm = None
            raise

    @classmethod
    async def connect_stdio(
        cls,
        command: str,
        args: List[str] | None = None,
        env: Dict[str, str] | None = None,
    ) -> "MCPServer":
        """启动子进程 MCP 服务器并完成初始化握手。"""
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        server = cls()
        await server._finish_connect(
            stdio_client(StdioServerParameters(command=command, args=args or [], env=env))
        )
        return server

    @classmethod
    async def connect_sse(
        cls, url: str, headers: Dict[str, str] | None = None
    ) -> "MCPServer":
        """通过 SSE 传输连接远程 MCP 服务器（url 指向 /sse 端点）。"""
        from mcp.client.sse import sse_client

        server = cls()
        await server._finish_connect(sse_client(url, headers=headers))
        return server

    @classmethod
    async def connect_http(
        cls, url: str, headers: Dict[str, str] | None = None
    ) -> "MCPServer":
        """通过 Streamable HTTP 传输连接远程 MCP 服务器（url 指向 /mcp 端点）。"""
        from mcp.client.streamable_http import streamable_http_client

        server = cls()
        http_client = None
        if headers:
            from mcp.shared._httpx_utils import create_mcp_http_client

            http_client = create_mcp_http_client(headers=headers)
        await server._finish_connect(streamable_http_client(url, http_client=http_client))
        return server

    async def disconnect(self) -> None:
        """关闭会话与传输（与连接的资源获取顺序相反）。

        逐层容错（审计 N-03）：一层的异常被收集，**两层都会被执行**——
        否则 session 关闭失败会让 transport（stdio 子进程）永远无人关闭。
        全部执行完后重抛第一个异常，调用方仍能感知失败。
        """
        errors: List[BaseException] = []
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except BaseException as exc:  # noqa: BLE001 —— 收集后继续关下一层
                errors.append(exc)
            finally:
                self._session_cm = None
                self._session = None
        if self._transport_cm is not None:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                self._transport_cm = None
        if errors:
            raise errors[0]

    # ------------------------------------------------------------------
    async def list_tools(self) -> List[Tool]:
        """拉取服务器工具清单，适配成已绑定转发的 nanoagent Tool 列表。"""
        self._ensure_connected()
        result = await self._session.list_tools()
        return [
            self._bind(mcp_schema_to_nanoagent(t), raw_name=str(getattr(t, "name", "") or ""))
            for t in result.tools
        ]

    async def tools(self) -> List[Tool]:
        """list_tools 的别名，语义上更贴近 Agent(tools=...) 的用法。"""
        return await self.list_tools()

    async def call(self, name: str, arguments: dict | None = None) -> str:
        """调用服务器上的一个工具，返回文本结果。"""
        self._ensure_connected()
        result = await self._session.call_tool(name, arguments or {})
        parts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
        # mcp 2.x 字段为 is_error，1.x 为 isError，二者兼容
        is_error = getattr(result, "is_error", None)
        if is_error is None:
            is_error = getattr(result, "isError", False)
        if is_error:
            return f"MCP 工具 '{name}' 执行出错: {''.join(parts) or '未知错误'}"
        return "".join(parts) or json.dumps({"ok": True}, ensure_ascii=False)

    # ------------------------------------------------------------------
    def _bind(self, tool: Tool, raw_name: str | None = None) -> Tool:
        """给适配后的 Tool 绑上"转发到本服务器"的调用函数。

        raw_name 是服务器侧的原始工具名：对外暴露的名字经过 sanitize
        （OpenAI 兼容服务要求 ^[a-zA-Z0-9_-]+$），转发调用必须用原始名。
        """
        server = self
        server_side_name = raw_name or tool.name

        async def runner(**kwargs):
            return await server.call(server_side_name, kwargs)

        runner.__name__ = tool.name
        runner.__doc__ = tool.description
        return Tool(
            name=tool.name, description=tool.description, parameters=tool.parameters, func=runner
        )

    def _ensure_connected(self) -> None:
        if self._session is None:
            raise RuntimeError("MCP 会话未连接，请先 await MCPServer.connect_stdio(...)")


class MCPManager:
    """MCP 服务器配置管理（安装/卸载/一键连接），对齐 dsh plugin add 的 MCP 形态。

    配置文件用业界通用的 mcpServers 格式（Claude Desktop / Cursor 等一致）：
        {"mcpServers": {"echo": {"command": "python", "args": ["server.py"], "env": {}}}}

    用法：
        manager = MCPManager(".nanoagent/mcp.json")
        manager.add("echo", command="python", args=["mcp_echo_server.py"])  # = 安装
        tools = await manager.connect_all()          # 拉起全部服务器，得到 Tool 列表
        agent = Agent(..., tools=tools)
        await manager.disconnect_all()               # 收尾
    """

    def __init__(self, config_path: str | Path = ".nanoagent/mcp.json") -> None:
        self.config_path = Path(config_path)
        self.config: Dict[str, Dict[str, Any]] = {}  # name -> {command, args, env}
        self._servers: Dict[str, MCPServer] = {}
        self._load()

    # -- 配置持久化 -------------------------------------------------------
    def _load(self) -> None:
        if not self.config_path.is_file():
            return
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # 损坏时明确报错而不是当空配置继续——否则后续 save 会把用户全部声明覆盖掉
            raise ValueError(f"MCP 配置文件无法解析（{self.config_path}）: {exc}") from exc
        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            raise ValueError(f"{self.config_path} 缺少 mcpServers 字段")
        self.config = servers

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        # 临时文件 + os.replace 原子替换：写一半崩溃不会留下半截 JSON
        tmp = self.config_path.with_name(self.config_path.name + ".tmp")
        tmp.write_text(
            json.dumps({"mcpServers": self.config}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.config_path)

    # -- 安装 / 卸载 -------------------------------------------------------
    def add(self, name: str, command: str | None = None, args: Optional[List[str]] = None,
            env: Optional[Dict[str, str]] = None, url: str | None = None,
            type: str | None = None, headers: Optional[Dict[str, str]] = None) -> None:
        """安装一个 MCP 服务器声明并持久化（stdio 用 command，远程用 url + type）。"""
        if not name:
            raise ValueError("add 需要 name")
        if url:
            cfg: Dict[str, Any] = {"url": url, "type": type or "http"}
            if headers:
                cfg["headers"] = dict(headers)
        elif command:
            cfg = {"command": command, "args": list(args or [])}
            if env:
                cfg["env"] = dict(env)
        else:
            raise ValueError("add 需要 command 或 url")
        self.config[name] = cfg
        self._save()

    def remove(self, name: str) -> None:
        """卸载一个服务器声明（不负责断开已建立的连接）。"""
        if name not in self.config:
            raise KeyError(f"未安装的 MCP 服务器 '{name}'，已安装: {', '.join(self.config) or '无'}")
        del self.config[name]
        self._save()

    def list(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.config)

    # -- 连接 -------------------------------------------------------------
    async def connect_all(self, stop_on_error: bool = False) -> List[Tool]:
        """连接全部已声明服务器，返回合并的工具列表（顺序 = 声明顺序）。

        配置支持两种形态：
            {"command": "...", "args": [...]}                stdio 子进程
            {"url": "...", "type": "sse"|"http", "headers": {...}}  远程服务器
        默认单个服务器连接失败不阻断其余；stop_on_error=True 时首个失败即抛。
        """
        tools: List[Tool] = []
        for name, cfg in self.config.items():
            try:
                if cfg.get("url"):
                    transport = (cfg.get("type") or "http").lower()
                    if transport == "sse":
                        server = await MCPServer.connect_sse(cfg["url"], cfg.get("headers"))
                    else:
                        server = await MCPServer.connect_http(cfg["url"], cfg.get("headers"))
                else:
                    server = await MCPServer.connect_stdio(
                        cfg["command"], cfg.get("args") or [], cfg.get("env")
                    )
            except Exception:
                if stop_on_error:
                    raise
                continue
            self._servers[name] = server
            tools.extend(await server.tools())
        return tools

    async def disconnect_all(self) -> None:
        for server in self._servers.values():
            await server.disconnect()
        self._servers.clear()
