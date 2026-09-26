"""测试用 MCP 服务器：提供 echo / add 两个工具，stdio 传输（mcp 2.x API）。

由 test_async_features.py 以子进程方式拉起，也可手动运行验证。
"""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("nanoagent-test-server")


@mcp.tool()
def echo(text: str) -> str:
    """原样返回输入文本。"""
    return f"echo: {text}"


@mcp.tool()
def add(a: int, b: int) -> int:
    """两数之和。"""
    return a + b


if __name__ == "__main__":
    mcp.run()
