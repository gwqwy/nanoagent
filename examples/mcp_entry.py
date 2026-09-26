"""MCP 接入入口：让编辑器（Claude Code / Cursor / VS Code / dsh 等）调用 nanoagent。

把这个文件的完整路径填进编辑器的 MCP 配置即可，例如 Claude Code：
    claude mcp add nanoagent -- python E:\\文件\\编程文件\\agent\\nanoagent\\examples\\mcp_entry.py

需要先在项目根目录的 .env 里配好模型服务。
"""

from nanoagent import Agent, serve_mcp

agent = Agent(
    name="nanoagent",
    instructions="你是 nanoagent，一个简洁、诚实的中文 AI 助手。",
)

if __name__ == "__main__":
    serve_mcp(agent)  # stdio 传输，阻塞运行
