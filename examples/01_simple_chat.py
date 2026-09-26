"""示例 01：最简单的对话 agent。

运行前：把 .env.example 复制为 .env 并填入你的 OpenAI 兼容服务配置。
"""

from nanoagent import Agent

agent = Agent(
    name="助手",
    instructions="你是一个简洁的中文助手，回答控制在三句话以内。",
)

if __name__ == "__main__":
    result = agent.run("用一句话介绍什么是 AI Agent")
    print(result.content)
    print(f"（共 {result.iterations} 轮，tokens: {result.usage}）")
