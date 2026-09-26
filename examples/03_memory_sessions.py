"""示例 03：会话记忆。

Memory 支持多会话与滑动窗口裁剪，重启程序后可从 JSON 恢复。
"""

from nanoagent import Agent, Memory

memory = Memory(max_messages=20, persist_path=".nanoagent/example-03.json")
agent = Agent(
    name="记忆助手",
    instructions="你是一个中文助手，记住用户告诉你的信息。",
    memory=memory,
)

if __name__ == "__main__":
    # 同一会话内的两轮对话，第二轮模型能记住第一轮的内容
    print(agent.run("我叫小明，我最喜欢的数字是 7", session_id="demo").content)
    print(agent.run("我叫什么？喜欢的数字是多少？", session_id="demo").content)

    # 不同会话互不干扰
    print(agent.run("我叫什么？", session_id="另一个会话").content)

    path = memory.save()
    print(f"会话已持久化到: {path}")
