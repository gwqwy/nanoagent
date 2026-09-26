"""示例 07：并行工具执行 + 摘要记忆。

- parallel_tools=True：同一次响应里的多个工具调用在线程池中并发执行
- SummaryMemory：历史超窗时，旧消息先被 LLM 压缩成摘要，不再直接丢弃
"""

from nanoagent import Agent, SummaryMemory, tool


@tool
def slow_search(keyword: str) -> str:
    """搜索资料（演示用，带 0.5 秒延迟模拟真实网络请求）。

    Args:
        keyword: 搜索关键词
    """
    import time

    time.sleep(0.5)
    return f"'{keyword}' 的搜索结果：要点 A、要点 B"


@tool
def slow_calculator(expression: str) -> str:
    """计算表达式（演示用，同样带延迟）。

    Args:
        expression: 数学表达式
    """
    import time

    time.sleep(0.5)
    return str(eval(expression))  # noqa: S307 —— 示例代码，仅演示


# 摘要记忆：超过 8 条时把最旧的消息压缩成摘要，保留最近 2 条
memory = SummaryMemory(max_messages=8, keep_recent=2, persist_path=".nanoagent/example-07.json")
agent = Agent(
    name="研究员",
    instructions="你是研究助手。可以同时搜索多个关键词、做多组计算；回答用中文。",
    tools=[slow_search, slow_calculator],
    memory=memory,
    parallel_tools=True,   # 默认即开启，这里显式写出便于阅读
    max_workers=4,
)

if __name__ == "__main__":
    # 模型会在一次响应里返回多个工具调用，它们将并发执行（总耗时约 0.5s 而非 1s+）
    result = agent.run("分别搜索「大模型」和「Agent」两个关键词，然后计算 (1+2)*100 等于多少")
    print(result.content)
    print(f"共 {result.iterations} 轮，工具调用 {len(result.tool_calls)} 次")

    # 多轮对话后（第二轮起摘要生效），history() 开头会出现摘要消息
    agent.run("再算一下 7*6")
    print("\n当前记忆状态:")
    print("  摘要:", memory.summary("default") or "（尚未触发压缩）")
    print("  保留消息数:", len(memory.history("default")) - (1 if memory.summary("default") else 0))
