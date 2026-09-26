"""示例 04：RAG 知识库。

把本地文档入库（embedding），再把检索能力注册成工具，
模型回答问题时会自动查知识库。需要服务的 /embeddings 接口可用。
"""

from nanoagent import Agent, KnowledgeBase

kb = KnowledgeBase(persist_path=".nanoagent/example-04-kb.json")

# 模拟两份文档入库（首次运行后持久化，再次运行不会重复入库）
if len(kb.store) == 0:
    kb.add_text(
        "nanoagent 是一个从零实现的轻量级 Python Agent 框架，"
        "核心是 agent loop：调用 LLM，若返回工具调用则执行并回填结果，循环直到给出最终回答。",
        metadata={"source": "intro.md"},
    )
    kb.add_text(
        "nanoagent 的工具用 @tool 装饰器定义，JSON Schema 从类型注解和 docstring 自动生成，"
        "支持多会话记忆、RAG 知识库、多 agent 编排（Team/Pipeline）和 JSONL trace。",
        metadata={"source": "features.md"},
    )

agent = Agent(
    name="文档助手",
    instructions="你是文档问答助手，优先根据知识库检索结果回答，并说明出处。",
    tools=[kb.as_tool()],
)

if __name__ == "__main__":
    result = agent.run("nanoagent 的 agent loop 是怎么工作的？")
    print(result.content)
