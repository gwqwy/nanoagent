"""示例 15：Anthropic 协议适配 —— 同一套 Agent 直连 Anthropic 兼容服务。

AnthropicLLM 与 OpenAI 版 LLM 平级：chat / 工具调用 / 流式接口完全一致，
Agent 侧零改动。协议差异（system 顶层化、tool_use/tool_result block、
input_schema、SSE 流）全部由适配器内部消化。

本示例连 DeepSeek 的 Anthropic 兼容端点（也可换 https://api.anthropic.com
接 Claude，填对应 api_key 即可）：
    base_url = https://api.deepseek.com/anthropic
"""

from nanoagent import Agent, AnthropicLLM, AsyncLLM, tool
from nanoagent.config import settings


@tool
def get_weather(city: str) -> str:
    """查询城市天气。

    Args:
        city: 城市中文名
    """
    return f"{city} 晴 25℃"


def main():
    cfg = settings()
    if not cfg.get("api_key") or cfg["api_key"].startswith("sk-xxxx"):
        raise SystemExit("请先在 .env 配置 NANOAGENT_API_KEY")

    # DeepSeek 的 Anthropic 兼容端点；用 Claude 时换成官方地址与 key
    llm = AnthropicLLM(
        base_url="https://api.deepseek.com/anthropic",
        model=cfg["model"],        # deepseek-flash 在 /anthropic 端点同样可用
        api_key=cfg["api_key"],
        max_tokens=2048,
    )

    # 1) 普通对话（system 由适配器提到请求顶层）
    agent = Agent(name="助手", llm=llm, tools=[get_weather])
    print(f"对话: {agent.run('只回答两个字：你好').content}")

    # 2) 工具调用（tool_use / tool_result block 自动双向转换）
    result = agent.run("北京天气怎么样？")
    print(f"工具调用: {result.content}（{result.iterations} 轮）")

    # 3) 流式（SSE 事件流解析）
    chunks = [ev["text"] for ev in agent.run_stream("用一句话介绍 agent") if ev["type"] == "delta"]
    print(f"流式: {''.join(chunks)[:60]}...")

    print("\n同一套 Agent 代码，OpenAI 协议（LLM）与 Anthropic 协议（AnthropicLLM）自由切换。")


if __name__ == "__main__":
    main()
