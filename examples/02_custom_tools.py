"""示例 02：自定义工具。

用 @tool 装饰器把普通函数变成 agent 可调用的工具，
模型会自主决定何时调用、传什么参数。
"""

from nanoagent import Agent, tool


@tool
def calculator(expression: str) -> str:
    """计算一个四则运算表达式的值。

    Args:
        expression: 数学表达式，例如 (3 + 5) * 12
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "只支持四则运算表达式"
    return str(eval(expression))  # noqa: S307 —— 已白名单校验字符


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气（演示用假数据）。

    Args:
        city: 城市中文名
    """
    fake = {"北京": "晴 25℃", "上海": "多云 28℃", "广州": "雷阵雨 31℃"}
    return fake.get(city, f"{city}：暂无数据")


agent = Agent(
    name="生活助手",
    instructions="你是生活助手。需要算数或查天气时调用对应工具，回答用中文。",
    tools=[calculator, get_weather],
)

if __name__ == "__main__":
    result = agent.run("北京和上海今天哪个热？热多少度？")
    print("回答:", result.content)
    for call in result.tool_calls:
        print(f"  🔧 {call['name']}({call['arguments']}) -> {call['result']}")
