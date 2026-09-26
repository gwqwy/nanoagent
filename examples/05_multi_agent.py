"""示例 05：多 agent 编排。

Team 模式：leader 统筹，成员作为工具被自主委托（OpenAI handoff 思想）；
Pipeline 模式：固定顺序流水线（类 CrewAI 的分工思想）。
"""

from nanoagent import Agent, Pipeline, Team

researcher = Agent(
    name="研究员",
    instructions="你负责收集和整理信息，输出条理清晰的事实要点，不做决策。",
)
writer = Agent(
    name="撰稿人",
    instructions="你负责把要点写成一段 100 字以内的中文短文。",
)

# 方式一：Pipeline —— 研究员先整理，撰稿人再成文
pipeline = Pipeline([researcher, writer])

# 方式二：Team —— 组长自主决定把任务委托给谁
leader = Agent(
    name="组长",
    instructions=(
        "你是组长，手下有研究员和撰稿人。收到任务后先委托研究员整理要点，"
        "再把要点委托给撰稿人成文，最后把成文返回给用户。"
    ),
)
team = Team(leader, [researcher, writer])

if __name__ == "__main__":
    task = "写一段关于「城市养猫需要准备什么」的短文。"
    print("=== Pipeline 输出 ===")
    print(pipeline.run(task))
    print("\n=== Team 输出 ===")
    print(team.run(task))
