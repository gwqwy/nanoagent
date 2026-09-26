"""示例 10：市面对齐特性——async 工作流 + 结构化输出 + checkpoint。

- 全链路异步：Workflow.arun → Agent.arun → AsyncLLM.achat，
  工具协程直接 await、普通函数进线程池，可混用
- 结构化输出：response_model 指定 pydantic 模型，最终回答自动校验为对象
- checkpoint：计划与每轮审查后落盘，进程中断后可用 Workflow.resume 续跑
"""

import asyncio

from pydantic import BaseModel, Field

from nanoagent import Agent, AsyncLLM, Workflow
from nanoagent.memory import Memory


# ---- 结构化输出：汇总员最终要产出符合该模型的对象 -------------------------
class DeliveryReport(BaseModel):
    deliverables: list[str] = Field(description="交付物清单")
    risks: list[str] = Field(description="风险与注意事项")
    conclusion: str = Field(description="一句话结论")


planner = Agent(
    name="计划员",
    instructions="把任务拆成开发计划：目标、步骤、文件清单。禁止写代码。",
    llm=AsyncLLM(),
)
coder = Agent(
    name="程序员",
    instructions="按计划写代码，输出完整文件内容；按审查意见逐条修正。",
    llm=AsyncLLM(),
)
reviewer = Agent(
    name="审查员",
    instructions=(
        "对照计划审查代码。最后一行输出 PASS 或 FAIL 与问题清单。"
    ),
    llm=AsyncLLM(),
)
summarizer = Agent(
    name="汇总员",
    instructions="把计划、代码、审查结论整理成简洁的交付说明（中文）。",
    memory=Memory(),
    llm=AsyncLLM(),
)

workflow = Workflow(
    planner, coder, reviewer, summarizer,
    max_rounds=3,
    checkpoint_path=".nanoagent/example-10-checkpoint.json",
)


async def main() -> None:
    result = await workflow.arun("写一个 Python 模块 text.py，提供 slugify(s) 与 truncate(s, n)")

    print(f"=== 工作流（共 {result.rounds} 轮，{'通过' if result.passed else '未通过'}）===")
    print(result.review)

    # 结构化输出：再起一个 agent 把审查结论转成强类型报告
    reporter = Agent(
        name="报告员",
        instructions="按用户要求输出结构化结果。",
        response_model=DeliveryReport,
        llm=AsyncLLM(),
    )
    report = (await reporter.arun(f"把以下工作流结论整理成报告：\n{result.summary}")).output
    print("\n=== 结构化交付报告 ===")
    print("交付物:", report.deliverables)
    print("风险:", report.risks)
    print("结论:", report.conclusion)
    print("\ncheckpoint 已写入 .nanoagent/example-10-checkpoint.json，"
          "中断后可用 Workflow.resume() 续跑")


if __name__ == "__main__":
    asyncio.run(main())
