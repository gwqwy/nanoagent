"""示例 08：多 agent 协作工作流（框架级 Workflow 编排）。

四个角色完成一项开发任务：
    计划员 ──► 程序员 ◄──┐
                │        │ 反馈回路（审查不过自动打回，最多 max_rounds 轮）
                ▼        │
              审查员 ────┘
                │ PASS
                ▼
              汇总员

每个角色只看到自己需要的"产物包"：
    程序员  收到 任务 + 计划 + 审查意见
    审查员  收到 计划 + 代码
    汇总员  收到 全部产物
"""

from nanoagent import Agent, Workflow

planner = Agent(
    name="计划员",
    instructions=(
        "你是计划员。把任务拆解成开发计划：目标、实现步骤、涉及的文件。"
        "只做规划，禁止写代码。回答用中文。"
    ),
)
coder = Agent(
    name="程序员",
    instructions=(
        "你是程序员。严格按照开发计划写代码，输出完整文件内容；"
        "如果收到审查意见，逐条修正后再输出。不要解释多余内容。"
    ),
)
reviewer = Agent(
    name="审查员",
    instructions=(
        "你是审查员。对照开发计划逐项审查代码：逻辑错误、命名、结构、边界条件。"
        "结论必须放在回答的最后一行：通过输出 PASS，不通过输出 FAIL 并附问题清单。"
    ),
)
summarizer = Agent(
    name="汇总员",
    instructions="你是汇总员。把计划、代码、审查结论整理成一份简洁的最终交付说明。",
)

workflow = Workflow(planner, coder, reviewer, summarizer, max_rounds=3)

if __name__ == "__main__":
    result = workflow.run("用 Python 写一个 utils.py，提供 add(a, b) 与 safe_div(a, b)，除零时返回 None")
    print("=== 审查结论 ===")
    print(result.review)
    print(f"\n=== 汇总（共 {result.rounds} 轮，{'通过' if result.passed else '未通过'}）===")
    print(result.summary)
