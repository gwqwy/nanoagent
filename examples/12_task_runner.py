"""示例 12：TaskRunner —— 给一个大目标，agent 自主拆解、逐项连续完成。

连续完成任务的三层：
    1. Agent.run() 一次内多步工具调用（单步连续）
    2. Workflow 固定 SOP 流程（计划→开发↔审查→汇总）
    3. TaskRunner 模型自己拆任务清单并逐项推进（本示例）——适合开放式目标

本示例用 MockLLM 演示完整机制，不调真实模型；真实使用只需换 Agent 的 llm。
"""

from nanoagent import Agent, TaskRunner
from nanoagent.llm import LLMResponse


class ScriptedLLM:
    """脚本化演示模型：规划 → 逐项执行 → 汇总。"""

    model = "demo"

    def __init__(self):
        self.step = 0

    def chat(self, messages, tools=None):
        self.step += 1
        if self.step == 1:  # 规划：只输出 JSON 数组
            return LLMResponse(content=(
                '[{"title": "盘点现有测试", "detail": "列出测试文件与覆盖点"},'
                ' {"title": "补全缺失测试", "detail": "针对未覆盖模块"},'
                ' {"title": "全量回归", "detail": "unittest 全绿"}]'
            ))
        if self.step <= 3:  # 逐项执行
            title = "盘点现有测试" if self.step == 2 else (
                "补全缺失测试" if self.step == 3 else "全量回归")
            return LLMResponse(content=f"已完成「{title}」，产物见报告。")
        return LLMResponse(content="汇总：3/3 项完成，测试全绿，可以发布。")


def main():
    agent = Agent(name="执行者", llm=ScriptedLLM())
    # on_progress：每项任务开始/结束时收到快照，可用来刷 UI 或记日志
    runner = TaskRunner(
        agent,
        checkpoint="task_demo.json",
        on_progress=lambda item: print(f"  [{item.status}] {item.id}. {item.title}"),
    )

    result = runner.run("为 nanoagent 项目补全测试并保证全部通过")

    print("\n=== 任务清单 ===")
    for line in result.summary_lines():
        print(line)
    print("\n=== 最终汇总 ===")
    print(result.summary)


if __name__ == "__main__":
    main()
