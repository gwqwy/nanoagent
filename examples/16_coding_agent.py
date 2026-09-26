"""示例 16：编程 agent —— CodingWorkspace 工具集 + TaskRunner 连续完成开发任务。

这就是"把 nanoagent 变成编程 agent"的完整配方：
    - CodingWorkspace：read/write/edit/list/search/run_command 八件套，
      路径监狱限制在工作区内，run_command 默认需确认（本示例 auto_approve）
    - CODING_INSTRUCTIONS：编程 agent 的标准行为准则
    - TaskRunner：模型自主把开发任务拆成清单逐项推进

任务：在工作区里实现一个 string_utils 模块（带测试），并保证测试通过。
"""

import sys

from nanoagent import Agent, CODING_INSTRUCTIONS, CodingWorkspace, TaskRunner
from nanoagent.config import settings


def main():
    cfg = settings()
    if not cfg.get("api_key") or cfg["api_key"].startswith("sk-xxxx"):
        raise SystemExit("请先在 .env 配置 NANOAGENT_API_KEY（编程 agent 需要真实模型）")

    workspace = CodingWorkspace("./examples/coding_workspace", auto_approve=True)
    agent = Agent(name="程序员", llm=None, tools=workspace.as_tools(),
                  instructions=CODING_INSTRUCTIONS)

    runner = TaskRunner(
        agent,
        max_tasks=6,
        on_progress=lambda item: print(f"  [{item.status}] {item.id}. {item.title}"),
    )
    result = runner.run(
        "在当前工作区实现 string_utils.py：提供 slugify(s)（转小写、非字母数字转连字符、"
        "去首尾连字符）与 truncate(s, n)（超长截断加省略号）；"
        "并写 test_string_utils.py 用 unittest 覆盖边界情况；"
        f"最后用 run_command 执行 `\"{sys.executable}\" -m unittest test_string_utils -v` 保证全绿"
    )

    print("\n=== 任务清单 ===")
    for line in result.summary_lines():
        print(line)

    print("\n=== 工作区产物 ===")
    print(workspace.list_files("**/*"))


if __name__ == "__main__":
    main()
