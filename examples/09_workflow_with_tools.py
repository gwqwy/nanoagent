"""示例 09：工具版工作流——程序员 agent 真实读写文件。

与示例 08 同一套 Workflow 编排，区别是：
    - 程序员挂上 write_file / read_file / list_files 工具，
      会真的把代码写进本目录下的 workflow_workspace/ 工作区
    - 审查员挂上只读工具（read_file / list_files），自己去工作区读文件审查
    - 所有文件工具都做了路径越界防护，只能操作系统工作区内的文件

运行结束后到 examples/workflow_workspace/ 查看程序员"交付"的代码。
"""

from pathlib import Path

from nanoagent import Agent, Workflow, tool

WORKSPACE = (Path(__file__).resolve().parent / "workflow_workspace").resolve()


def _safe_path(path: str) -> Path:
    """把相对路径解析到工作区内，防止越界读写。"""
    if Path(path).is_absolute():
        raise ValueError(f"只允许相对路径，收到绝对路径: {path}")
    target = (WORKSPACE / path).resolve()
    if target != WORKSPACE and WORKSPACE not in target.parents:
        raise ValueError(f"路径越界：只允许操作工作区 {WORKSPACE} 内的文件")
    return target


@tool
def write_file(path: str, content: str) -> str:
    """把文件写入工作区。

    Args:
        path: 相对工作区的文件路径，例如 fib.py
        content: 完整的文件内容
    """
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {path}（{len(content)} 字符）"


@tool
def read_file(path: str) -> str:
    """读取工作区里的一个文件。

    Args:
        path: 相对工作区的文件路径
    """
    target = _safe_path(path)
    if not target.exists():
        return f"文件不存在: {path}。当前文件: {', '.join(sorted(p.name for p in WORKSPACE.rglob('*') if p.is_file())) or '工作区为空'}"
    return target.read_text(encoding="utf-8")


@tool
def list_files() -> str:
    """列出工作区里的全部文件。"""
    files = sorted(p.relative_to(WORKSPACE).as_posix() for p in WORKSPACE.rglob("*") if p.is_file())
    return "\n".join(files) if files else "工作区为空"


# 工具版：程序员可写，审查员只读 —— 权限按职责分配
coder = Agent(
    name="程序员",
    instructions=(
        "你是程序员。收到开发计划与审查意见后，调用 write_file 工具把代码"
        "真实写入工作区（每个文件调用一次）。写完后简要说明写了哪些文件。"
    ),
    tools=[write_file, read_file, list_files],
)
reviewer = Agent(
    name="审查员",
    instructions=(
        "你是审查员。用 list_files 和 read_file 工具读取工作区里的实际文件，"
        "对照开发计划审查。结论放在最后一行：通过输出 PASS，否则输出 FAIL 加问题清单。"
    ),
    tools=[read_file, list_files],
)
planner = Agent(
    name="计划员",
    instructions="你是计划员。把任务拆成开发计划：目标、步骤、文件清单。禁止写代码。",
)
summarizer = Agent(
    name="汇总员",
    instructions="你是汇总员。把计划、实际产出的文件、审查结论汇总成最终交付说明。",
)

workflow = Workflow(planner, coder, reviewer, summarizer, max_rounds=3)

if __name__ == "__main__":
    WORKSPACE.mkdir(exist_ok=True)
    result = workflow.run(
        "在工作区实现 fib.py：提供 fib(n) 函数（迭代实现，n<0 时抛 ValueError），"
        "并附带 test_fib.py，用 assert 写 3 个测试用例"
    )
    print("=== 审查结论 ===")
    print(result.review)
    print(f"\n=== 交付说明（共 {result.rounds} 轮，{'通过' if result.passed else '未通过'}）===")
    print(result.summary)
    print(f"\n=== 工作区实际文件（{WORKSPACE}）===")
    for f in sorted(p.relative_to(WORKSPACE).as_posix() for p in WORKSPACE.rglob("*") if p.is_file()):
        print("  ", f)
