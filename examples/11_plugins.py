"""示例 11：插件生态 —— 自定义插件 / 技能 / MCP 的一键装配（对齐 DeepSeek Harness）。

本示例在临时目录里手工搭出一个 .nanoagent 生态再 bootstrap：
    .nanoagent/plugins/time-tools/   nanoagent 原生插件（提供工具）
    .nanoagent/plugins/dsh-like/     模拟 dsh 插件包（skills + mcp 声明）
    .nanoagent/skills/review/        散装 SKILL.md 技能
运行后 agent 同时拥有：插件工具 + SKILL.md 渐进式披露（use_skill）。

真实项目里用命令行安装即可：
    python -m nanoagent.manage install-plugin <路径|git URL>
    python -m nanoagent.manage install-skill  <路径|git URL>
    python -m nanoagent.manage install-mcp    <name> --command CMD
    python -m nanoagent.manage list
"""

import asyncio
import tempfile
from pathlib import Path

from nanoagent import bootstrap_agent, tool


def build_demo_workspace(root: Path) -> Path:
    """搭一个演示用的 .nanoagent 生态。"""
    base = root / ".nanoagent"

    # 1) nanoagent 原生插件：plugin.py 提供 register(ctx)，ctx 是 mini-Cordis 上下文
    plugin = base / "plugins" / "time-tools"
    plugin.mkdir(parents=True)
    (plugin / "plugin.json").write_text('{"name": "time-tools", "version": "0.1.0"}', "utf-8")
    (plugin / "plugin.py").write_text(
        '''
from nanoagent import tool

@tool
def now() -> str:
    """返回当前时间戳（演示用固定值）。"""
    return "2026-09-25 12:00:00"

def register(ctx):
    ctx.tools.register(now)          # 注册工具
    ctx.provide("tz", "Asia/Shanghai")  # 提供服务（对齐 dsh 的 ctx.provide）
''',
        "utf-8",
    )

    # 2) 模拟 dsh 插件包：package.json 带 dsh 声明 + SKILL.md + mcpServers
    dsh = base / "plugins" / "dsh-like"
    (dsh / "skills" / "wiki").mkdir(parents=True)
    (dsh / "package.json").write_text(
        '{"name": "@demo/dsh-like", "dsh": {"bundle": "dsh-tools"}, '
        '"keywords": ["dsh-plugin"]}',
        "utf-8",
    )
    (dsh / "skills" / "wiki" / "SKILL.md").write_text(
        "---\nname: wiki\ndescription: 查询内部百科的规范流程\n---\n"
        "# wiki 使用步骤\n1. 先确定词条名\n2. 用 search 工具检索\n3. 汇总引用来源",
        "utf-8",
    )
    (dsh / ".mcp.json").write_text(
        '{"mcpServers": {}}', "utf-8"  # 真实 dsh 插件在这里声明要连接的 MCP 服务器
    )

    # 3) 散装技能
    review = base / "skills" / "review"
    review.mkdir(parents=True)
    (review / "SKILL.md").write_text(
        "---\nname: review\ndescription: 代码审查五步法\n---\n"
        "# 代码审查步骤\n1. 逻辑正确性\n2. 命名与可读性\n3. 结构与边界\n4. 并发安全\n5. 测试覆盖",
        "utf-8",
    )
    return base


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config_dir = build_demo_workspace(Path(tmp))

        agent, report = await bootstrap_agent(config_dir, llm=None)
        # 真实项目不带 llm 参数，会按 .env / 环境变量构造 LLM()；
        # 本示例只演示装配结果，不真正调用模型。
        print("=== 装配报告 ===")
        print(report.summary())

        # 直接验证技能的渐进式披露（不经过模型）
        print("\n=== 技能索引（注入 system prompt 的第一层） ===")
        print(agent.skills.list_summary())
        print("\n=== use_skill('review') 加载的正文（第二层） ===")
        print(agent.skills.load("review").body)

        # 清理 MCP 连接（本示例没有真实 MCP 服务器，保持习惯即可）
        await report.aclose()


if __name__ == "__main__":
    asyncio.run(main())
