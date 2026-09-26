"""命令行聊天入口。

用法：
    python -m nanoagent.cli                 # 默认从 .env 读配置
    python -m nanoagent.cli --model xx --base-url https://...
    python -m nanoagent.cli --no-trace      # 关闭 trace 记录

会话内命令：
    /image <路径|URL>  附加图片，随下一条消息发送（需模型支持视觉）
    /new      清空当前会话记忆
    /tools    查看已注册工具
    /sessions 查看历史会话
    /save     立即保存会话到磁盘
    /exit     退出（Ctrl+C / Ctrl+Z 亦可）
"""

from __future__ import annotations

import argparse
import sys
import time

from .agent import Agent
from .config import load_dotenv, settings
from .llm import LLM
from .media import image_part
from .memory import Memory
from .tracing import Tracer

SESSIONS_FILE = ".nanoagent/sessions.json"
TRACE_FILE = f"traces/trace-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

WELCOME = """nanoagent v{version} | 模型: {model} | 服务: {base_url}
输入消息开始对话；/image 附加图片；/tools /sessions /new /save /exit 为内置命令"""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoagent 命令行聊天")
    parser.add_argument("--model", default=None, help="模型名，默认取配置")
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容服务地址")
    parser.add_argument("--api-key", default=None, help="API Key，默认取配置")
    parser.add_argument("--session", default="cli", help="会话 id，默认 cli")
    parser.add_argument("--no-trace", action="store_true", help="关闭 trace 记录")
    parser.add_argument("--trace-file", default=TRACE_FILE, help="trace 输出文件")
    return parser.parse_args(argv)


def build_agent(args: argparse.Namespace) -> Agent:
    cfg = settings()
    if not cfg["api_key"]:
        print(
            "未配置 API Key。请先把 .env.example 复制为 .env 并填入你的 base_url / api_key / model。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    tracer = Tracer(enabled=not args.no_trace, path=args.trace_file)
    llm = LLM(model=args.model, base_url=args.base_url, api_key=args.api_key)
    return Agent(
        name="nanoagent",
        instructions="你是 nanoagent，一个简洁、诚实的中文 AI 助手。回答保持准确，不确定时明确说明。",
        llm=llm,
        memory=Memory(persist_path=SESSIONS_FILE),
        tracer=tracer,
    )


def repl(agent: Agent, session_id: str) -> None:
    print(WELCOME.format(version=_version(), model=agent.llm.model, base_url=agent.llm.base_url))
    pending_images: list = []  # /image 附加的图片，随下一条消息发送
    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input in ("/exit", "/quit", "/q"):
            print("再见！")
            break
        if user_input == "/new":
            agent.memory.clear(session_id)
            print("已清空当前会话记忆。")
            continue
        if user_input == "/tools":
            print("已注册工具:", ", ".join(agent.tools.names()) or "（无）")
            continue
        if user_input == "/sessions":
            print("历史会话:", ", ".join(agent.memory.sessions()) or "（无）")
            continue
        if user_input == "/save":
            path = agent.memory.save()
            print(f"已保存到 {path}" if path else "未启用持久化。")
            continue
        if user_input.startswith("/image"):
            source = user_input[len("/image"):].strip()
            if not source:
                print("用法: /image <本地路径|https://URL>，可连续多次附加，随下一条消息发送。")
                continue
            try:
                pending_images.append(image_part(source))
                print(f"已附加第 {len(pending_images)} 张图片。")
            except Exception as exc:  # noqa: BLE001
                print(f"[出错] {type(exc).__name__}: {exc}")
            continue

        images = pending_images or None
        pending_images = []
        try:
            for event in agent.run_stream(user_input, session_id=session_id, images=images):
                if event["type"] == "delta":
                    print(event["text"], end="", flush=True)
                elif event["type"] == "tool_call":
                    print(f"\n🔧 调用工具 {event['name']}({event['arguments']})")
                elif event["type"] == "done":
                    print()
        except KeyboardInterrupt:
            print("\n（已中断本次回答）")
        except Exception as exc:  # noqa: BLE001 —— REPL 不因单次错误退出
            print(f"\n[出错] {type(exc).__name__}: {exc}")
    agent.memory.save()


def _version() -> str:
    from . import __version__

    return __version__


def main() -> None:
    load_dotenv()
    args = parse_args()
    agent = build_agent(args)
    repl(agent, args.session)


if __name__ == "__main__":
    main()
