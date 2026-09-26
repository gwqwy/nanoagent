"""插件/技能/MCP 命令行管理（对齐 dsh plugin 子命令的用法）。

用法：
    python -m nanoagent.manage install-plugin <本地路径|git URL|注册表名> [--registry URL]
    python -m nanoagent.manage install-skill  <本地路径|git URL|注册表名> [--registry URL]
    python -m nanoagent.manage install-mcp    <name> --command CMD [--args "a b"] [--env K=V]
    python -m nanoagent.manage remove-plugin  <name>
    python -m nanoagent.manage remove-skill   <name>
    python -m nanoagent.manage remove-mcp     <name>
    python -m nanoagent.manage list

本地来源直接复制，http(s)/git 来源用 git clone，注册表名经 --registry 指向的
JSON 索引解析成 URL 再安装。索引格式：
    {"plugins": {"名字": {"url": "...", "description": "..."}}, "skills": {...}}
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional

from .mcp import MCPManager
from .plugins import PluginManager

DEFAULT_CONFIG = ".nanoagent"


def _is_url(src: str) -> bool:
    return src.startswith(("http://", "https://", "git@", "ssh://")) or src.endswith(".git")


def _lookup_registry(kind: str, name: str, registry_url: str) -> str:
    """在注册表 JSON 索引里查名字，返回其安装 URL。"""
    try:
        with urllib.request.urlopen(registry_url, timeout=30) as resp:
            index = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"注册表不可用: {registry_url}（{type(exc).__name__}: {exc}）") from exc
    entry = (index.get(kind) or {}).get(name)
    if not entry or not entry.get("url"):
        available = ", ".join((index.get(kind) or {})) or "（空）"
        raise SystemExit(f"注册表中没有 {kind} '{name}'。可用: {available}")
    return entry["url"]


def _resolve_source(source: str, kind: str, registry_url: Optional[str]) -> str:
    """注册表名 → URL；路径/URL 原样返回。"""
    if registry_url and not _is_url(source) and not Path(source).exists():
        return _lookup_registry(kind, source, registry_url)
    return source


def _unique_dir(dest_root: Path, name: str) -> Path:
    dest = dest_root / name
    counter = 2
    while dest.exists():
        dest = dest_root / f"{name}-{counter}"
        counter += 1
    return dest


def _fetch(src: str, dest_root: Path, name: Optional[str] = None) -> Path:
    """把来源（本地路径或 git URL）取到 dest_root/<name>，返回目标目录。"""
    final_name = name or (Path(src.rstrip("/")).name or "plugin")
    if final_name.endswith(".git"):
        final_name = final_name[:-4]
    dest = _unique_dir(dest_root, final_name)
    if _is_url(src):
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", src, str(dest)],
                check=True,
            )
        except FileNotFoundError as exc:
            raise SystemExit("git 不可用：请先安装 git，或改用本地路径安装") from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"git clone 失败（退出码 {exc.returncode}）") from exc
    else:
        source = Path(src)
        if not source.exists():
            raise SystemExit(f"来源不存在: {source}")
        if source.is_dir():
            shutil.copytree(source, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest / source.name)
    return dest


# ----------------------------------------------------------------------
def cmd_install_plugin(args: argparse.Namespace) -> int:
    source = _resolve_source(args.source, "plugins", args.registry)
    dest = _fetch(source, Path(args.config) / "plugins")
    manager = PluginManager()
    manager.scan(Path(args.config) / "plugins")
    match = manager.plugins.get(dest.name)
    fmt = match.format if match else "unknown"
    print(f"已安装插件 {dest.name} -> {dest}（格式: {fmt}）")
    return 0


def cmd_install_skill(args: argparse.Namespace) -> int:
    source = _resolve_source(args.source, "skills", args.registry)
    source_path = Path(source)
    if not _is_url(source) and source_path.is_file():
        # 单个 markdown 文件：装成 skills/<名字>/SKILL.md 才能被 SkillRegistry 收录
        dest = _unique_dir(Path(args.config) / "skills", source_path.stem)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest / "SKILL.md")
    else:
        dest = _fetch(source, Path(args.config) / "skills")
    print(f"已安装技能 {dest.name} -> {dest}")
    return 0


def cmd_install_mcp(args: argparse.Namespace) -> int:
    manager = MCPManager(Path(args.config) / "mcp.json")
    env = None
    if args.env:
        env = dict(pair.split("=", 1) for pair in args.env if "=" in pair)
    manager.add(args.name, args.command, args.args.split() if args.args else None, env)
    print(f"已安装 MCP 服务器 '{args.name}' -> {manager.config_path}")
    return 0


def cmd_remove_plugin(args: argparse.Namespace) -> int:
    target = Path(args.config) / "plugins" / args.name
    if not target.is_dir():
        raise SystemExit(f"插件不存在: {target}")
    shutil.rmtree(target)
    print(f"已移除插件 '{args.name}'")
    return 0


def cmd_remove_skill(args: argparse.Namespace) -> int:
    target = Path(args.config) / "skills" / args.name
    if not target.is_dir():
        raise SystemExit(f"技能不存在: {target}")
    shutil.rmtree(target)
    print(f"已移除技能 '{args.name}'")
    return 0


def cmd_remove_mcp(args: argparse.Namespace) -> int:
    manager = MCPManager(Path(args.config) / "mcp.json")
    manager.remove(args.name)
    print(f"已移除 MCP 服务器 '{args.name}'")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.config)
    manager = PluginManager()
    manager.scan(root / "plugins")
    plugins = manager.list()
    if plugins:
        print("== 插件 ==")
        for plugin in plugins:
            line = f"  {plugin['name']}  [{plugin['format']}:{plugin['state']}]"
            if plugin["provided"]:
                provided = ", ".join(
                    f"{kind}={','.join(items)}" for kind, items in plugin["provided"].items() if items
                )
                line += f"  {provided}"
            if plugin["skipped"]:
                line += f"  (skipped: {'; '.join(plugin['skipped'])})"
            if plugin["error"]:
                line += f"  错误: {plugin['error']}"
            print(line)
    else:
        print("== 插件 ==（无）")

    skills = manager.skills
    skills.add_dir(root / "skills")
    print("== 技能 ==")
    summary = skills.list_summary()
    print("  " + (summary.replace("\n", "\n  ") if summary else "（无）"))

    mcp_path = root / "mcp.json"
    if mcp_path.is_file():
        print("== MCP 服务器 ==")
        for name, cfg in MCPManager(mcp_path).list().items():
            args_str = " ".join(cfg.get("args") or [])
            print(f"  {name}  {cfg.get('command')} {args_str}".rstrip())
    else:
        print("== MCP 服务器 ==（未配置）")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanoagent.manage",
        description="nanoagent 插件 / 技能 / MCP 管理工具",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="配置目录（默认 .nanoagent）")
    parser.add_argument("--registry", default=None,
                        help="插件/技能注册表索引 URL（JSON），install 命令支持按名安装")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("install-plugin", help="安装插件（本地路径或 git URL）")
    p.add_argument("source")
    p.set_defaults(func=cmd_install_plugin)

    p = sub.add_parser("install-skill", help="安装技能（本地路径或 git URL）")
    p.add_argument("source")
    p.set_defaults(func=cmd_install_skill)

    p = sub.add_parser("install-mcp", help="安装 MCP 服务器声明")
    p.add_argument("name")
    p.add_argument("--command", required=True)
    p.add_argument("--args", default="", help="空格分隔的启动参数")
    p.add_argument("--env", nargs="*", default=None, help="K=V 形式的环境变量")
    p.set_defaults(func=cmd_install_mcp)

    p = sub.add_parser("remove-plugin", help="移除插件")
    p.add_argument("name")
    p.set_defaults(func=cmd_remove_plugin)

    p = sub.add_parser("remove-skill", help="移除技能")
    p.add_argument("name")
    p.set_defaults(func=cmd_remove_skill)

    p = sub.add_parser("remove-mcp", help="移除 MCP 服务器声明")
    p.add_argument("name")
    p.set_defaults(func=cmd_remove_mcp)

    p = sub.add_parser("list", help="列出插件 / 技能 / MCP")
    p.set_defaults(func=cmd_list)
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
