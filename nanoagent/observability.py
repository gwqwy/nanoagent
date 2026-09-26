"""可观测性：trace JSONL 的 HTML 可视化报告与 OpenTelemetry 导出。

trace 文件由 Tracer 产生（每行一个 JSON 事件，含 run_id / event / ts）。

    from nanoagent import trace_report
    trace_report("traces/trace-xxx.jsonl", "report.html")   # 生成单文件 HTML 报告

    from nanoagent import export_traces_to_otel
    export_traces_to_otel("traces/trace-xxx.jsonl")          # 需安装 opentelemetry-sdk
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_trace(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def trace_summary(path: str | Path) -> Dict[str, Any]:
    """把 trace 文件汇总成结构化统计：按 run 分组的调用数、token、工具。"""
    events = load_trace(path)
    runs: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "llm_calls": 0, "tool_calls": 0, "prompt_tokens": 0,
        "completion_tokens": 0, "tools": [],
    })
    for event in events:
        run = runs[event.get("run_id", "unknown")]
        kind = event.get("event")
        if kind == "llm_call":
            run["llm_calls"] += 1
            usage = event.get("usage") or {}
            run["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
            run["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        elif kind == "tool_call":
            run["tool_calls"] += 1
            run["tools"].append(event.get("tool", "?"))
    return {"runs": dict(runs), "total_events": len(events)}


def trace_report(path: str | Path, out_path: str | Path) -> Path:
    """把 trace JSONL 渲染成单文件 HTML 报告（零依赖，浏览器直接打开）。"""
    summary = trace_summary(path)
    events = load_trace(path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for run_id, info in summary["runs"].items():
        run_rows.append(
            f"<tr><td>{html.escape(str(run_id))}</td>"
            f"<td>{info['llm_calls']}</td><td>{info['tool_calls']}</td>"
            f"<td>{info['prompt_tokens']}</td><td>{info['completion_tokens']}</td>"
            f"<td>{html.escape(', '.join(info['tools']) or '—')}</td></tr>"
        )

    event_rows = []
    for event in events[:500]:  # 明细最多展示 500 条，防止报告过大
        payload = {k: v for k, v in event.items() if k not in ("run_id", "event", "ts")}
        event_rows.append(
            f"<tr><td>{html.escape(str(event.get('run_id', '')))}</td>"
            f"<td>{html.escape(str(event.get('event', '')))}</td>"
            f"<td>{html.escape(str(event.get('ts', '')))}</td>"
            f"<td><code>{html.escape(json.dumps(payload, ensure_ascii=False))[:300]}</code></td></tr>"
        )

    page = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>nanoagent trace 报告</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2rem; color: #222; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
th, td {{ border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }}
th {{ background: #f5f5f5; }}
code {{ font-size: .8rem; }}
</style></head><body>
<h1>nanoagent trace 报告</h1>
<p>来源: {html.escape(str(path))} · 事件总数: {summary['total_events']} · run 数: {len(summary['runs'])}</p>
<h2>按 run 汇总</h2>
<table><tr><th>run_id</th><th>LLM 调用</th><th>工具调用</th><th>prompt tokens</th><th>completion tokens</th><th>工具</th></tr>
{''.join(run_rows)}</table>
<h2>事件明细（前 500 条）</h2>
<table><tr><th>run_id</th><th>事件</th><th>时间</th><th>内容</th></tr>
{''.join(event_rows)}</table>
</body></html>"""
    out.write_text(page, encoding="utf-8")
    return out


def export_traces_to_otel(path: str | Path, service_name: str = "nanoagent") -> int:
    """把 trace 事件导出为 OpenTelemetry spans（每次 run 一个父 span，调用为其子 span）。

    需要安装 opentelemetry-sdk（pip install nanoagent[otel]）；
    返回导出的 span 数。输出到进程已配置的全局 TracerProvider（如 OTLP exporter）。
    """
    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError as exc:
        raise ImportError(
            "OTel 导出需要先安装: pip install nanoagent[otel] （或 opentelemetry-sdk）"
        ) from exc

    tracer = trace.get_tracer(service_name)
    events = load_trace(path)
    runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        runs[event.get("run_id", "unknown")].append(event)

    exported = 0
    for run_id, run_events in runs.items():
        with tracer.start_as_current_span(f"agent.run", kind=SpanKind.INTERNAL) as parent:
            parent.set_attribute("nanoagent.run_id", str(run_id))
            for event in run_events:
                kind = event.get("event", "event")
                with tracer.start_as_current_span(f"nanoagent.{kind}") as child:
                    for key, value in event.items():
                        if key in ("run_id", "event"):
                            continue
                        child.set_attribute(f"nanoagent.{key}", str(value)[:200])
                    exported += 1
            parent.set_status(Status(StatusCode.OK))
    return exported
