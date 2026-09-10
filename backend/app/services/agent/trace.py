"""Presentation helpers: tool traces and viewer commands from agent runs."""

import re

_FIND_HEADER = re.compile(r"Series with '(\w+)' orientation")
_FIND_LINE = re.compile(r"Series (\d+): (.+)")
_FIND_SLICE = re.compile(r"Requested slice: (\d+)")


def summarize_trace(messages, max_len: int = 500) -> list[dict]:
    """Flatten AI tool calls + tool results into a UI-friendly trace."""
    trace: list[dict] = []
    for m in messages or []:
        mtype = getattr(m, "type", None)
        if mtype == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls or []:
                trace.append({"tool": tc.get("name"), "args": tc.get("args", {})})
        elif mtype == "tool":
            trace.append({"tool": getattr(m, "name", "tool"), "result": str(m.content)[:max_len]})
    return trace


def extract_viz_commands(messages) -> list[dict]:
    """Derive viewer commands only from actual find_series tool results."""
    commands: list[dict] = []
    for m in messages or []:
        if getattr(m, "type", None) != "tool" or getattr(m, "name", "") != "find_series":
            continue
        text = str(m.content)
        header = _FIND_HEADER.search(text)
        orientation = header.group(1) if header else ""
        requested = _FIND_SLICE.search(text)
        slice_number = int(requested.group(1)) if requested else None
        for num, desc in _FIND_LINE.findall(text):
            commands.append(
                {"tool": "find_series", "orientation": orientation,
                 "series_number": int(num), "series_description": desc.strip(),
                 "slice_number": slice_number}
            )
    return commands


def last_ai_text(messages) -> str:
    """Return the most recent AI message text (the report, or the chat reply)."""
    for m in reversed(messages or []):
        if getattr(m, "type", None) == "ai" and getattr(m, "content", None):
            content = m.content
            return content if isinstance(content, str) else str(content)
    return ""
