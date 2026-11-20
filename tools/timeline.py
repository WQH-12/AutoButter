"""从带 ANSI 颜色的 stdout 日志里抽出时间线，并算出每个阶段耗时。"""
from __future__ import annotations

import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")


def classify(line: str) -> str | None:
    if "reco hit" in line:
        m = re.search(r"result\.name=([^\]]+)\]", line)
        return f"RECO HIT   {m.group(1) if m else ''}"
    if "Node.Recognition.Starting" in line:
        return "reco start"
    if "Node.Recognition.Failed" in line:
        return "reco FAIL"
    if "Node.Action.Starting" in line:
        return "action start"
    if "Node.Action.Succeeded" in line:
        return "action ok"
    if "Tasker.Task.Succeeded" in line:
        return "TASK OK"
    if "Tasker.Task.Failed" in line:
        return "TASK FAIL"
    return None


def main() -> int:
    path = sys.argv[1]
    show_all = "--all" in sys.argv
    events = []
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = ANSI.sub("", raw)
        m = TS.search(line)
        if not m:
            continue
        kind = classify(line)
        if kind:
            events.append((m.group(1), kind))

    if not events:
        print("no events parsed")
        return 1

    t0 = events[0][0]
    prev = None
    for ts, kind in events:
        if not show_all and kind in ("reco start",):
            continue
        delta = ""
        if prev:
            from datetime import datetime
            fmt = "%Y-%m-%d %H:%M:%S.%f"
            d = (datetime.strptime(ts, fmt) - datetime.strptime(prev, fmt)).total_seconds()
            delta = f"  (+{d:.2f}s)"
        print(f"{ts[11:]}  {kind:24s}{delta}")
        prev = ts

    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    total = (datetime.strptime(events[-1][0], fmt)
             - datetime.strptime(events[0][0], fmt)).total_seconds()
    print(f"\ntotal span: {total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
