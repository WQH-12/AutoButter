"""从 maafw.log 抽出一条时间线，看每个节点什么时候识别、什么时候点击。"""
from __future__ import annotations

import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

LOG = r"E:\work\AutoButter\maafw.log"

pat_time = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")


def classify(line: str) -> str | None:
    if "reco hit" in line:
        m = re.search(r"result\.name=([^\]]+)\]", line)
        return f"RECO HIT  {m.group(1) if m else ''}"
    if "Node.Action.Starting" in line:
        return "ACTION START"
    if "Node.Action.Succeeded" in line:
        return "ACTION OK"
    if "Node.Action.Failed" in line:
        return "ACTION FAIL"
    if "Node.Recognition.Failed" in line:
        return "RECO FAIL"
    if "PipelineNode.Succeeded" in line:
        return "NODE OK"
    if "PipelineNode.Failed" in line:
        return "NODE FAIL"
    if "Tasker.Task.Succeeded" in line:
        return "TASK OK"
    if "Tasker.Task.Failed" in line:
        return "TASK FAIL"
    return None


def main() -> int:
    since = sys.argv[1] if len(sys.argv) > 1 else ""
    with open(LOG, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat_time.match(line)
            if not m:
                continue
            ts = m.group(1)
            if since and ts[11:19] < since:
                continue
            kind = classify(line)
            if kind:
                print(f"{ts[11:]}  {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
