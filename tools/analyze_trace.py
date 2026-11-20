"""分析 run_task_trace.py 存下的日志，统计每个节点被识别/点击多少次。"""
from __future__ import annotations

import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    path = sys.argv[1]
    text = open(path, encoding="utf-8", errors="replace").read()

    reco = re.findall(r"reco hit \[result\.name=([^\]]+)\] \[result\.box=\[([^\]]*)\]\]", text)
    actions = re.findall(r'"action":"(\w+)"', text)
    succeeded = text.count("Node.Action.Succeeded")
    failed = text.count("Node.Action.Failed")
    reco_fail = text.count("Node.Recognition.Failed")

    print(f"file: {path}")
    print(f"  recognition hits : {len(reco)}")
    print(f"  action succeeded : {succeeded}")
    print(f"  action failed    : {failed}")
    print(f"  recognition fails: {reco_fail}")
    print()

    print("  hits by node:")
    for name, cnt in Counter(n for n, _ in reco).most_common():
        print(f"    {cnt:3d}  {name}")

    if reco:
        print("\n  hit sequence:")
        for name, box in reco:
            print(f"    {name:20s} box=[{box}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
