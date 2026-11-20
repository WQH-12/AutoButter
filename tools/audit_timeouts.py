"""审计 pipeline 的 `timeout`：找「会白等 20s」的节点。

背景（HANDOVER §3.44）：Maa 的 `timeout` 是**候选扫描窗口**，
不是动作超时。父节点没写 timeout 时用默认 20000ms；只要 next/on_error
里有「可能识别失败」的候选（非 DirectHit），一次没命中就白等 20s。

用法：

    python tools\\audit_timeouts.py            # 只列有风险的节点
    python tools\\audit_timeouts.py --all      # 列出所有带候选的节点
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "resource" / "pipeline" / "main.json"
DEFAULT_TIMEOUT = 20000
SLOW_MS = 8000


def load() -> dict:
    body = re.sub(r"(?m)^\s*//.*$", "", PIPE.read_text(encoding="utf-8"))
    return json.loads(body)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    show_all = "--all" in sys.argv
    nodes = load()

    risky = []
    for name, node in nodes.items():
        cands = list(node.get("next") or []) + list(node.get("on_error") or [])
        if not cands:
            continue
        may_fail = [c for c in cands
                    if c in nodes and nodes[c].get("recognition") != "DirectHit"]
        if not may_fail:
            continue
        # 候选列表里只要有一个 DirectHit（永远命中、且是纯 Python/零成本），
        # 扫描就一定会很快结束，不会吃满 timeout。只有「全部候选都可能失败」
        # 才会真的白等。
        always_hit = [c for c in cands
                      if c in nodes and nodes[c].get("recognition") == "DirectHit"]
        if always_hit and not show_all:
            continue
        to = node.get("timeout")
        effective = DEFAULT_TIMEOUT if to is None else int(to)
        if show_all or effective >= SLOW_MS:
            risky.append((name, to, effective, may_fail, bool(always_hit)))

    if not risky:
        print("没有慢候选窗口：所有带可能失败候选的节点 timeout 都 < %dms"
              % SLOW_MS)
        return 0

    print("%-16s %-9s %-9s %-6s %s"
          % ("node", "timeout", "effective", "兜底", "可能失败的候选"))
    for name, to, eff, mf, has_hit in sorted(risky, key=lambda r: -r[2]):
        print("%-16s %-9s %-9s %-6s %s"
              % (name, "-" if to is None else to, eff,
                 "有" if has_hit else "无", ",".join(mf)))
    print("\n共 %d 个节点会白等 >= %dms。见 HANDOVER §3.44。" % (len(risky), SLOW_MS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
