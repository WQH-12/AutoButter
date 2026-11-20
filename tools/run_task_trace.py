"""跑 pipeline 并把 stdout 存下来，用于事后分析每个节点的时序。"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry")
    ap.add_argument("--dump", default="")
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    log = args.log or f"screenshots/_trace_{args.entry}.log"
    cmd = [sys.executable, str(ROOT / "tools" / "run_task.py"), args.entry]
    if args.dump:
        cmd += ["--dump", args.dump]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, cwd=str(ROOT), check=False)
    elapsed = time.perf_counter() - t0

    out = proc.stdout.decode("utf-8", errors="replace")
    Path(ROOT / log).write_text(out, encoding="utf-8")
    print(f"elapsed={elapsed:.1f}s  exit={proc.returncode}  log={log}")
    for line in out.splitlines():
        if "[task]" in line or "[dump]" in line:
            print("  " + line[:120])
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
