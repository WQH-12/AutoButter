"""对比裸 adb 的几种截图方式，看能不能绕开 Maa 的 1.2s 开销。"""
from __future__ import annotations

import statistics
import subprocess
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

SERIAL = "3a1e6a75"
W, H = 1080, 2340


def run(cmd: list[str]) -> bytes:
    return subprocess.run(cmd, capture_output=True, check=False).stdout


def bench(name, fn, n=6):
    times = []
    out = None
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t) * 1000)
    size = len(out) if isinstance(out, (bytes, bytearray)) else "?"
    print(f"{name:38s} median={statistics.median(times):7.1f}ms "
          f"min={min(times):7.1f}ms bytes={size}")
    return times


def main() -> int:
    print(f"device {SERIAL}\n")

    bench("adb exec-out screencap -p (PNG)", lambda: run(
        ["adb", "-s", SERIAL, "exec-out", "screencap", "-p"]))

    bench("adb exec-out screencap (raw RGBA)", lambda: run(
        ["adb", "-s", SERIAL, "exec-out", "screencap"]))

    bench("adb shell screencap -p /sdcard/x.png", lambda: run(
        ["adb", "-s", SERIAL, "shell", "screencap", "-p", "/sdcard/x.png"]))

    # 只截一小块（screencap 不支持，用 -d 裁剪不行；改用 raw + 本地裁）
    raw = run(["adb", "-s", SERIAL, "exec-out", "screencap"])
    if len(raw) >= 12:
        w = int.from_bytes(raw[0:4], "little")
        h = int.from_bytes(raw[4:8], "little")
        print(f"\nraw header: {w}x{h} fmt={int.from_bytes(raw[8:12], 'little')} "
              f"payload={len(raw) - 12} bytes")

    print("\n参考：Maa AdbController 单次 1096~1328ms（tools/bench_io.py）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
