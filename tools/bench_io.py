"""量一下实机截图/点击的往返耗时，用来判断弹窗循环的瓶颈在哪。"""
from __future__ import annotations

import statistics
import sys
import time

from maa.controller import AdbController
from maa.define import MaaAdbScreencapMethodEnum
from adb_util import adb_path, pick_serial


def main() -> int:
    serial = sys.argv[1] if len(sys.argv) > 1 else pick_serial()
    method = int(MaaAdbScreencapMethodEnum.All)
    ctrl = AdbController(adb_path(), serial, screencap_methods=method)
    ctrl.set_screenshot_target_short_side(720)
    job = ctrl.post_connection().wait()
    if not job.succeeded:
        print("[FATAL] connect failed")
        return 2

    caps = []
    for _ in range(8):
        t = time.perf_counter()
        ctrl.post_screencap().wait().get()
        caps.append((time.perf_counter() - t) * 1000)

    clicks = []
    for _ in range(4):
        t = time.perf_counter()
        ctrl.post_click(360, 760).wait()
        clicks.append((time.perf_counter() - t) * 1000)

    def show(name, xs):
        print(f"{name:12s} n={len(xs)} min={min(xs):7.1f}ms "
              f"median={statistics.median(xs):7.1f}ms max={max(xs):7.1f}ms")

    show("screencap", caps)
    show("click", clicks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
