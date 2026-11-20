"""验证在秘境面板上连点 (350,1326) 不会误触任何 UI。

如果这个点在空白处，那"批量连点"优化才成立：弹窗清完后多余的点击
落在空白处不会造成副作用。
"""
import sys
import time

import cv2
import numpy as np

from maa.controller import AdbController

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"


def diff_ratio(a, b):
    if a.shape != b.shape:
        return 1.0
    d = cv2.absdiff(a, b)
    g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return float((g > 25).mean())


def main() -> int:
    ctrl = AdbController("adb", "3a1e6a75")
    ctrl.set_screenshot_target_short_side(720)
    if not ctrl.post_connection().wait().succeeded:
        print("[FATAL] connect failed")
        return 2

    before = ctrl.post_screencap().wait().get()
    cv2.imwrite(f"{ROOT}\\screenshots\\_safepoint_before.png", before)
    print(f"[before] saved, shape={before.shape}")

    for i in range(10):
        ctrl.post_click(350, 1326).wait()
        time.sleep(0.35)
    time.sleep(0.8)

    after = ctrl.post_screencap().wait().get()
    cv2.imwrite(f"{ROOT}\\screenshots\\_safepoint_after.png", after)
    r = diff_ratio(before, after)
    print(f"[after] 10 clicks at (350,1326), changed pixels = {r * 100:.2f}%")
    print("[verdict]", "SAFE (no UI change)" if r < 0.02 else "UI CHANGED - do not burst-click")
    return 0


if __name__ == "__main__":
    sys.exit(main())
