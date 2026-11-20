"""测试秘境面板上哪些 y 值连点安全（不会关掉面板、不会触发按钮）。

做法：在 x=360 上逐 y 点 3 次，比对整屏变化。变化大 = 触发了 UI。
每个 y 测完都重新进入秘境，保证起点一致。
"""
import subprocess
import sys
import time

import cv2
import numpy as np

from maa.controller import AdbController

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
SERIAL = "3a1e6a75"


def enter_realm():
    subprocess.run(
        ["python", f"{ROOT}\\tools\\run_task.py", "进入淫乱秘境"],
        capture_output=True, cwd=ROOT, check=False,
    )


def main() -> int:
    ys = [int(v) for v in sys.argv[1:]] or [980, 1000, 1020, 1040, 1060]
    ctrl = AdbController("adb", SERIAL)
    ctrl.set_screenshot_target_short_side(720)
    if not ctrl.post_connection().wait().succeeded:
        print("[FATAL] connect failed")
        return 2

    for y in ys:
        enter_realm()
        time.sleep(0.6)
        before = ctrl.post_screencap().wait().get()
        for _ in range(3):
            ctrl.post_click(360, y).wait()
            time.sleep(0.3)
        time.sleep(0.6)
        after = ctrl.post_screencap().wait().get()
        d = cv2.absdiff(before, after)
        g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        ratio = float((g > 25).mean()) * 100
        verdict = "SAFE" if ratio < 2 else "CHANGED"
        print(f"y={y:4d}  x=360  changed={ratio:5.2f}%  {verdict}", flush=True)
        if verdict == "CHANGED":
            cv2.imwrite(f"{ROOT}\\screenshots\\_unsafe_y{y}.png", after)

    return 0


if __name__ == "__main__":
    sys.exit(main())
