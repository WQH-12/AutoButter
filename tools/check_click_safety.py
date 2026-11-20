"""检查候选点击点在「没有弹窗」时是否安全（不会误触 UI）。

只点 1 次、比对前后整屏差异，不做连点。
用法:
    python tools/check_click_safety.py 360,470 360,1326
"""
import subprocess
import sys
import time

import cv2

from maa.controller import AdbController

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
SERIAL = "3a1e6a75"


def main() -> int:
    specs = sys.argv[1:] or ["360,470"]
    ctrl = AdbController("adb", SERIAL)
    ctrl.set_screenshot_target_short_side(720)
    if not ctrl.post_connection().wait().succeeded:
        print("[FATAL] connect failed")
        return 2

    for spec in specs:
        x, y = (int(v) for v in spec.split(","))
        # 每次都重新进入秘境，保证起点一致
        subprocess.run(
            ["python", f"{ROOT}\\tools\\run_task.py", "进入淫乱秘境"],
            capture_output=True, cwd=ROOT, check=False,
        )
        time.sleep(0.8)

        before = ctrl.post_screencap().wait().get()
        ctrl.post_click(x, y).wait()
        time.sleep(0.9)
        after = ctrl.post_screencap().wait().get()

        d = cv2.cvtColor(cv2.absdiff(before, after), cv2.COLOR_BGR2GRAY)
        ratio = float((d > 25).mean()) * 100
        print(f"click ({x},{y}) once -> changed {ratio:5.2f}%  "
              f"{'SAFE' if ratio < 2 else 'TRIGGERED UI'}", flush=True)
        if ratio >= 2:
            cv2.imwrite(f"{ROOT}\\screenshots\\_unsafe_{x}_{y}.png", after)

    return 0


if __name__ == "__main__":
    sys.exit(main())
