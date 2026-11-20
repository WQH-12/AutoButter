"""MFA 调试小工具：以 720 画布坐标为基准做点击/截图，复用同一控制器。

用法:
    python tools/ui.py shot 31_realm
    python tools/ui.py tap 360 1400 32_after
    python tools/ui.py tap 360 1400            # 只点不截图
    python tools/ui.py swipe 360 1200 360 800 33_swiped
"""
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "screenshots"
import adb_util  # noqa: E402

ADB = adb_util.adb_path()
SERIAL = adb_util.pick_serial(ADB)

from maa.controller import AdbController  # noqa: E402

_ctrl = None


def ctrl() -> AdbController:
    global _ctrl
    if _ctrl is None:
        c = AdbController(ADB, SERIAL)
        c.set_screenshot_target_short_side(720)
        job = c.post_connection().wait()
        if not job.succeeded:
            raise SystemExit("controller connect failed")
        _ctrl = c
    return _ctrl


def shot(name: str) -> Path:
    img = ctrl().post_screencap().wait().get()
    out = SHOTS / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"{out} {img.shape[1]}x{img.shape[0]}")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    a = sys.argv[2:]
    if cmd == "shot":
        shot(a[0])
    elif cmd == "tap":
        x, y = int(a[0]), int(a[1])
        ctrl().post_click(x, y).wait()
        time.sleep(1.2)
        if len(a) > 2:
            shot(a[2])
        else:
            print(f"tapped ({x},{y})")
    elif cmd == "swipe":
        x1, y1, x2, y2 = map(int, a[:4])
        ctrl().post_swipe(x1, y1, x2, y2, 400).wait()
        time.sleep(1.0)
        if len(a) > 4:
            shot(a[4])
        else:
            print(f"swiped ({x1},{y1})->({x2},{y2})")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
