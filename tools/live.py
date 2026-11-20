"""持久化 MaaFramework 会话：连接一次，之后每条命令都在同一进程里执行。

用法:
    python tools/live.py            # 交互模式，从 stdin 读命令

命令:
    shot <name>                     截图到 screenshots/<name>.png
    tap <x> <y>                     click（720 画布坐标）
    swipe <x1> <y1> <x2> <y2> [ms]  滑动
    sleep <ms>                      等待
    match <tpl> [x y w h]           模板匹配，打印 score / 中心点
    tapimg <tpl> [x y w h]          匹配成功后点中心
    find <tpl> [x y w h]            返回 JSON（给脚本用）
    res                             打印当前分辨率
    quit                            退出

多行可一次性写入，逐行执行。
"""
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "resource" / "image"
SHOTS = ROOT / "screenshots"
SHOTS.mkdir(exist_ok=True)

from maa.controller import AdbController  # noqa: E402

import adb_util  # noqa: E402

ADB = adb_util.adb_path()
SERIAL = adb_util.pick_serial(ADB)

ctrl = AdbController(ADB, SERIAL)
ctrl.set_screenshot_target_short_side(720)
if not ctrl.post_connection().wait().succeeded:
    print("[FATAL] connect failed", flush=True)
    sys.exit(2)

_cache = {"img": None}


def grab(force=False):
    if force or _cache["img"] is None:
        _cache["img"] = ctrl.post_screencap().wait().get()
    return _cache["img"]


def invalidate():
    _cache["img"] = None


def do_match(tpl_name, roi=None):
    img = grab()
    tpl_path = IMG / tpl_name
    if not tpl_path.exists():
        return {"error": f"template not found: {tpl_path}"}
    tpl = cv2.imread(str(tpl_path), cv2.IMREAD_COLOR)
    if tpl is None:
        return {"error": f"template unreadable: {tpl_path}"}
    ox = oy = 0
    search = img
    if roi:
        x, y, w, h = roi
        search = img[y:y + h, x:x + w]
        ox, oy = x, y
    if search.shape[0] < tpl.shape[0] or search.shape[1] < tpl.shape[1]:
        return {"error": "roi smaller than template"}
    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(res)
    th, tw = tpl.shape[:2]
    return {
        "template": tpl_name,
        "score": round(float(mx), 4),
        "center": [ox + ml[0] + tw // 2, oy + ml[1] + th // 2],
        "box": [ox + ml[0], oy + ml[1], tw, th],
    }


def handle(line: str) -> bool:
    line = line.strip()
    if not line or line.startswith("#"):
        return True
    parts = line.split()
    cmd, a = parts[0].lower(), parts[1:]

    if cmd == "quit":
        return False
    if cmd == "res":
        print(f"resolution={ctrl.resolution} shape={grab(force=True).shape}", flush=True)
    elif cmd == "shot":
        img = grab(force=True)
        out = SHOTS / f"{a[0]}.png"
        cv2.imwrite(str(out), img)
        print(f"[shot] {out} {img.shape[1]}x{img.shape[0]}", flush=True)
    elif cmd == "tap":
        ctrl.post_click(int(a[0]), int(a[1])).wait()
        invalidate()
        print(f"[tap] {a[0]},{a[1]}", flush=True)
    elif cmd == "swipe":
        ms = int(a[4]) if len(a) > 4 else 400
        ctrl.post_swipe(int(a[0]), int(a[1]), int(a[2]), int(a[3]), ms).wait()
        invalidate()
        print(f"[swipe] {a[0]},{a[1]} -> {a[2]},{a[3]}", flush=True)
    elif cmd == "sleep":
        time.sleep(int(a[0]) / 1000.0)
        print(f"[sleep] {a[0]}ms", flush=True)
    elif cmd in ("match", "find", "tapimg"):
        roi = [int(v) for v in a[1:5]] if len(a) >= 5 else None
        r = do_match(a[0], roi)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        if cmd == "tapimg" and "center" in r:
            ctrl.post_click(r["center"][0], r["center"][1]).wait()
            invalidate()
            print(f"[tapimg] clicked {r['center']}", flush=True)
    else:
        print(f"[?] unknown: {cmd}", flush=True)
    return True


print(f"[ready] serial={SERIAL} resolution={ctrl.resolution}", flush=True)
for line in sys.stdin:
    if not handle(line):
        break
print("[bye]", flush=True)
