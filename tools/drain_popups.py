"""清空「恭喜獲得」弹窗，并测量每个弹窗的实际耗时。

用于回答两个问题：
  1. 现在到底堆了多少个弹窗
  2. 每个弹窗最少要花多久（post_delay 能压到多少）

用法:
    python tools/drain_popups.py
    python tools/drain_popups.py --delay 300
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from maa.controller import AdbController
from adb_util import adb_path, pick_serial

ROOT = r"E:\work\AutoButter"
TPL = f"{ROOT}\\resource\\image\\obtain_resource.png"
ROI = (0, 400, 720, 400)  # x, y, w, h
THR = 0.9
# 与 pipeline 的 target_offset 保持一致：标题框中心 + 775 -> y≈1326
# 旧的 +300 落在弹窗面板内部，点了关不掉，会空转（见 HANDOVER §3.8）
CLICK_DY = 775


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None)
    ap.add_argument("--delay", type=int, default=250, help="点击后等待毫秒")
    ap.add_argument("--click-dy", type=int, default=CLICK_DY,
                    help="相对标题框中心的 y 偏移，默认 775（面板下方空白区）")
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--dump", default="")
    ap.add_argument("--dry-run", action="store_true", help="只报告当前有没有弹窗，不点击")
    args = ap.parse_args()

    if not args.serial:
        try:
            args.serial = pick_serial()
        except Exception as exc:  # noqa: BLE001
            print(f"[FATAL] 选择设备失败：{exc}")
            return 2
    print(f"[ok] adb serial: {args.serial}")

    tpl = cv2.imread(TPL)
    if tpl is None:
        print(f"[FATAL] template missing: {TPL}")
        return 2

    ctrl = AdbController(adb_path(), args.serial)
    ctrl.set_screenshot_target_short_side(720)
    if not ctrl.post_connection().wait().succeeded:
        print("[FATAL] connect failed")
        return 2

    x, y, w, h = ROI
    t_start = time.perf_counter()
    count = 0
    cap_total = 0.0
    last_item = None

    while count < args.max:
        t0 = time.perf_counter()
        img = ctrl.post_screencap().wait().get()
        cap_total += time.perf_counter() - t0

        crop = img[y:y + h, x:x + w]
        res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx < THR:
            print(f"[done] no popup left (best score {mx:.4f})")
            break

        if args.dry_run:
            print(f"[dry-run] popup present, best score {mx:.4f}; not clicking")
            return 0

        cx = x + ml[0] + tpl.shape[1] // 2
        cy = y + ml[1] + tpl.shape[0] // 2 + args.click_dy
        ctrl.post_click(cx, cy).wait()
        count += 1
        if count % 10 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  cleared {count} popups, {elapsed / count:.2f}s each", flush=True)
        time.sleep(args.delay / 1000.0)

    total = time.perf_counter() - t_start
    print()
    print(f"popups cleared : {count}")
    print(f"wall time      : {total:.1f}s")
    print(f"screencap time : {cap_total:.1f}s ({cap_total / max(count, 1) * 1000:.0f}ms per popup)")
    if count:
        print(f"per popup      : {total / count:.2f}s")

    if args.dump:
        img = ctrl.post_screencap().wait().get()
        cv2.imwrite(args.dump, img)
        print(f"[dump] {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
