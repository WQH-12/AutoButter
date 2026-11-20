"""找出「恭喜獲得」弹窗真正能被关掉的点击位置。

判定方法：不是看标题分数（下一个弹窗标题位置相同，分数不变），
而是看弹窗内容区（物品图标带）的像素签名是否变化。

  - 签名变化 -> 点击生效，弹窗换了一条
  - 签名不变 -> 点击没生效，弹窗还是同一条
  - 标题消失 -> 弹窗清空

用法:
    python tools/probe_dismiss_point.py
    python tools/probe_dismiss_point.py --points 360,800 120,800
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time

import cv2

from maa.controller import AdbController

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
TPL = f"{ROOT}\\resource\\image\\obtain_resource.png"
TITLE_ROI = (0, 400, 720, 400)
# 物品图标带：标题下方
ITEM_ROI = (60, 640, 600, 180)
THR = 0.9


def title_score(img, tpl):
    x, y, w, h = TITLE_ROI
    crop = img[y:y + h, x:x + w]
    return float(cv2.minMaxLoc(cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED))[1])


def item_signature(img):
    x, y, w, h = ITEM_ROI
    crop = img[y:y + h, x:x + w]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (60, 18), interpolation=cv2.INTER_AREA)
    return hashlib.md5(g.tobytes()).hexdigest()[:10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="*", default=[])
    ap.add_argument("--wait", type=float, default=0.7)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    tpl = cv2.imread(TPL)
    ctrl = AdbController("adb", "3a1e6a75")
    ctrl.set_screenshot_target_short_side(720)
    if not ctrl.post_connection().wait().succeeded:
        print("[FATAL] connect failed")
        return 2

    def snap():
        img = ctrl.post_screencap().wait().get()
        return img, title_score(img, tpl), item_signature(img)

    img, s, sig = snap()
    print(f"[start] title={s:.4f} item_sig={sig}")
    if s < THR:
        print("[skip] 当前没有弹窗")
        return 0

    candidates = args.points or [
        "360,470", "360,800", "360,900", "360,1000", "360,1050",
        "120,800", "600,800", "120,1000", "600,1000",
    ]

    for spec in candidates:
        x, y = (int(v) for v in spec.split(","))
        print(f"\n[test] ({x},{y})")
        for i in range(args.repeat):
            before_img, before_s, before_sig = snap()
            if before_s < THR:
                print("  弹窗已清空，结束")
                return 0
            ctrl.post_click(x, y).wait()
            time.sleep(args.wait)
            after_img, after_s, after_sig = snap()
            if after_s < THR:
                print(f"  #{i + 1} 标题消失 -> 关闭成功")
                cv2.imwrite(f"{ROOT}\\screenshots\\_dismiss_ok_{x}_{y}.png", after_img)
                return 0
            changed = after_sig != before_sig
            print(f"  #{i + 1} title={after_s:.4f} sig {before_sig}->{after_sig} "
                  f"{'CHANGED(生效)' if changed else 'SAME(无效)'}")
            if changed:
                cv2.imwrite(f"{ROOT}\\screenshots\\_dismiss_work_{x}_{y}.png", after_img)
                break

    print("\n[RESULT] 没有找到生效的落点")
    return 1


if __name__ == "__main__":
    sys.exit(main())
