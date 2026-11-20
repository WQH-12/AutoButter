"""在单物品「恭喜獲得」弹窗上逐个试点击位置，看哪个能关掉。

每次点击后截图，比对标题模板分数是否跌破阈值。
用法: python tools/_probe_tap.py
"""
import sys
import time

import cv2

from maa.controller import AdbController

ROOT = r"E:\work\AutoButter"
TPL = f"{ROOT}\\resource\\image\\obtain_resource.png"
ROI = (0, 400, 720, 400)
THR = 0.9


def popup_score(img, tpl):
    x, y, w, h = ROI
    crop = img[y:y + h, x:x + w]
    res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, _ = cv2.minMaxLoc(res)
    return float(mx)


def main() -> int:
    tpl = cv2.imread(TPL)
    ctrl = AdbController("adb", "3a1e6a75")
    ctrl.set_screenshot_target_short_side(720)
    if not ctrl.post_connection().wait().succeeded:
        print("[FATAL] connect failed")
        return 2

    img = ctrl.post_screencap().wait().get()
    s0 = popup_score(img, tpl)
    print(f"[start] popup score = {s0:.4f}")
    if s0 < THR:
        print("[skip] no popup on screen")
        return 0

    # 720 画布上的候选点
    candidates = [
        ("标题上方空白", 360, 420),
        ("标题本体", 360, 551),
        ("物品图标", 360, 700),
        ("物品下方文字", 360, 762),
        ("当前落点(851)", 360, 851),
        ("弹窗底部空白", 360, 950),
        ("屏幕正中", 360, 780),
        ("左下空白", 120, 900),
        ("右下空白", 600, 900),
        ("屏幕最底部", 360, 1500),
    ]

    for name, x, y in candidates:
        ctrl.post_click(x, y).wait()
        time.sleep(0.45)
        img = ctrl.post_screencap().wait().get()
        s = popup_score(img, tpl)
        print(f"  tap ({x:4d},{y:4d}) {name:16s} -> score {s:.4f} "
              f"{'CLOSED' if s < THR else ''}", flush=True)
        if s < THR:
            print(f"\n[RESULT] 关闭成功：({x}, {y}) {name}")
            cv2.imwrite(f"{ROOT}\\screenshots\\150_after_close.png", img)
            return 0

    print("\n[RESULT] 所有候选点都没关掉")
    cv2.imwrite(f"{ROOT}\\screenshots\\150_still_open.png", img)
    return 1


if __name__ == "__main__":
    sys.exit(main())
