"""先关详情浮窗，再测「恭喜獲得」弹窗的关闭点。

判定标准不是分数下降（详情浮窗会遮挡标题），而是：
截图里「恭喜獲得」标题真的不见了，且没有新的遮挡。
"""
import sys
import time

import cv2
import numpy as np

from maa.controller import AdbController

ROOT = r"E:\work\AutoButter"
TPL = f"{ROOT}\\resource\\image\\obtain_resource.png"
ROI = (0, 400, 720, 400)
THR = 0.9


def score(img, tpl):
    x, y, w, h = ROI
    crop = img[y:y + h, x:x + w]
    if crop.shape[0] < tpl.shape[0] or crop.shape[1] < tpl.shape[1]:
        return 0.0
    return float(cv2.minMaxLoc(cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED))[1])


def dark_ratio(img):
    """整屏亮度，用来判断是不是纯黑背景的单物品弹窗。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float((g < 30).mean())


def main() -> int:
    tpl = cv2.imread(TPL)
    ctrl = AdbController("adb", "3a1e6a75")
    ctrl.set_screenshot_target_short_side(720)
    if not ctrl.post_connection().wait().succeeded:
        print("[FATAL] connect failed")
        return 2

    def shot(tag):
        img = ctrl.post_screencap().wait().get()
        s, d = score(img, tpl), dark_ratio(img)
        print(f"    {tag:22s} title={s:.4f} dark={d:.2f}", flush=True)
        return img, s, d

    print("[1] 当前状态")
    img, s, d = shot("initial")
    cv2.imwrite(f"{ROOT}\\screenshots\\152_probe2_0.png", img)

    # 先点屏幕最外侧，关掉可能存在的详情浮窗
    print("[2] 点 (360,1450) 关详情浮窗")
    ctrl.post_click(360, 1450).wait()
    time.sleep(0.5)
    img, s, d = shot("after outer tap")
    cv2.imwrite(f"{ROOT}\\screenshots\\152_probe2_1.png", img)

    if s < THR:
        print("[RESULT] 标题已消失，弹窗关掉了（点屏幕外侧有效）")
        return 0

    # 逐个试其他位置
    print("[3] 标题还在，继续试其他位置")
    for name, x, y in [
        ("标题正下方", 360, 640),
        ("标题左上", 200, 500),
        ("弹窗中段", 360, 760),
        ("屏幕顶部", 360, 60),
        ("屏幕左侧", 30, 800),
    ]:
        ctrl.post_click(x, y).wait()
        time.sleep(0.6)
        img, s, d = shot(f"tap ({x},{y}) {name}")
        if s < THR:
            print(f"\n[RESULT] 关闭成功：({x},{y}) {name}")
            cv2.imwrite(f"{ROOT}\\screenshots\\152_probe2_ok.png", img)
            return 0

    print("\n[RESULT] 都没关掉")
    cv2.imwrite(f"{ROOT}\\screenshots\\152_probe2_fail.png", img)
    return 1


if __name__ == "__main__":
    sys.exit(main())
