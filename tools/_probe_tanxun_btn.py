"""量「探尋」按钮在三种界面上的位置与外观差异。只读。

三种界面：
  自己秘境面板   User_screenshot/full/720_realm_panel.png   （右下是「探尋託管中」）
  别人秘境       User_screenshot/full/720_fullscreen_others_home.png（右下是「探尋」）
  探寻列表       User_screenshot/full/720_list_top.png      （右下是「探尋託管中」）
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

USER = r"E:\work\AutoButter\User_screenshot\full"


def ascii_map(im, x0, x1, y0, y1, step=4, xstep=4):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    for y in range(y0, y1, step):
        line = ""
        for x in range(x0, x1, xstep):
            c = g[y:y + step, x:x + xstep].mean()
            line += " " if c < 60 else ("." if c < 110 else
                                        ("-" if c < 160 else ("+" if c < 210 else "#")))
        print("%4d %s" % (y, line))


def main():
    files = {
        "realm_panel": os.path.join(USER, "720_realm_panel.png"),
        "others_home": os.path.join(USER, "720_fullscreen_others_home.png"),
        "list_top": os.path.join(USER, "720_list_top.png"),
    }
    imgs = {k: cv2.imread(v) for k, v in files.items()}

    print("=== 底部按钮条 y 1250..1330 的版面")
    for k, im in imgs.items():
        print("\n--", k)
        ascii_map(im, 440, 720, 1250, 1330, 4, 4)

    print("\n=== 右下角（x 460..660, y 1265..1325）互相匹配")
    rois = {}
    for k, im in imgs.items():
        rois[k] = im[1265:1325, 460:660]
    keys = list(rois)
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i == j:
                continue
            a, b = rois[keys[i]], rois[keys[j]]
            d = cv2.absdiff(a, b)
            print("   %-12s vs %-12s max=%3d mean=%5.2f"
                  % (keys[i], keys[j], int(d.max()), float(d.mean())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
