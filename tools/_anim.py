import cv2, numpy as np
# 抢占用动画：用连续截图看鸡妖/进度条的变化范围
shots = ["52_go_row2.png", "56_robbed.png"]
for s in shots:
    im = cv2.imread(rf"E:\work\AutoButter\screenshots\{s}")
    if im is None: continue
    # 格子下方区域（鸡妖动画通常在这里）
    print(f"{s}: {im.shape[1]}x{im.shape[0]}")
