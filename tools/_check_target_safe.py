"""确认「点面板外」这个落点在无弹窗时不会误触 UI。"""
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter\screenshots"
img = cv2.imread(f"{ROOT}\\154_after_fix.png")
print("shape", img.shape)

# 标出 (350, 1326)
out = img.copy()
cv2.drawMarker(out, (350, 1326), (0, 0, 255), cv2.MARKER_CROSS, 60, 4)
cv2.imwrite(f"{ROOT}\\_target_on_realm.png", out)

# 裁出该点周围，看看压到什么
cv2.imwrite(f"{ROOT}\\_target_around.png", img[1250:1420, 240:470])
print("saved _target_on_realm.png / _target_around.png")

# 逐行统计亮像素，定位底部按钮区
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bright = (g > 120).mean(axis=1)
runs = []
start = None
for y, v in enumerate(bright):
    if v > 0.10 and start is None:
        start = y
    elif v <= 0.10 and start is not None:
        if y - start > 8:
            runs.append((start, y))
        start = None
print("bright row bands:", runs[-6:])
