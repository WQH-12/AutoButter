import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter\screenshots"
img = cv2.imread(f"{ROOT}\\154_after_fix.png")
# 弹窗面板下方到 管理 按钮之间
cv2.imwrite(f"{ROOT}\\_band_950_1120.png", img[950:1120, :])
# 再往上一点，含弹窗底边
cv2.imwrite(f"{ROOT}\\_band_900_1050.png", img[900:1050, :])
print("saved _band_950_1120.png / _band_900_1050.png")
print("image", img.shape)
