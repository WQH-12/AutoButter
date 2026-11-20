import cv2, numpy as np
img = cv2.imread(r"E:\work\AutoButter\screenshots\burst\pad_test2\f00_720.png")
# cell0 上方：鸡妖 + 倒计时
crop = img[380:700, 40:200]
cv2.imwrite(r"E:\work\AutoButter\screenshots\90_robbed_zoom.png",
            cv2.resize(crop, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC))
print("90_robbed_zoom.png", crop.shape, "origin=(40,380)")
