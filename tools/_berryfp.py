import cv2, numpy as np
img = cv2.imread(r"E:\work\AutoButter\screenshots\43_manage_now.png")
snap = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png")

# berry 模板 (b 方案)
x,y,w,h = 570,708,89,91
t = snap[y+10:y+h-30, x+10:x+w-10]

res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
_, mx, _, ml = cv2.minMaxLoc(res)
print(f"berry_1(b) best={mx:.3f} at {ml}")

patch = img[ml[1]:ml[1]+t.shape[0], ml[0]:ml[0]+t.shape[1]]
H = 150
a = cv2.resize(t, (int(t.shape[1]*H/t.shape[0]), H), interpolation=cv2.INTER_NEAREST)
b = cv2.resize(patch, (a.shape[1], H), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(r"E:\work\AutoButter\screenshots\82_berry_fp.png", np.hstack([a, np.full((H,10,3),255,np.uint8), b]))
print("saved 82_berry_fp.png")

# 全图位置
cv2.imwrite(r"E:\work\AutoButter\screenshots\83_berry_fp_context.png",
            cv2.resize(img[max(0,ml[1]-60):ml[1]+t.shape[0]+60, max(0,ml[0]-60):ml[0]+t.shape[1]+60], None, fx=2, fy=2))
