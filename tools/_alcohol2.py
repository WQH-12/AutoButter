import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

# alcohol_1 自检 0.945：把模板与最佳位置并排放大
img = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png")
t = tpls["alcohol_1"]
res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
_, mx, _, ml = cv2.minMaxLoc(res)
patch = img[ml[1]:ml[1]+t.shape[0], ml[0]:ml[0]+t.shape[1]]

H = 260
a = cv2.resize(t, (int(t.shape[1]*H/t.shape[0]), H), interpolation=cv2.INTER_NEAREST)
b = cv2.resize(patch, (a.shape[1], H), interpolation=cv2.INTER_NEAREST)
out = np.hstack([a, np.full((H,12,3),255,np.uint8), b])
out = cv2.copyMakeBorder(out, 24, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255,255,255))
cv2.putText(out, f"alcohol_1 TEMPLATE {t.shape[1]}x{t.shape[0]}", (6,17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
cv2.putText(out, f"BEST MATCH {mx:.3f}", (a.shape[1]+22, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
cv2.imwrite(r"E:\work\AutoButter\screenshots\85_alcohol_zoom.png", out)
print(f"score={mx:.4f} at {ml}")
