import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(p)[:-4]: cv2.imread(p, cv2.IMREAD_COLOR) for p in glob.glob(USER + r"\*.png")}

# honey 放大看
t = tpls["honey_1"]
cv2.imwrite(r"E:\work\AutoButter\screenshots\72_honey_zoom.png",
            cv2.resize(t, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST))

# 全库扫描 honey，列出所有 >0.55 的位置
print("=== honey_1 across all screenshots (score > 0.55) ===")
for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")):
    img = cv2.imread(p)
    if img is None or img.shape[0] < 900: continue
    if img.shape[0] < t.shape[0] or img.shape[1] < t.shape[1]: continue
    res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(res)
    if mx > 0.55:
        print(f"  {os.path.basename(p):26s} {mx:.3f} at {ml}")
