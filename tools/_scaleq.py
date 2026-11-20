import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

# 关键问题：66x47 的模板能否匹配「秘境大格子」(89x92) 里的图标？
img = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")
print("=== 42_after_close: 每个模板的所有 >=0.85 命中位置 ===")
for n, t in tpls.items():
    res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    rr = res.copy()
    pts = []
    while True:
        _, mx, _, ml = cv2.minMaxLoc(rr)
        if mx < 0.85: break
        pts.append((round(float(mx),3), ml))
        rr[max(0,ml[1]-20):ml[1]+20, max(0,ml[0]-20):ml[0]+20] = -1
    print(f"  {n:11s} {pts}")

print("\n秘境格子坐标: (62,681) (163,680) (367,696) (468,712) (570,688) 尺寸 ~89x92")
