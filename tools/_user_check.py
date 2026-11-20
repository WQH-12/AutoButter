import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {}
for p in sorted(glob.glob(USER + r"\*.png")):
    name = os.path.basename(p).replace(".png", "")
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    tpls[name] = im

# 交叉相似度矩阵
names = list(tpls)
print("=== cross-correlation (want < 0.7) ===")
print("            " + " ".join(f"{n[:9]:>9s}" for n in names))
for a in names:
    row = []
    for b in names:
        ta, tb = tpls[a], tpls[b]
        if ta.shape != tb.shape:
            tb = cv2.resize(tb, (ta.shape[1], ta.shape[0]))
        r = cv2.matchTemplate(ta, tb, cv2.TM_CCOEFF_NORMED)[0][0]
        row.append(f"{r:9.2f}")
    print(f"{a[:10]:>10s}  " + " ".join(row))

# 底色采样：四角 + 边缘中点
print("\n=== border colour (BGR) ===")
for n, t in tpls.items():
    h, w = t.shape[:2]
    pts = [(1,1),(w-2,1),(1,h-2),(w-2,h-2),(w//2,1),(1,h//2)]
    cols = [t[y,x].tolist() for x,y in pts]
    hsv = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2HSV)[0,0].tolist() for c in cols]
    print(f"{n:12s} {cols[0]} .. {cols[1]}  H={hsv[0][0]:3d} S={hsv[0][1]:3d} V={hsv[0][2]:3d}")
