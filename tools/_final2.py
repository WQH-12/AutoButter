import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}
names = list(tpls)

print("=== 7x7 交叉相关（同尺寸，应低）===")
print("            " + " ".join(f"{n[:8]:>8s}" for n in names))
for a in names:
    row = []
    for b in names:
        r = cv2.matchTemplate(tpls[a], tpls[b], cv2.TM_CCOEFF_NORMED)[0][0]
        row.append(f"{r:8.2f}")
    print(f"{a:12s}" + " ".join(row))

# 易混对：jade(玉璧) vs stone(云英) 都是圆形
print("\n=== 圆形物品易混测试：在 131441 里 jade 会不会误命中 stone ===")
img = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-131441_720.png")
for n in ["jade_1", "stone_1", "honey_1", "berry_1"]:
    t = tpls[n]
    res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    rr = res.copy(); pts=[]
    while True:
        _, mx, _, ml = cv2.minMaxLoc(rr)
        if mx < 0.80: break
        pts.append((round(float(mx),3), (ml[0]+t.shape[1]//2, ml[1]+t.shape[0]//2)))
        rr[max(0,ml[1]-20):ml[1]+20, max(0,ml[0]-20):ml[0]+20] = -1
    print(f"  {n:10s} {pts}")

# 误报临界点
print("\n=== 43_manage_now 里 berry_1=0.89 的位置 ===")
img2 = cv2.imread(r"E:\work\AutoButter\screenshots\43_manage_now.png")
t = tpls["berry_1"]
res = cv2.matchTemplate(img2, t, cv2.TM_CCOEFF_NORMED)
_, mx, _, ml = cv2.minMaxLoc(res)
print(f"  score={mx:.3f} at {ml}")
patch = img2[ml[1]:ml[1]+t.shape[0], ml[0]:ml[0]+t.shape[1]]
H=140
a = cv2.resize(t, (int(t.shape[1]*H/t.shape[0]), H), interpolation=cv2.INTER_NEAREST)
b = cv2.resize(patch, (a.shape[1], H), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(r"E:\work\AutoButter\screenshots\79_falsepos.png", np.hstack([a, np.full((H,10,3),255,np.uint8), b]))
print("  saved 79_falsepos.png (left=template, right=false positive)")
