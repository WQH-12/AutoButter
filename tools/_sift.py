import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

# 测试：SIFT 特征匹配对底色是否免疫
sift = cv2.SIFT_create()
def feat(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(g, None)
    return kp, des

bf = cv2.BFMatcher()
def sift_score(a, b):
    ka, da = feat(a); kb, db = feat(b)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return 0.0, 0
    m = bf.knnMatch(da, db, k=2)
    good = [p for p, q in m if len([p,q]) == 2 and p.distance < 0.75 * q.distance]
    return len(good) / max(1, min(len(ka), len(kb))), len(good)

# 同一物品不同底色：jade 橙 vs 蓝(stone_1 是蓝底，但不同物品)
# 用真实截图：42_after_close 有橙底 fabric，31_realm_popup 有蓝底 fabric
tests = [
    ("fabric_1 橙底 vs 蓝底", tpls["fabric_1"], None),
]
img_blue = cv2.imread(r"E:\work\AutoButter\screenshots\31_realm_popup.png")
img_orange = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")

# 蓝底 fabric：31_realm_popup slot#2 (367,696,89,92)
blue_fabric = img_blue[696+10:696+92-30, 367+10:367+89-10]
orange_fabric = img_orange[680+10:680+92-30, 163+10:163+89-10]
print("blue fabric", blue_fabric.shape, "orange fabric", orange_fabric.shape)

print("\n=== TemplateMatch (TM_CCOEFF_NORMED) ===")
for name, a in [("orange", orange_fabric), ("blue", blue_fabric)]:
    b = cv2.resize(a, (66, 47), interpolation=cv2.INTER_AREA)
    s = cv2.matchTemplate(tpls["fabric_1"], b, cv2.TM_CCOEFF_NORMED)[0][0]
    print(f"  fabric_1(橙) vs {name}: {s:.3f}")

print("\n=== SIFT ===")
for name, a in [("orange", orange_fabric), ("blue", blue_fabric)]:
    b = cv2.resize(a, (66, 47), interpolation=cv2.INTER_AREA)
    sc, n = sift_score(tpls["fabric_1"], b)
    print(f"  fabric_1(橙) vs {name}: inlier_ratio={sc:.3f} good={n}")

print("\n=== SIFT 区分不同物品 ===")
for n1 in ["fabric_1","jade_1","berry_1"]:
    for n2 in ["fabric_1","jade_1","berry_1"]:
        sc, g = sift_score(tpls[n1], tpls[n2])
        print(f"  {n1:10s} vs {n2:10s} ratio={sc:.3f} good={g}")
