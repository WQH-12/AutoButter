import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

img = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png")

# 造一个“同物品换底色”样本：把 honey 的紫框改成橙框
honey = tpls["honey_1"]
hsv = cv2.cvtColor(honey, cv2.COLOR_BGR2HSV)
mask = ((hsv[:,:,0] >= 110) & (hsv[:,:,0] <= 170) & (hsv[:,:,1] > 45)).astype(np.uint8)
print("purple frame pixels:", int(mask.sum()), "/", mask.size)
recol = honey.copy()
recol[mask.astype(bool)] = (100, 180, 250)   # BGR 橙
cv2.imwrite(r"E:\work\AutoButter\screenshots\78_honey_recolored.png", recol)

# 三种模式：彩色 / 灰度
def g(x): return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x

print("\n=== 同物品，紫框模板 vs 橙框改色样本 ===")
print(f"  color : {cv2.matchTemplate(honey, recol, cv2.TM_CCOEFF_NORMED)[0][0]:.3f}")
print(f"  gray  : {cv2.matchTemplate(g(honey), g(recol), cv2.TM_CCOEFF_NORMED)[0][0]:.3f}")

# 在真实截图里找 honey（紫框）与各模板，比较 color vs gray 的区分度
print("\n=== 真实截图 130821：color vs gray 的分数 ===")
print(f"  {'tpl':11s} {'color':>6s} {'gray':>6s}")
for n, t in tpls.items():
    sc = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED).max()
    sg = cv2.matchTemplate(g(img), g(t), cv2.TM_CCOEFF_NORMED).max()
    print(f"  {n:11s} {sc:6.3f} {sg:6.3f}")
