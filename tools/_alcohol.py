import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
img = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png")

# alcohol_1 自检只有 0.945，定位它到底匹配在哪
t = cv2.imread(os.path.join(USER, "alcohol_1.png"))
res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
_, mx, _, ml = cv2.minMaxLoc(res)
print(f"alcohol_1 best={mx:.4f} at {ml}  (检测框 163,568)")
print("差异可能来自：格子上叠了鸡妖/计时器，或裁切时含了半格边框")

# 对比：模板 vs 实际区域
patch = img[ml[1]:ml[1]+t.shape[0], ml[0]:ml[0]+t.shape[1]]
diff = cv2.absdiff(t, patch)
print(f"mean abs diff = {diff.mean():.1f}, max = {diff.max()}")

H = 180
a = cv2.resize(t, (int(t.shape[1]*H/t.shape[0]), H), interpolation=cv2.INTER_NEAREST)
b = cv2.resize(patch, (int(patch.shape[1]*H/patch.shape[0]), H), interpolation=cv2.INTER_NEAREST)
d = cv2.resize(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), (a.shape[1], H), interpolation=cv2.INTER_NEAREST)
d3 = cv2.applyColorMap(d, cv2.COLORMAP_JET)
canvas = np.hstack([a, np.full((H,8,3),255,np.uint8), b, np.full((H,8,3),255,np.uint8), d3])
canvas = cv2.copyMakeBorder(canvas, 26, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255,255,255))
for i, s in enumerate(["TEMPLATE", "SCREEN", "DIFF"]):
    cv2.putText(canvas, s, (4 + i*(a.shape[1]+8), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
cv2.imwrite(r"E:\work\AutoButter\screenshots\77_alcohol_diff.png", canvas)
print("saved 77_alcohol_diff.png")

# 每个模板四角颜色，判断含多少边框
print("\n=== 四角颜色（判断边框含量）===")
for p in sorted(glob.glob(USER + r"\*.png")):
    tt = cv2.imread(p, cv2.IMREAD_COLOR)
    h, w = tt.shape[:2]
    corners = [tt[0,0], tt[0,w-1], tt[h-1,0], tt[h-1,w-1]]
    print(f"  {os.path.basename(p):14s} TL={corners[0]} TR={corners[1]} BL={corners[2]} BR={corners[3]}")
