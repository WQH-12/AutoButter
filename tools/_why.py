import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(p)[:-4]: cv2.imread(p, cv2.IMREAD_COLOR) for p in glob.glob(USER + r"\*.png")}

# 逐张查：为什么 honey_1 / jade_1 全屏最佳分数恒定 0.501 / 0.283
for name in ["honey_1", "jade_1"]:
    tpl = tpls[name]
    print(f"\n### {name} {tpl.shape}")
    # 打印模板自身是否有大片纯色
    hsv = cv2.cvtColor(tpl, cv2.COLOR_BGR2HSV)
    print("  border mean BGR:", tpl[0,:,:].mean(axis=0).round(1), tpl[-1,:,:].mean(axis=0).round(1))
    print("  corner BGR:", tpl[1,1], tpl[1,-2], tpl[-2,1], tpl[-2,-2])

# 找出 42_after_close 里 honey 和 jade 的真实位置
img = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")
print("\n42_after_close shape", img.shape)

# 用较小阈值看所有响应峰
for name in ["honey_1","jade_1","silver_1"]:
    tpl = tpls[name]
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    # 取前 5 个局部极大
    r = res.copy()
    pts = []
    for _ in range(5):
        _, mx, _, ml = cv2.minMaxLoc(r)
        pts.append((round(mx,3), ml))
        x0,y0 = max(0,ml[0]-20), max(0,ml[1]-20)
        r[y0:ml[1]+20, x0:ml[0]+20] = -1
    print(f"{name:10s}", pts)
