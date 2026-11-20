import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

# 失败1: 48_go_target 里 berry 只有 1 个 >=0.90，看看第二个在哪、多少分
img = cv2.imread(r"E:\work\AutoButter\screenshots\48_go_target.png")
t = tpls["berry_1"]
res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
rr = res.copy()
print("48_go_target berry_1 peaks:")
for i in range(6):
    _, mx, _, ml = cv2.minMaxLoc(rr)
    print(f"  #{i} {mx:.3f} @({ml[0]+33},{ml[1]+23})")
    rr[max(0,ml[1]-18):ml[1]+18, max(0,ml[0]-18):ml[0]+18] = -1

# 失败2: 45_tanxun_list 里 fabric(卷轴) 的分数
img2 = cv2.imread(r"E:\work\AutoButter\screenshots\45_tanxun_list.png")
t2 = tpls["fabric_1"]
res2 = cv2.matchTemplate(img2, t2, cv2.TM_CCOEFF_NORMED)
rr2 = res2.copy()
print("\n45_tanxun_list fabric_1 peaks:")
for i in range(4):
    _, mx, _, ml = cv2.minMaxLoc(rr2)
    print(f"  #{i} {mx:.3f} @({ml[0]+33},{ml[1]+23})")
    rr2[max(0,ml[1]-18):ml[1]+18, max(0,ml[0]-18):ml[0]+18] = -1

# 列表页图标尺寸 vs 秘境格子内图标尺寸
print("\n=== 列表页卷轴 vs 秘境格子卷轴 尺寸对比 ===")
lst_icon = img2[489:489+47, 257:257+66]   # jade 在列表里的框
print(f"  列表图标: 66x47 (已知)")
# 秘境格子：42 里 fabric @ (62,681,86,91)，内部图标约
print(f"  秘境格子: 86x91 外框，内部图标约 66x47（模板尺寸）")
