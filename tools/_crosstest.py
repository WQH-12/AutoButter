import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
raw = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

new = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png")

# 用户模板 66x47：外框约 3px + 内边距，图标主体大约从 (5,4) 到 (61,43)
def trim(t, inset):
    return t[inset:t.shape[0]-inset, inset:t.shape[1]-inset]

# 目标：同物品不同底色。honey 是紫底 2级；找一张橙底/蓝底蜂蜜？先看已知格子
# 用 42_after_close 的橙底格子测试「去边框后能否跨底色匹配」
img = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")
# 42 里 fabric 橙底 2级 有多处；取 31_realm_popup 的蓝底 fabric
img31 = cv2.imread(r"E:\work\AutoButter\screenshots\31_realm_popup.png")

print("=== 跨底色测试：fabric 橙(模板) vs 蓝(实测) ===")
# 31_realm_popup slot#2 蓝底 fabric box=(367,696,89,92)  实际是 fabric
blue_patch = img31[696:696+92, 367:367+89]
for inset in [0, 3, 5, 7, 9, 12]:
    t = trim(raw["fabric_1"], inset)
    if t.shape[0] < 8: break
    # 蓝底 patch 里搜索
    res = cv2.matchTemplate(blue_patch, t, cv2.TM_CCOEFF_NORMED)
    s1 = res.max()
    # 橙底同尺寸 patch（42 slot#1 box=(62,681,86,91) 是 fabric）
    orange_patch = img[681:681+91, 62:62+86]
    res2 = cv2.matchTemplate(orange_patch, t, cv2.TM_CCOEFF_NORMED)
    s2 = res2.max()
    print(f"  inset={inset:2d} tpl={t.shape[1]}x{t.shape[0]}  blue={s1:.3f}  orange={s2:.3f}")
