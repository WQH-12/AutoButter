import cv2, numpy as np

# 决定性测试：只裁图标主体（去框去字），橙框 vs 蓝框 的同一物品能否互认
o = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")      # 橙框 fabric
b = cv2.imread(r"E:\work\AutoButter\screenshots\31_realm_popup.png")      # 蓝框 fabric（通道对调，先修）

# 31 是通道对调图，先修回来
b_fixed = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

def icon_only(img, x, y, w, h, ix=10, top=10, dbot=30):
    return img[y+top:y+h-dbot, x+ix:x+w-ix]

# 42: fabric 在 (62,681,86,91)
orange = icon_only(o, 62, 681, 86, 91)
# 31: fabric 在 (367,696,89,92)
blue = icon_only(b_fixed, 367, 696, 89, 92)
print("orange", orange.shape, "blue", blue.shape)

# 尺寸对齐
blue_r = cv2.resize(blue, (orange.shape[1], orange.shape[0]), interpolation=cv2.INTER_AREA)

print(f"\ncolor cross-frame : {cv2.matchTemplate(orange, blue_r, cv2.TM_CCOEFF_NORMED)[0][0]:.3f}")
print(f"gray  cross-frame : {cv2.matchTemplate(cv2.cvtColor(orange,cv2.COLOR_BGR2GRAY), cv2.cvtColor(blue_r,cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)[0][0]:.3f}")

# 带框版本做对照
orange_f = o[681:681+91, 62:62+86]
blue_f = b_fixed[696:696+92, 367:367+89]
blue_fr = cv2.resize(blue_f, (orange_f.shape[1], orange_f.shape[0]), interpolation=cv2.INTER_AREA)
print(f"\nWITH frame color  : {cv2.matchTemplate(orange_f, blue_fr, cv2.TM_CCOEFF_NORMED)[0][0]:.3f}")
print(f"WITH frame gray   : {cv2.matchTemplate(cv2.cvtColor(orange_f,cv2.COLOR_BGR2GRAY), cv2.cvtColor(blue_fr,cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)[0][0]:.3f}")

# 拼图确认两者确实同一物品
H=160
def big(p): return cv2.resize(p,(int(p.shape[1]*H/p.shape[0]),H),interpolation=cv2.INTER_NEAREST)
cv2.imwrite(r"E:\work\AutoButter\screenshots\84_icononly_crossframe.png",
            np.hstack([big(orange), np.full((H,10,3),255,np.uint8), big(blue_r)]))
print("\nsaved 84_icononly_crossframe.png (left=orange frame, right=blue frame)")
