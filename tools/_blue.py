import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

# 1) 尺寸一致性
print("=== size / format ===")
for n, t in tpls.items():
    rgba = cv2.imread(os.path.join(USER, n + ".png"), cv2.IMREAD_UNCHANGED)
    print(f"  {n:11s} {t.shape[1]}x{t.shape[0]}  alpha_all_255={bool((rgba[:,:,3]==255).all())}")

# 2) 蓝屏状态：31_realm_popup 全屏蓝化，看色相
img31 = cv2.imread(r"E:\work\AutoButter\screenshots\31_realm_popup.png")
img42 = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")
for tag, im in [("31_realm_popup", img31), ("42_after_close", img42)]:
    roi = im[600:800, 50:680]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    print(f"\n{tag}: 内容区中位色 H={int(np.median(hsv[:,:,0]))} S={int(np.median(hsv[:,:,1]))} V={int(np.median(hsv[:,:,2]))}")
    print(f"  蓝像素占比(H 95-115, S>60) = {100.0*(((hsv[:,:,0]>=95)&(hsv[:,:,0]<=115)&(hsv[:,:,1]>60)).mean()):.1f}%")

# 3) 蓝屏下把模板也做同样的蓝色偏移，能否恢复匹配
print("\n=== 蓝屏补偿测试 ===")
def blue_tint(t):
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:,:,0] = (hsv[:,:,0] + 90) % 180   # 橙->蓝
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

for n in ["alcohol_1", "fabric_1"]:
    t = tpls[n]
    tb = blue_tint(t)
    r1 = cv2.matchTemplate(img31, t, cv2.TM_CCOEFF_NORMED).max()
    r2 = cv2.matchTemplate(img31, tb, cv2.TM_CCOEFF_NORMED).max()
    print(f"  {n:11s} original={r1:.3f}  blue-shifted={r2:.3f}")
