import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

img_blue = cv2.imread(r"E:\work\AutoButter\screenshots\31_realm_popup.png")
img_norm = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")

def prep(img, mode):
    if mode == "color":
        return img
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "gray":
        return g
    if mode == "edge":
        return cv2.Canny(g, 50, 150)
    if mode == "norm":
        # 逐通道去均值归一化，抵消整体色偏
        f = img.astype(np.float32)
        f -= f.reshape(-1,3).mean(axis=0)
        return f

print(f"{'mode':6s} {'item':11s} {'blue_img':>9s} {'normal_img':>11s}")
print("-"*44)
for mode in ["color", "gray", "edge"]:
    for n in ["alcohol_1", "fabric_1", "jade_1", "berry_1"]:
        t = prep(tpls[n], mode)
        a = prep(img_blue, mode)
        b = prep(img_norm, mode)
        s1 = cv2.matchTemplate(a, t, cv2.TM_CCOEFF_NORMED).max()
        s2 = cv2.matchTemplate(b, t, cv2.TM_CCOEFF_NORMED).max()
        print(f"{mode:6s} {n:11s} {s1:9.3f} {s2:11.3f}")
    print()
