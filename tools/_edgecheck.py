import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

print("=== 用户模板边缘是什么颜色？（判断是否含边框）===")
print(f"{'tpl':11s} {'top strip':>18s} {'bottom strip':>18s} {'left':>14s} {'right':>14s}")
for n, t in tpls.items():
    h, w = t.shape[:2]
    def med(strip):
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        return f"H{int(np.median(hsv[:,:,0])):3d}S{int(np.median(hsv[:,:,1])):3d}"
    print(f"{n:11s} {med(t[0:2, w//3:2*w//3]):>18s} {med(t[h-2:h, w//3:2*w//3]):>18s} "
          f"{med(t[h//3:2*h//3, 0:2]):>14s} {med(t[h//3:2*h//3, w-2:w]):>14s}")

print("\n=== 定位：honey_1 在 130821 里命中 1.000 的位置，反推它裁自哪个框 ===")
img = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png")
t = tpls["honey_1"]
res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
_, mx, _, ml = cv2.minMaxLoc(res)
print(f"  honey_1 {t.shape[1]}x{t.shape[0]} score={mx:.4f} at ({ml[0]},{ml[1]})")
print(f"  该处格子检测框是 (460,683,95,93)")
print(f"  反推 inset: left={ml[0]-460} top={ml[1]-683} right={(460+95)-(ml[0]+t.shape[1])} bottom={(683+93)-(ml[1]+t.shape[0])}")

print("\n=== 同样反推其它模板 ===")
for n in ["alcohol_1", "jade_1", "berry_1"]:
    tt = tpls[n]
    r = cv2.matchTemplate(img, tt, cv2.TM_CCOEFF_NORMED)
    _, m, _, l = cv2.minMaxLoc(r)
    print(f"  {n:11s} score={m:.3f} at ({l[0]},{l[1]})  size={tt.shape[1]}x{tt.shape[0]}")
