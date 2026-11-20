import cv2, numpy as np, glob, os

p = r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png"
img = cv2.imread(p)
print("screenshot:", img.shape[1], "x", img.shape[0])

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

print("\n=== full-screen match ===")
for n, t in tpls.items():
    res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(res)
    th, tw = t.shape[:2]
    print(f"  {n:11s} {mx:.3f}  at ({ml[0]:3d},{ml[1]:3d}) center=({ml[0]+tw//2:3d},{ml[1]+th//2:3d})")
