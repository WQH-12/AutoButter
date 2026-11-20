import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
print("=== 每个模板能否在已有截图里找到 1.000 的精确来源 ===")
for p in sorted(glob.glob(USER + r"\*.png")):
    name = os.path.basename(p)[:-4]
    t = cv2.imread(p, cv2.IMREAD_COLOR)
    best = (0.0, None, None)
    for sp in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")) + sorted(glob.glob(r"E:\work\AutoButter\screenshots\snap\*_720.png")):
        im = cv2.imread(sp)
        if im is None or im.shape[0] < t.shape[0] or im.shape[1] < t.shape[1]: continue
        _, mx, _, ml = cv2.minMaxLoc(cv2.matchTemplate(im, t, cv2.TM_CCOEFF_NORMED))
        if mx > best[0]:
            best = (float(mx), os.path.basename(sp), ml)
    flag = "EXACT" if best[0] > 0.999 else ("close" if best[0] > 0.99 else "NO EXACT SOURCE")
    print(f"  {name:11s} best={best[0]:.4f} in {str(best[1]):28s} at {best[2]}  {flag}")
