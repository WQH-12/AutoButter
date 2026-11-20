import cv2, numpy as np, glob, os
from collections import defaultdict

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

def border_kind(img, x, y, cw, ch):
    strip = img[y+2:y+6, x+cw//2-12:x+cw//2+12]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    h = int(np.median(hsv[:,:,0])); s = int(np.median(hsv[:,:,1]))
    if s < 55: return "gray"
    if h < 22 or h > 170: return "ORANGE"
    if 22 <= h < 40: return "YELLOW"
    if 40 <= h < 92: return "GREEN"
    if 92 <= h <= 112: return "BLUE"
    return "PURPLE"

# 全库扫描：每个物品在哪些底色下拿到 >=0.9 分
hits = defaultdict(lambda: defaultdict(int))
for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")) + sorted(glob.glob(r"E:\work\AutoButter\screenshots\snap\*.png")):
    base = os.path.basename(p)
    if base[:3] in ("60_","61_","62_","64_","65_","66_","70_","71_","72_","73_","74_","75_"): continue
    img = cv2.imread(p)
    if img is None or img.shape[0] < 1000: continue
    for n, t in tpls.items():
        if img.shape[0] < t.shape[0] or img.shape[1] < t.shape[1]: continue
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        # 取所有 >=0.9 的局部极大
        rr = res.copy()
        for _ in range(8):
            _, mx, _, ml = cv2.minMaxLoc(rr)
            if mx < 0.90: break
            col = border_kind(img, ml[0], ml[1], t.shape[1], t.shape[0])
            hits[n][col] += 1
            rr[max(0,ml[1]-15):ml[1]+15, max(0,ml[0]-15):ml[0]+15] = -1

print("=== 各物品出现过的边框色（score >= 0.90）===")
for n in sorted(hits):
    print(f"  {n:11s} {dict(hits[n])}")
