import cv2, numpy as np, glob, os
from collections import defaultdict

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

# 灰度模式：全库重测，统计每个物品的最佳分与误报
shots = []
for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")) + sorted(glob.glob(r"E:\work\AutoButter\screenshots\snap\*.png")):
    b = os.path.basename(p)
    if b[:3] in ("60_","61_","62_","64_","65_","66_","70_","71_","72_","73_","74_","75_","76_","_s31"): continue
    im = cv2.imread(p)
    if im is None or im.shape[0] < 900: continue
    shots.append((b, im))

def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x

print("=== 灰度模板：各图最高分（含误报风险）===")
for n, t0 in tpls.items():
    t = gray(t0)
    rows = []
    for b, im in shots:
        g = gray(im)
        if g.shape[0] < t.shape[0] or g.shape[1] < t.shape[1]: continue
        res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
        rr = res.copy(); cnt = 0; mx_all = 0
        while True:
            _, mx, _, ml = cv2.minMaxLoc(rr)
            if mx < 0.85: break
            cnt += 1; mx_all = max(mx_all, mx)
            rr[max(0,ml[1]-20):ml[1]+20, max(0,ml[0]-20):ml[0]+20] = -1
        if cnt: rows.append((b, cnt, mx_all))
    print(f"\n  {n}")
    for b, cnt, mx in sorted(rows, key=lambda r: -r[2])[:6]:
        print(f"    {b:30s} hits={cnt} best={mx:.3f}")
