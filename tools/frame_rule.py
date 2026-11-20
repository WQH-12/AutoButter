import cv2, numpy as np, glob, os
from collections import defaultdict

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

def is_swapped(img):
    roi = img[int(img.shape[0]*.38):int(img.shape[0]*.58), 60:660]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:,:,0]>=95)&(hsv[:,:,0]<=115)&(hsv[:,:,1]>60)).mean()
    warm = (((hsv[:,:,0]<=25)|(hsv[:,:,0]>=170))&(hsv[:,:,1]>60)).mean()
    return blue > warm

# 假设：边框色 = 物品稀有度，每件物品固定
# 验证：只在“干净”截图上，用 0.95 高阈值统计
agg = defaultdict(lambda: defaultdict(int))
files = sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")) + sorted(glob.glob(r"E:\work\AutoButter\screenshots\snap\*_720.png"))
for p in files:
    b = os.path.basename(p)
    if b[0] == "_" or b[:2] in ("60","61","62","64","65","66","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86"): continue
    im = cv2.imread(p)
    if im is None or im.shape[0] < 900 or is_swapped(im): continue
    for name, t in tpls.items():
        if im.shape[0] < t.shape[0] or im.shape[1] < t.shape[1]: continue
        res = cv2.matchTemplate(im, t, cv2.TM_CCOEFF_NORMED)
        rr = res.copy()
        while True:
            _, mx, _, ml = cv2.minMaxLoc(rr)
            if mx < 0.95: break
            strip = im[ml[1]:ml[1]+3, ml[0]+t.shape[1]//3:ml[0]+2*t.shape[1]//3]
            hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
            h = int(np.median(hsv[:,:,0])); s = int(np.median(hsv[:,:,1]))
            col = "gray" if s < 55 else ("orange" if (h<22 or h>170) else ("green" if 40<=h<92 else ("blue" if 92<=h<=112 else "purple")))
            agg[name][col] += 1
            rr[max(0,ml[1]-18):ml[1]+18, max(0,ml[0]-18):ml[0]+18] = -1

print("=== 边框色 vs 物品（阈值 0.95，只统计干净截图）===")
for n in sorted(agg):
    d = dict(agg[n])
    total = sum(d.values())
    dominant = max(d, key=d.get)
    consistent = len(d) == 1
    print(f"  {n:11s} {d}   -> 主色 {dominant} ({100*d[dominant]//total}%)  {'一致' if consistent else '不一致'}")
