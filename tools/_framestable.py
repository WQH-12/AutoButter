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

# 只用干净截图
files = sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")) + sorted(glob.glob(r"E:\work\AutoButter\screenshots\snap\*_720.png"))
clean = []
for p in files:
    b = os.path.basename(p)
    if b[:3] in ("60_","61_","62_","64_","65_","66_","70_","71_","72_","73_","74_","75_","76_","77_","78_","79_","80_","81_","82_","83_","84_","_s31"): continue
    im = cv2.imread(p)
    if im is None or im.shape[0] < 900: continue
    if is_swapped(im): continue
    clean.append((b, im))
print(f"clean screenshots: {len(clean)}")

# 每个物品：命中位置的边框色统计
agg = defaultdict(lambda: defaultdict(int))
for name, t in tpls.items():
    for b, im in clean:
        if im.shape[0] < t.shape[0] or im.shape[1] < t.shape[1]: continue
        res = cv2.matchTemplate(im, t, cv2.TM_CCOEFF_NORMED)
        rr = res.copy()
        while True:
            _, mx, _, ml = cv2.minMaxLoc(rr)
            if mx < 0.92: break
            strip = im[ml[1]:ml[1]+3, ml[0]+t.shape[1]//3:ml[0]+2*t.shape[1]//3]
            hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
            h = int(np.median(hsv[:,:,0])); s = int(np.median(hsv[:,:,1]))
            col = "gray" if s < 55 else ("orange" if (h<22 or h>170) else ("green" if 40<=h<92 else ("blue" if 92<=h<=112 else "purple")))
            agg[name][col] += 1
            rr[max(0,ml[1]-18):ml[1]+18, max(0,ml[0]-18):ml[0]+18] = -1

print("\n=== 每个物品在各截图里出现的边框色（阈值 0.92）===")
for n in sorted(agg):
    print(f"  {n:11s} {dict(agg[n])}")
