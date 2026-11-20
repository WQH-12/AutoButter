import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

def is_swapped(img):
    roi = img[600:900, 60:660]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:,:,0]>=95)&(hsv[:,:,0]<=115)&(hsv[:,:,1]>60)).mean()
    orange = (((hsv[:,:,0]<=25)|(hsv[:,:,0]>=170))&(hsv[:,:,1]>60)).mean()
    return blue > orange

# 只看干净的 snap 截图
print("=== 三张 snap 截图的完整识别结果 ===")
for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\snap\*_720.png")):
    img = cv2.imread(p)
    name = os.path.basename(p)
    if is_swapped(img):
        print(f"\n{name}: CORRUPTED (channel swap), skip"); continue
    print(f"\n{name}  ({img.shape[1]}x{img.shape[0]})")
    for n, t in tpls.items():
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        rr = res.copy(); pts = []
        while True:
            _, mx, _, ml = cv2.minMaxLoc(rr)
            if mx < 0.85: break
            pts.append((round(float(mx),3), (ml[0]+t.shape[1]//2, ml[1]+t.shape[0]//2)))
            rr[max(0,ml[1]-20):ml[1]+20, max(0,ml[0]-20):ml[0]+20] = -1
        if pts:
            print(f"  {n:11s} " + "  ".join(f"{s:.3f}@({c[0]},{c[1]})" for s,c in pts))
