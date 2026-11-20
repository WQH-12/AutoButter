import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

def is_swapped(img):
    roi = img[int(img.shape[0]*.38):int(img.shape[0]*.58), 60:660]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:,:,0]>=95)&(hsv[:,:,0]<=115)&(hsv[:,:,1]>60)).mean()
    warm = (((hsv[:,:,0]<=25)|(hsv[:,:,0]>=170))&(hsv[:,:,1]>60)).mean()
    return blue > warm

t = tpls["berry_1"]
print("=== 所有 berry_1 命中 >=0.92 的实例：位置 / 边框色 / 分数 ===")
for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")) + sorted(glob.glob(r"E:\work\AutoButter\screenshots\snap\*_720.png")):
    b = os.path.basename(p)
    if b[:3] in ("60_","61_","62_","64_","65_","66_","70_","71_","72_","73_","74_","75_","76_","77_","78_","79_","80_","81_","82_","83_","84_","85_"): continue
    im = cv2.imread(p)
    if im is None or im.shape[0] < 900: continue
    sw = is_swapped(im)
    if sw: continue
    res = cv2.matchTemplate(im, t, cv2.TM_CCOEFF_NORMED)
    rr = res.copy()
    while True:
        _, mx, _, ml = cv2.minMaxLoc(rr)
        if mx < 0.92: break
        strip = im[ml[1]:ml[1]+3, ml[0]+t.shape[1]//3:ml[0]+2*t.shape[1]//3]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        h = int(np.median(hsv[:,:,0])); s = int(np.median(hsv[:,:,1]))
        col = "gray" if s < 55 else ("ORANGE" if (h<22 or h>170) else ("green" if 40<=h<92 else ("BLUE" if 92<=h<=112 else "PURPLE")))
        # 读等级文字区域（格子右下角）亮度特征
        print(f"  {b[:26]:26s} {mx:.3f} @({ml[0]+33:3d},{ml[1]+23:4d})  frame={col:6s} H={h:3d} S={s:3d}")
        rr[max(0,ml[1]-18):ml[1]+18, max(0,ml[0]-18):ml[0]+18] = -1
