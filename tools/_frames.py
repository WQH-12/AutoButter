import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

def frame_of(patch, t):
    # 用模板尺寸取样：模板上边框内侧 2px 条带
    x, y = patch
    return None

# 对每张干净 snap，找到每个物品，然后读它所在格子的边框色 + 等级文字区域亮度
snaps = sorted(glob.glob(r"E:\work\AutoButter\screenshots\snap\*_720.png"))
print(f"{'snap':22s} {'item':9s} {'score':>6s} {'center':>12s} {'frameHSV':>16s}")
print("-"*72)
for p in snaps:
    img = cv2.imread(p)
    if img is None: continue
    name = os.path.basename(p).replace("_720.png","")
    for n, t in tpls.items():
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        rr = res.copy()
        while True:
            _, mx, _, ml = cv2.minMaxLoc(rr)
            if mx < 0.92: break
            # 上边框条带（模板 y 往下 2-5px）
            strip = img[ml[1]+1:ml[1]+5, ml[0]+t.shape[1]//2-10:ml[0]+t.shape[1]//2+10]
            hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
            h = int(np.median(hsv[:,:,0])); s = int(np.median(hsv[:,:,1])); v = int(np.median(hsv[:,:,2]))
            col = "gray" if s < 55 else ("orange" if (h<22 or h>170) else ("green" if 40<=h<92 else ("blue" if 92<=h<=112 else "purple")))
            cx, cy = ml[0]+t.shape[1]//2, ml[1]+t.shape[0]//2
            print(f"{name:22s} {n:9s} {mx:6.3f} ({cx:4d},{cy:4d}) {col:7s} H={h:3d} S={s:3d}")
            rr[max(0,ml[1]-20):ml[1]+20, max(0,ml[0]-20):ml[0]+20] = -1
