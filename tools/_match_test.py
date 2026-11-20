import cv2, numpy as np, os

shot = cv2.imread(r"E:\work\AutoButter\screenshots\23_verify_now.png", cv2.IMREAD_COLOR)
h, w = shot.shape[:2]
scale = 720.0 / min(h, w)
img720 = cv2.resize(shot, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
print("720 canvas:", img720.shape)
cv2.imwrite(r"E:\work\AutoButter\screenshots\24_verify_720.png", img720)

def match(name, roi=None, thr=0.8):
    tpl = cv2.imread(rf"E:\work\AutoButter\resource\image\{name}", cv2.IMREAD_COLOR)
    search = img720
    ox, oy = 0, 0
    if roi:
        x, y, rw, rh = roi
        search = img720[y:y+rh, x:x+rw]
        ox, oy = x, y
    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    minv, maxv, minl, maxl = cv2.minMaxLoc(res)
    th, tw = tpl.shape[:2]
    cx, cy = ox + maxl[0] + tw//2, oy + maxl[1] + th//2
    print(f"{name:24s} score={maxv:.4f} center=({cx},{cy}) box=({ox+maxl[0]},{oy+maxl[1]},{tw},{th}) hit={maxv>=thr}")

match("nav_home.png", [0,1400,720,160])
match("nav_adventure.png", [0,1400,720,160])
match("home_menu_first.png", [0,1000,400,400])
