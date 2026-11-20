import cv2, numpy as np, os, glob

tpl_dir = r"E:\work\AutoButter\resource\image"
tpls = {os.path.basename(p): cv2.imread(p, cv2.IMREAD_COLOR) for p in glob.glob(tpl_dir + r"\*.png")}
tpls = {k: v for k, v in tpls.items() if v is not None}

def to720(img):
    h, w = img.shape[:2]
    s = 720.0 / min(h, w)
    return cv2.resize(img, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA)

rois = {"nav_home.png": [0,1400,720,160], "nav_adventure.png": [0,1400,720,160], "home_menu_first.png": [0,1000,400,400]}

for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None: continue
    img = to720(img)
    line = [os.path.basename(p), f"{img.shape[1]}x{img.shape[0]}"]
    for name, tpl in tpls.items():
        roi = rois.get(name)
        if roi and (img.shape[1] < roi[0]+roi[2] or img.shape[0] < roi[1]+roi[3]):
            continue
        if roi:
            x,y,w_,h_ = roi
            search = img[y:y+h_, x:x+w_]
        else:
            search = img
        if search.shape[0] < tpl.shape[0] or search.shape[1] < tpl.shape[1]:
            continue
        r = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(r)
        line.append(f"{name.replace('.png','')}={mx:.3f}")
    print("  ".join(line))
