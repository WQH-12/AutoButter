import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(p)[:-4]: cv2.imread(p, cv2.IMREAD_COLOR) for p in glob.glob(USER + r"\*.png")}
shots = {os.path.basename(p)[:-4]: cv2.imread(p) for p in glob.glob(r"E:\work\AutoButter\screenshots\*.png")}
shots = {k: v for k, v in shots.items() if v is not None and v.shape[0] >= 1000}

print("=== scale sweep ===")
for name in ["honey_1", "jade_1", "alcohol_1", "stone_1"]:
    tpl0 = tpls[name]
    best = (0.0, 0.0, "", None)
    for s in np.arange(0.60, 1.45, 0.05):
        t = cv2.resize(tpl0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if t.shape[0] < 12 or t.shape[1] < 12:
            continue
        for sname, img in shots.items():
            if img.shape[0] < t.shape[0] or img.shape[1] < t.shape[1]:
                continue
            res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, ml = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (float(mx), float(s), sname, ml)
    print(f"{name:10s} best={best[0]:.3f} scale={best[1]:.2f} in {best[2]} at {best[3]}")
