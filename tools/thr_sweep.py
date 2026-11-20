import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

def is_swapped(img):
    roi = img[int(img.shape[0]*.38):int(img.shape[0]*.58), 60:660]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:,:,0]>=95)&(hsv[:,:,0]<=115)&(hsv[:,:,1]>60)).mean()
    warm = (((hsv[:,:,0]<=25)|(hsv[:,:,0]>=170))&(hsv[:,:,1]>60)).mean()
    return blue > warm

# 真值：只统计秘境主界面（我们的目标界面）
truth = {
    "snap/20261108-130821_720.png": {"alcohol":1,"jade":1,"honey":1,"berry":2},
    "snap/20261108-131441_720.png": {"alcohol":2,"fabric":1,"stone":2},
    "snap/20261108-132031_720.png": {"berry":3,"silver":1,"stone":2},
    "42_after_close.png":           {"alcohol":1,"fabric":4},
    "44_realm.png":                 {"fabric":4,"stone":1},
    "48_go_target.png":             {"stone":2,"berry":2,"fabric":2},
    "52_go_row2.png":               {"silver":1,"fabric":1,"stone":1,"berry":3},
}
negatives = ["01_current.png", "23_verify_now.png", "53_rob_popup.png"]

for THR in [0.86, 0.88, 0.90, 0.92]:
    ok = tot = 0
    over = 0
    for rel, expect in truth.items():
        p = os.path.join(r"E:\work\AutoButter\screenshots", rel.replace("/", os.sep))
        img = cv2.imread(p)
        if img is None or is_swapped(img): continue
        for item, want in expect.items():
            t = tpls[item + "_1"]
            res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
            rr = res.copy(); got = 0
            while True:
                _, mx, _, ml = cv2.minMaxLoc(rr)
                if mx < THR: break
                got += 1
                rr[max(0,ml[1]-18):ml[1]+18, max(0,ml[0]-18):ml[0]+18] = -1
            tot += 1; ok += (got >= want)
    fp = 0
    for f in negatives:
        img = cv2.imread(rf"E:\work\AutoButter\screenshots\{f}")
        if img is None: continue
        for n, t in tpls.items():
            res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
            rr = res.copy()
            while True:
                _, mx, _, ml = cv2.minMaxLoc(rr)
                if mx < THR: break
                fp += 1
                rr[max(0,ml[1]-18):ml[1]+18, max(0,ml[0]-18):ml[0]+18] = -1
    print(f"THR={THR:.2f}  recall={ok}/{tot} ({100*ok//tot}%)  false_positives={fp}")
