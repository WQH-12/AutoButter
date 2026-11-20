import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

def is_swapped(img):
    roi = img[int(img.shape[0]*.38):int(img.shape[0]*.58), 60:660]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:,:,0]>=95)&(hsv[:,:,0]<=115)&(hsv[:,:,1]>60)).mean()
    warm = (((hsv[:,:,0]<=25)|(hsv[:,:,0]>=170))&(hsv[:,:,1]>60)).mean()
    return blue > warm

SKIP = ("60_","61_","62_","64_","65_","66_","70_","71_","72_","73_","74_","75_",
        "76_","77_","78_","79_","80_","81_","82_","83_","84_","85_","86_","_s31","_t_","v_","t_")

# 干净截图 + 已知答案
truth = {
    "snap/20261108-130821_720.png": {"alcohol":1,"jade":1,"honey":1,"berry":2},
    "snap/20261108-131441_720.png": {"alcohol":2,"fabric":1,"stone":2},
    "snap/20261108-132031_720.png": {"berry":3,"silver":1,"stone":2},
    "42_after_close.png":           {"alcohol":1,"fabric":4},
    "44_realm.png":                 {"fabric":4,"stone":1},
    "48_go_target.png":             {"stone":2,"berry":2,"fabric":2},
    "52_go_row2.png":               {"silver":1,"fabric":1,"stone":1,"berry":3},
    "45_tanxun_list.png":           {"alcohol":1,"berry":1,"fabric":1,"jade":1,"silver":1,"stone":1},
    "49_list2.png":                 {"alcohol":1,"berry":1,"fabric":1,"jade":1,"silver":1,"stone":1},
}

THR = 0.90
print(f"threshold = {THR}")
print(f"{'screenshot':28s} {'item':9s} {'want':>4s} {'got':>4s} {'best':>6s}  result")
print("-"*68)
total_ok = total = 0
for rel, expect in truth.items():
    p = os.path.join(r"E:\work\AutoButter\screenshots", rel.replace("/", os.sep))
    img = cv2.imread(p)
    if img is None: continue
    if is_swapped(img):
        print(f"{rel:28s} SKIPPED (channel-swapped)"); continue
    for item, want in expect.items():
        t = tpls[item + "_1"]
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        rr = res.copy(); got = 0; best = 0.0
        while True:
            _, mx, _, ml = cv2.minMaxLoc(rr)
            if mx < THR: break
            got += 1; best = max(best, mx)
            rr[max(0,ml[1]-18):ml[1]+18, max(0,ml[0]-18):ml[0]+18] = -1
        ok = got >= want
        total += 1; total_ok += ok
        print(f"{rel[:28]:28s} {item:9s} {want:4d} {got:4d} {best:6.3f}  {'OK' if ok else 'MISS'}")

print("-"*68)
print(f"pass {total_ok}/{total}")
