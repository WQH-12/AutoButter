import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(p)[:-4]: cv2.imread(p, cv2.IMREAD_COLOR) for p in glob.glob(USER + r"\*.png")}

print("=== template self-stats (std near 0 => flat, useless) ===")
for n, t in tpls.items():
    g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    print(f"{n:11s} {t.shape[1]}x{t.shape[0]}  mean={g.mean():6.1f} std={g.std():5.1f}  "
          f"uniq={len(np.unique(g)):3d}")

# 只测这两张“物品齐全”的列表页，速度快
targets = {
    "45_tanxun_list": cv2.imread(r"E:\work\AutoButter\screenshots\45_tanxun_list.png"),
    "49_list2":       cv2.imread(r"E:\work\AutoButter\screenshots\49_list2.png"),
}
print("\n=== match against item-list screens ===")
for sname, img in targets.items():
    print(f"\n-- {sname} {img.shape[1]}x{img.shape[0]} --")
    for n, t in tpls.items():
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        th, tw = t.shape[:2]
        print(f"  {n:11s} {mx:.3f}  at ({ml[0]:3d},{ml[1]:3d}) center=({ml[0]+tw//2:3d},{ml[1]+th//2:3d})")
