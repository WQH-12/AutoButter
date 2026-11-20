import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

# 误报检查：不含任何物品格子的界面
negatives = ["01_current.png", "23_verify_now.png", "43_manage_now.png", "53_rob_popup.png"]
print("=== 误报检查（这些界面不该有物品格）===")
for f in negatives:
    p = rf"E:\work\AutoButter\screenshots\{f}"
    img = cv2.imread(p)
    if img is None: continue
    out = []
    for n, t in tpls.items():
        mx = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED).max()
        if mx >= 0.85: out.append(f"{n}={mx:.2f}")
    print(f"  {f:22s} {'FALSE POSITIVE: ' + ', '.join(out) if out else 'clean'}")

# 每张模板在自己的来源图上必须 1.000（自检）
print("\n=== 自检：模板在自己的图里 ===")
src_map = {
    "alcohol_1": "snap/20261108-130821_720.png",
    "honey_1":   "snap/20261108-130821_720.png",
    "jade_1":    "snap/20261108-130821_720.png",
    "berry_1":   "snap/20261108-130821_720.png",
}
for n, rel in src_map.items():
    img = cv2.imread(os.path.join(r"E:\work\AutoButter\screenshots", rel.replace("/", "\\")))
    mx = cv2.matchTemplate(img, tpls[n], cv2.TM_CCOEFF_NORMED).max()
    print(f"  {n:11s} {mx:.4f}")
