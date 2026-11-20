import cv2, numpy as np, glob, os

SNAP = r"E:\work\AutoButter\screenshots\snap"
img = cv2.imread(os.path.join(SNAP, "20261108-130821_720.png"))

# 已知格子在 130821 里的位置与身份
truth = {
    "alcohol": (163,568,87,91),
    "jade":    (367,688,87,92),
    "honey":   (460,683,95,93),
    "berry":   (570,708,89,91),
    "berry2":  (271,756,83,86),
}

variants = [("a",8,8,24), ("b",10,10,30), ("c",12,12,34), ("d",12,14,38)]

negatives = ["01_current.png", "23_verify_now.png", "43_manage_now.png", "53_rob_popup.png"]

print(f"{'var':4s} {'tpl':8s} {'size':>8s} {'self':>6s} {'cross-neg':>10s}")
print("-"*46)
for tag, ix, itop, dbot in variants:
    worst_neg = 0
    for name in ["alcohol", "jade", "honey", "berry"]:
        x,y,w,h = truth[name]
        t = img[y+itop:y+h-dbot, x+ix:x+w-ix]
        # self
        s_self = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED).max()
        # 负样本最高分
        neg = 0
        for f in negatives:
            im = cv2.imread(rf"E:\work\AutoButter\screenshots\{f}")
            if im is None: continue
            neg = max(neg, cv2.matchTemplate(im, t, cv2.TM_CCOEFF_NORMED).max())
        worst_neg = max(worst_neg, neg)
        print(f"{tag:4s} {name:8s} {t.shape[1]:3d}x{t.shape[0]:<3d} {s_self:6.3f} {neg:10.3f}")
    print(f"     -> worst negative across items: {worst_neg:.3f}")
    print()
