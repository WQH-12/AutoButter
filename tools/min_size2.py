"""Minimum icon size, done correctly.

For every screenshot we know exactly which items it contains (ground truth).
We downscale screenshot AND templates by the same factor, then check:
  self   -- score at the item's TRUE location (not global max)
  wrong  -- score of items that are NOT in this screenshot (true negatives)
An icon size is usable when min(self) > threshold > max(wrong).
"""
import cv2, numpy as np, os

ROOT = r"E:\work\AutoButter"
USER = os.path.join(ROOT, "User_screenshot")
tpls = {n: cv2.imread(os.path.join(USER, n + "_1.png"), cv2.IMREAD_COLOR)
        for n in ["alcohol", "berry", "fabric", "honey", "jade", "silver", "stone"]}

# (screenshot, present items, true centres of those items)
cases = [
    ("screenshots/snap/20261108-130821_720.png",
     {"alcohol": [(207,606)], "jade": [(411,727)], "honey": [(513,721)], "berry": [(615,746),(310,795)]}),
    ("screenshots/snap/20261108-131441_720.png",
     {"alcohol": [(207,738),(512,883)], "fabric": [(106,749)], "stone": [(411,733),(615,730)]}),
    ("screenshots/snap/20261108-132031_720.png",
     {"berry": [(615,742),(513,750),(310,739)], "silver": [(106,718)], "stone": [(411,722),(207,738)]}),
]

print(f"{'icon':>10s} {'min self':>9s} {'max wrong':>10s} {'margin':>8s}  verdict")
print("-"*54)
for w in [66, 56, 48, 40, 36, 32, 28, 24, 20, 16]:
    s = w / 66.0
    h = max(8, round(47 * s))
    self_s, wrong_s = [], []
    for rel, present in cases:
        img = cv2.imread(os.path.join(ROOT, rel.replace("/", os.sep)))
        img_s = cv2.resize(img, (round(img.shape[1]*s), round(img.shape[0]*s)), interpolation=cv2.INTER_AREA)
        # scale the true centres too
        for name, t0 in tpls.items():
            t = cv2.resize(t0, (w, h), interpolation=cv2.INTER_AREA)
            if img_s.shape[0] < h or img_s.shape[1] < w: continue
            if name in present:
                for cx, cy in present[name]:
                    x, y = round(cx*s - w/2), round(cy*s - h/2)
                    x = max(0, min(x, img_s.shape[1]-w)); y = max(0, min(y, img_s.shape[0]-h))
                    patch = img_s[y:y+h, x:x+w]
                    if patch.shape[:2] == (h, w):
                        self_s.append(float(cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED)[0][0]))
            else:
                wrong_s.append(float(cv2.matchTemplate(img_s, t, cv2.TM_CCOEFF_NORMED).max()))
    if not self_s: continue
    mn, mx = min(self_s), max(wrong_s)
    v = "OK" if mn > mx + 0.10 else ("TIGHT" if mn > mx else "FAIL")
    print(f"{w:4d}x{h:<5d} {mn:9.3f} {mx:10.3f} {mn-mx:8.3f}  {v}")
