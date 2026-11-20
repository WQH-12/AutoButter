"""Minimum icon size -- discrimination at the true location.

For each screenshot and each icon width:
  for every item present, cut the template out of the shrunken screen at its
  true centre, then score that template at EVERY item's true centre.
  diagonal  = score at its own slot   (should be high)
  offdiag   = best score at someone else's slot (should be low)
"""
import cv2, numpy as np, os

ROOT = r"E:\work\AutoButter"
cases = [
    ("screenshots/snap/20261108-130821_720.png",
     {"alcohol": [(207,606)], "jade": [(411,727)], "honey": [(513,721)], "berry": [(615,746)]}),
    ("screenshots/snap/20261108-131441_720.png",
     {"alcohol": [(207,738)], "fabric": [(106,749)], "stone": [(411,733)]}),
    ("screenshots/snap/20261108-132031_720.png",
     {"berry": [(615,742)], "silver": [(106,718)], "stone": [(411,722)]}),
    ("screenshots/42_after_close.png",
     {"alcohol": [(309,582)], "fabric": [(513,750)]}),
]
W0, H0 = 66, 47

print(f"{'icon':>10s} {'min diag':>9s} {'max off':>8s} {'margin':>8s}  verdict")
print("-"*50)
for w in [66, 56, 48, 40, 36, 32, 28, 24, 20, 16]:
    s = w / W0
    h = max(8, round(H0 * s))
    diag, off = [], []
    for rel, present in cases:
        img = cv2.imread(os.path.join(ROOT, rel.replace("/", os.sep)))
        if img is None: continue
        small = cv2.resize(img, (round(img.shape[1]*s), round(img.shape[0]*s)), interpolation=cv2.INTER_AREA)
        slots = {}
        for name, centres in present.items():
            cx, cy = centres[0]
            x, y = round(cx*s - w/2), round(cy*s - h/2)
            x = max(0, min(x, small.shape[1]-w)); y = max(0, min(y, small.shape[0]-h))
            slots[name] = small[y:y+h, x:x+w]
        for src_name, t in slots.items():
            if t.shape[:2] != (h, w) or t.std() < 4: continue
            for dst_name, patch in slots.items():
                if patch.shape[:2] != (h, w): continue
                sc = float(cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED)[0][0])
                (diag if src_name == dst_name else off).append(sc)
    if not diag or not off: continue
    mn, mx = min(diag), max(off)
    v = "OK" if mn > mx + 0.10 else ("TIGHT" if mn > mx else "FAIL")
    print(f"{w:4d}x{h:<5d} {mn:9.3f} {mx:8.3f} {mn-mx:8.3f}  {v}")
