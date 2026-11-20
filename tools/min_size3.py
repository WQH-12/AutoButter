"""Minimum icon size -- honest simulation.

The earlier attempt downscaled template and screenshot separately, which
double-blurs the template and gives fake failures. Correct approach: shrink
the whole screenshot, then CUT the template out of that shrunken screenshot.
That is exactly what a natively-smaller icon would look like.

self  = the item found at its own (scaled) location
wrong = other items' best score in the same shrunken screen
"""
import cv2, numpy as np, os

ROOT = r"E:\work\AutoButter"
USER = os.path.join(ROOT, "User_screenshot")

# (screenshot, {item: [true centres]})
cases = [
    ("screenshots/snap/20261108-130821_720.png",
     {"alcohol": [(207,606)], "jade": [(411,727)], "honey": [(513,721)], "berry": [(615,746),(310,795)]}),
    ("screenshots/snap/20261108-131441_720.png",
     {"alcohol": [(207,738),(512,883)], "fabric": [(106,749)], "stone": [(411,733),(615,730)]}),
    ("screenshots/snap/20261108-132031_720.png",
     {"berry": [(615,742),(513,750),(310,739)], "silver": [(106,718)], "stone": [(411,722),(207,738)]}),
]
ALL = ["alcohol","berry","fabric","honey","jade","silver","stone"]
W0, H0 = 66, 47

print(f"{'icon':>10s} {'min self':>9s} {'max wrong':>10s} {'margin':>8s}  verdict")
print("-"*54)
for w in [66, 56, 48, 40, 36, 32, 28, 24, 20]:
    s = w / W0
    h = max(8, round(H0 * s))
    self_s, wrong_s = [], []
    for rel, present in cases:
        img = cv2.imread(os.path.join(ROOT, rel.replace("/", os.sep)))
        small = cv2.resize(img, (round(img.shape[1]*s), round(img.shape[0]*s)), interpolation=cv2.INTER_AREA)
        # cut each template from the SHRUNKEN image at its true centre
        cut = {}
        for name, centres in present.items():
            cx, cy = centres[0]
            x, y = round(cx*s - w/2), round(cy*s - h/2)
            x = max(0, min(x, small.shape[1]-w)); y = max(0, min(y, small.shape[0]-h))
            t = small[y:y+h, x:x+w]
            if t.shape[:2] == (h, w):
                cut[name] = t
                self_s.append(1.0 if t.std() > 5 else 0.0)   # sanity
        for name, t in cut.items():
            # does it find itself? (search a window around the truth)
            for cx, cy in present[name]:
                x, y = round(cx*s - w/2), round(cy*s - h/2)
                x0, y0 = max(0,x-6), max(0,y-6)
                win = small[y0:y0+h+12, x0:x0+w+12]
                if win.shape[0] < h or win.shape[1] < w: continue
                self_s.append(float(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max()))
            # does it fire on items that are absent?
            for other in ALL:
                if other in present: continue
                wrong_s.append(float(cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED).max()))
    if not self_s: continue
    mn, mx = min(self_s), max(wrong_s) if wrong_s else 0.0
    v = "OK" if mn > mx + 0.10 else ("TIGHT" if mn > mx else "FAIL")
    print(f"{w:4d}x{h:<5d} {mn:9.3f} {mx:10.3f} {mn-mx:8.3f}  {v}")
