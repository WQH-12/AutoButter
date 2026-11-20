"""Minimum recognisable icon size.

Downscales BOTH the screenshot and the template by the same factor, so the
icon is genuinely rendered smaller -- this answers "how small can an icon be
before matching breaks", not "what happens if template and screen disagree
on scale".

Reports, per target icon width:
  self    -- score of the item in its own downscaled screenshot
  cross   -- highest score of the OTHER items in the same downscaled screen
  margin  -- self - cross  (needs to stay positive and wide)
"""
import cv2, numpy as np, glob, os

ROOT = r"E:\work\AutoButter"
USER = os.path.join(ROOT, "User_screenshot")
srcs = {
    "alcohol": "screenshots/snap/20261108-131441_720.png",
    "berry":   "screenshots/snap/20261108-130821_720.png",
    "fabric":  "screenshots/snap/20261108-131441_720.png",
    "honey":   "screenshots/snap/20261108-130821_720.png",
    "jade":    "screenshots/snap/20261108-130821_720.png",
    "silver":  "screenshots/snap/20261108-132031_720.png",
    "stone":   "screenshots/snap/20261108-131441_720.png",
}

tpls = {n: cv2.imread(os.path.join(USER, n + "_1.png"), cv2.IMREAD_COLOR) for n in srcs}
imgs = {n: cv2.imread(os.path.join(ROOT, rel.replace("/", os.sep)), cv2.IMREAD_COLOR) for n, rel in srcs.items()}

widths = [66, 56, 48, 40, 36, 32, 28, 24, 20]
print(f"{'icon_w':>7s} {'icon_h':>7s} {'min_self':>9s} {'max_cross':>10s} {'margin':>8s}  verdict")
print("-" * 62)
for w in widths:
    s = w / 66.0
    h = round(47 * s)
    self_scores, cross_scores = [], []
    for name, t0 in tpls.items():
        img = imgs[name]
        img_s = cv2.resize(img, (round(img.shape[1]*s), round(img.shape[0]*s)), interpolation=cv2.INTER_AREA)
        for other, o0 in tpls.items():
            t = cv2.resize(o0, (w, h), interpolation=cv2.INTER_AREA)
            if img_s.shape[0] < h or img_s.shape[1] < w:
                continue
            sc = cv2.matchTemplate(img_s, t, cv2.TM_CCOEFF_NORMED).max()
            (self_scores if other == name else cross_scores).append(float(sc))
    if not self_scores:
        continue
    mn, mx = min(self_scores), max(cross_scores) if cross_scores else 0.0
    verdict = "OK" if mn > mx + 0.15 else ("TIGHT" if mn > mx else "FAIL")
    print(f"{w:7d} {h:7d} {mn:9.3f} {mx:10.3f} {mn-mx:8.3f}  {verdict}")
