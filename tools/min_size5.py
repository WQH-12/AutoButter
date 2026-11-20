"""Cross-instance size test: template from screenshot A, searched in B.

That is the real question -- not "can a template find itself", but "can a
template cut at size N still find the same item in a DIFFERENT screenshot".
"""
import cv2, numpy as np, os, itertools

ROOT = r"E:\work\AutoButter"
# item -> {screenshot: centre}
data = {
    "alcohol": {"snap/20261108-130821_720.png": (207,606), "snap/20261108-131441_720.png": (207,738), "42_after_close.png": (309,582)},
    "berry":   {"snap/20261108-130821_720.png": (615,746), "snap/20261108-132031_720.png": (615,742), "52_go_row2.png": (513,717)},
    "fabric":  {"snap/20261108-131441_720.png": (106,749), "42_after_close.png": (513,750), "48_go_target.png": (513,750)},
    "honey":   {"snap/20261108-130821_720.png": (513,721)},
    "jade":    {"snap/20261108-130821_720.png": (411,727), "45_tanxun_list.png": (290,512)},
    "silver":  {"snap/20261108-132031_720.png": (106,718), "52_go_row2.png": (106,747)},
    "stone":   {"snap/20261108-131441_720.png": (411,733), "42_after_close.png": (207,718)},
}
cache = {}
def load(rel):
    if rel not in cache:
        cache[rel] = cv2.imread(os.path.join(ROOT, "screenshots", rel.replace("/", os.sep)))
    return cache[rel]

W0, H0 = 66, 47
print(f"{'icon':>10s} {'pairs':>5s} {'min':>6s} {'median':>7s} {'max':>6s}  {'all>0.88':>8s}")
print("-"*54)
for w in [66, 56, 48, 40, 36, 32, 28, 24, 20, 16]:
    s = w / W0
    h = max(8, round(H0*s))
    scores = []
    for item, shots in data.items():
        for a, b in itertools.permutations(shots.keys(), 2):
            ia, ib = load(a), load(b)
            if ia is None or ib is None: continue
            cx, cy = shots[a]
            x, y = round(cx*s - w/2), round(cy*s - h/2)
            t = cv2.resize(ia, (round(ia.shape[1]*s), round(ia.shape[0]*s)), interpolation=cv2.INTER_AREA)[y:y+h, x:x+w]
            if t.shape[:2] != (h, w) or t.std() < 4: continue
            small_b = cv2.resize(ib, (round(ib.shape[1]*s), round(ib.shape[0]*s)), interpolation=cv2.INTER_AREA)
            if small_b.shape[0] < h or small_b.shape[1] < w: continue
            # search only near the expected location +-25 px (scaled)
            bx, by = shots[b]
            ex, ey = round(bx*s - w/2), round(by*s - h/2)
            m = 25
            x0, y0 = max(0, ex-m), max(0, ey-m)
            win = small_b[y0:min(small_b.shape[0], ey+m+h), x0:min(small_b.shape[1], ex+m+w)]
            if win.shape[0] < h or win.shape[1] < w: continue
            scores.append(float(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max()))
    if not scores: continue
    arr = np.array(scores)
    print(f"{w:4d}x{h:<5d} {len(arr):5d} {arr.min():6.3f} {np.median(arr):7.3f} {arr.max():6.3f}  "
          f"{'yes' if arr.min()>0.88 else 'no':>8s}")
