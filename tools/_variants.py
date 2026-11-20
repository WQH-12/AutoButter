import cv2, numpy as np

img = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png")
cells = {"alcohol": (163,568,87,91), "honey": (460,683,95,93), "jade": (367,688,87,92)}

variants = [("a", 8, 8, 24), ("b", 10, 10, 30), ("c", 12, 12, 34), ("d", 12, 14, 38)]
rows = []
for tag, ix, itop, dbot in variants:
    tiles = []
    for name, (x,y,w,h) in cells.items():
        p = img[y+itop:y+h-dbot, x+ix:x+w-ix]
        if p.size == 0: continue
        t = cv2.resize(p, (150, 110), interpolation=cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(t, 18, 0, 0, 6, cv2.BORDER_CONSTANT, value=(255,255,255))
        cv2.putText(t, f"{name} {p.shape[1]}x{p.shape[0]}", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1)
        tiles.append(t)
    row = np.hstack(tiles)
    row = cv2.copyMakeBorder(row, 20, 4, 4, 4, cv2.BORDER_CONSTANT, value=(230,230,230))
    cv2.putText(row, f"{tag}: inset_x={ix} top={itop} drop_bottom={dbot}", (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
    rows.append(row)

W = max(r.shape[1] for r in rows)
rows = [cv2.copyMakeBorder(r, 0, 0, 0, W-r.shape[1], cv2.BORDER_CONSTANT, value=(230,230,230)) for r in rows]
out = np.vstack(rows)
cv2.imwrite(r"E:\work\AutoButter\screenshots\81_inset_variants.png", out)
print("saved", out.shape)
