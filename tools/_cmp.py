import cv2, numpy as np

lst = cv2.imread(r"E:\work\AutoButter\screenshots\45_tanxun_list.png")
realm = cv2.imread(r"E:\work\AutoButter\screenshots\31_realm_popup.png")

# jade in list (matched 0.988) vs jade in realm (slot 0)
a = lst[489:489+47, 257:257+66]
b = realm[616:616+92, 61:61+89]
print("list  jade:", a.shape, "ratio", round(a.shape[1]/a.shape[0],2))
print("realm jade:", b.shape, "ratio", round(b.shape[1]/b.shape[0],2))

# 放大并排
H = 240
a2 = cv2.resize(a, (int(a.shape[1]*H/a.shape[0]), H), interpolation=cv2.INTER_NEAREST)
b2 = cv2.resize(b, (int(b.shape[1]*H/b.shape[0]), H), interpolation=cv2.INTER_NEAREST)
canvas = np.full((H+30, a2.shape[1]+b2.shape[1]+30, 3), 255, np.uint8)
canvas[0:H, 0:a2.shape[1]] = a2
canvas[0:H, a2.shape[1]+30:a2.shape[1]+30+b2.shape[1]] = b2
cv2.putText(canvas, f"LIST {a.shape[1]}x{a.shape[0]}", (2, H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
cv2.putText(canvas, f"REALM {b.shape[1]}x{b.shape[0]}", (a2.shape[1]+32, H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
cv2.imwrite(r"E:\work\AutoButter\screenshots\73_jade_compare.png", canvas)
print("saved", canvas.shape)
