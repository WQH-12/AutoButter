import cv2, numpy as np
a = cv2.imread(r"E:\work\AutoButter\screenshots\37_color_check.png")
b = cv2.imread(r"E:\work\AutoButter\screenshots\37_color_adb.png")
b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
for name,(y,x) in {"bg_top":(60,600), "skin":(500,120), "text_area":(1250,400)}.items():
    print(name, "maa", a[y,x], "adb", b[y,x])
print("mean abs diff:", float(np.abs(a.astype(int)-b.astype(int)).mean()))
