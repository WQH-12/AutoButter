import cv2, numpy as np, glob, os
for p in sorted(glob.glob(r"E:\work\AutoButter\User_screenshot\*.png")):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    a = im[:,:,3]
    print(f"{os.path.basename(p):15s} alpha min={a.min():3d} max={a.max():3d} "
          f"transparent%={100.0*(a<128).mean():5.1f}  "
          f"opaque%={100.0*(a>=250).mean():5.1f}")
