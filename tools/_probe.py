import cv2, glob, os
for f in sorted(glob.glob(r"E:\work\AutoButter\resource\image\*.png")):
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    print(os.path.basename(f), img.shape if img is not None else "FAIL")
for f in [r"E:\work\AutoButter\screenshots\23_verify_now.png"]:
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    print(os.path.basename(f), img.shape)
