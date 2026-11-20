import cv2, numpy as np
img = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png")
# honey 检测框
x,y,w,h = 460,683,95,93
patch = img[y:y+h, x:x+w]
print("box", x,y,w,h)
for label, strip in [
    ("top 0:3",   patch[0:3, w//2-10:w//2+10]),
    ("top 2:6",   patch[2:6, w//2-10:w//2+10]),
    ("bottom",    patch[h-4:h, w//2-10:w//2+10]),
    ("left",      patch[h//2-10:h//2+10, 0:4]),
    ("right",     patch[h//2-10:h//2+10, w-4:w]),
]:
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    hh,ss,vv = [int(np.median(hsv[:,:,i])) for i in range(3)]
    print(f"  {label:9s} H={hh:3d} S={ss:3d} V={vv:3d}  BGR={strip.reshape(-1,3).mean(axis=0).round(0)}")
