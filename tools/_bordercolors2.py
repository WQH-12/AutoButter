import cv2, numpy as np, glob, os

p = r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png"
img = cv2.imread(p)
cells = [(163,568,87,91,"alcohol"),(367,688,87,92,"jade"),(460,683,95,93,"honey"),(570,708,89,91,"berry"),(271,756,83,86,"berry2")]

print("=== 新截图里每格的真实边框颜色（BGR / HSV）===")
for x,y,cw,ch,name in cells:
    # 上边框中间条带
    strip = img[y+2:y+6, x+cw//2-12:x+cw//2+12]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    h,s,v = [int(np.median(hsv[:,:,i])) for i in range(3)]
    b,g,r = [int(np.median(strip[:,:,i])) for i in range(3)]
    print(f"  {name:8s} box=({x:3d},{y:3d})  BGR=({b:3d},{g:3d},{r:3d})  H={h:3d} S={s:3d} V={v:3d}")

print()
print("=== 对照：其它截图同物品的边框色 ===")
ref = {
    "42_after_close.png": [(62,681,86,91,"fabric"),(265,544,89,92,"alcohol")],
    "31_realm_popup.png": [(61,616,89,92,"alcohol"),(367,696,89,92,"fabric")],
}
for f, lst in ref.items():
    im = cv2.imread(rf"E:\work\AutoButter\screenshots\{f}")
    for x,y,cw,ch,name in lst:
        strip = im[y+2:y+6, x+cw//2-12:x+cw//2+12]
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        h,s,v = [int(np.median(hsv[:,:,i])) for i in range(3)]
        b,g,r = [int(np.median(strip[:,:,i])) for i in range(3)]
        print(f"  {f[:18]:18s} {name:8s} BGR=({b:3d},{g:3d},{r:3d}) H={h:3d} S={s:3d} V={v:3d}")
