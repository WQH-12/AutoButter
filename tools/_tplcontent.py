import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

print("=== 模板里是否含等级文字（右下角 20x14 区域是否有高对比白/黑像素）===")
for n, t in tpls.items():
    h, w = t.shape[:2]
    corner = t[h-16:h-2, w-22:w-2]
    g = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    # 等级文字是白字带黑描边，局部方差很高
    print(f"  {n:11s} corner std={g.std():5.1f} min={g.min():3d} max={g.max():3d} "
          f"{'<-- 可能含等级文字' if g.std() > 45 else ''}")

print("\n=== 模板里边框占比（上/下/左/右各 4px）===")
for n, t in tpls.items():
    h, w = t.shape[:2]
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]
    border = np.zeros((h,w), bool)
    border[:4,:] = border[-4:,:] = border[:,:4] = border[:,-4:] = True
    # 边框是纯色高饱和，图标内部也有高饱和，用「方差低+饱和高」判
    print(f"  {n:11s} border S median={np.median(S[border]):5.1f}  inner S median={np.median(S[~border]):5.1f}")
