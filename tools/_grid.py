import cv2, numpy as np

def grid(src_path, out_path, x0, y0, x1, y1, step=50, scale=2):
    img = cv2.imread(src_path)
    crop = img[y0:y1, x0:x1]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for gx in range(0, x1-x0, step):
        px = gx*scale
        cv2.line(big, (px,0), (px,big.shape[0]), (0,0,255), 1)
        cv2.putText(big, str(x0+gx), (px+2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    for gy in range(0, y1-y0, step):
        py = gy*scale
        cv2.line(big, (0,py), (big.shape[1],py), (255,0,0), 1)
        cv2.putText(big, str(y0+gy), (2, py+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    cv2.imwrite(out_path, big)
    print(out_path, big.shape)

grid(r"E:\work\AutoButter\screenshots\44_realm.png", r"E:\work\AutoButter\screenshots\58_grid_realm.png", 0, 550, 720, 800, 50, 2)
