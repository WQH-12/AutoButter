import cv2, numpy as np

# 验证假设：用户模板是「探寻列表」里的小图标(66x46)，不是秘境里的大格子
lst = cv2.imread(r"E:\work\AutoButter\screenshots\45_tanxun_list.png")
realm = cv2.imread(r"E:\work\AutoButter\screenshots\44_realm.png")

print("list icon box test (jade_1 matched at 257,489 size 66x47):")
crop = lst[489:489+47, 257:257+66]
print("  list crop shape", crop.shape)

# 列表图标间距
print("\nlist row1 icon x positions (from earlier match):")
for name, x in [("jade",257),("alcohol",564)]:
    print(f"  {name} at x={x}")

# 秘境格子 vs 列表图标 尺寸对比
print("\nrealm cell: 89x92 (outer), inner ~71x60")
print("list  icon: 66x47")

# 关键：把用户模板按比例放大到秘境尺寸，看能否匹配
tpl = cv2.imread(r"E:\work\AutoButter\User_screenshot\jade_1.png")
for s in [1.0, 1.2, 1.3, 1.4, 1.5, 1.6]:
    t = cv2.resize(tpl, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    res = cv2.matchTemplate(realm, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(res)
    print(f"  jade scale={s:.1f} size={t.shape[1]}x{t.shape[0]} score={mx:.3f} at {ml}")
