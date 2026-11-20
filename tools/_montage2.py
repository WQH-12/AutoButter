import cv2, numpy as np, glob, os
OUT = r"E:\work\AutoButter\resource\image"
names = ["item_scroll.png","item_crystal.png","item_goji.png","item_jade.png","item_rice.png","item_silver.png"]
tiles=[]
for n in names:
    p=os.path.join(OUT,n)
    if not os.path.exists(p): print("missing",n); continue
    t=cv2.imread(p)
    t=cv2.resize(t,(120,120),interpolation=cv2.INTER_NEAREST)
    tiles.append((n,t))
canvas=np.full((150,len(tiles)*130,3),255,np.uint8)
for i,(n,t) in enumerate(tiles):
    canvas[5:125,i*130+5:i*130+125]=t
    cv2.putText(canvas,n.replace("item_","").replace(".png",""),(i*130+5,145),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)
cv2.imwrite(r"E:\work\AutoButter\screenshots\63_tpl_montage.png",canvas)
print(canvas.shape)
