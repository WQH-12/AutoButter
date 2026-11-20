"""鸡妖自定义识别（SIFT 多模板 + 抹除迭代 + 边界/尺寸过滤）。

为什么不能用原生 FeatureMatch：
  MaaFramework 5.6 的 JFeatureMatch 只有 template/roi/detector/order_by/
  count/index/green_mask/ratio，**没有内点阈值参数**。raw_detail 里每框的
  count 就是内点数，但无法设下限，因此噪声框（内点 1~4）无法在 pipeline 层滤掉。

判定口径（实测）：内点 >= 10，投影四边形完整落在 ROI 内（留 8px 边），
投影面积 / 模板面积在 0.4~3.0。
"""
from __future__ import annotations

import json
import os
import sys

try:
    import cv2
except ImportError:  # Android 的 agent core 不带 opencv，用 numpy/Pillow 子集
    import cv2lite as cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _resolve_tpl_dir():
    """找模板目录。

    工作区与打包目录都是 <root>/resource/image（agent/ 与 resource/ 同级）。
    找不到时返回候选路径（保持旧行为，让调用方能报出具体路径）。
    """
    candidates = [
        os.path.join(ROOT, "resource", "image"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]


TPL_DIR = _resolve_tpl_dir()

POSES = {
    "frontage": ["chicken_frontage_1.png", "chicken_frontage_2.png",
                 "chicken_frontage_3.png"],
    "back": ["chicken_back_1.png", "chicken_back_2.png", "chicken_back_3.png"],
}


def _imread_any(path):
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


class ChickenFinder:
    """不依赖 Maa 的纯算法实现，便于离线单测。"""

    def __init__(self, tpl_dir=TPL_DIR, min_inliers=10, ratio=0.75,
                 ransac=5.0, margin=8, area_lo=0.4, area_hi=3.0):
        self.min_inliers = min_inliers
        self.ratio = ratio
        self.ransac = ransac
        self.margin = margin
        self.area_lo = area_lo
        self.area_hi = area_hi
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.templates = []      # (name, pose, kp, des, corners, tpl_area)
        for pose, names in POSES.items():
            for n in names:
                p = os.path.join(tpl_dir, n)
                if not os.path.exists(p):
                    continue
                img = _imread_any(p)
                kp, des = self.sift.detectAndCompute(img, None)
                if des is None:
                    continue
                th, tw = img.shape[:2]
                corners = np.float32(
                    [[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
                self.templates.append((n, pose, kp, des, corners, float(tw * th)))

    @property
    def ready(self):
        return bool(self.templates)

    def find(self, image, roi, max_rounds=8):
        """返回 [(inliers, cx, cy, (x0,y0,x1,y1), tpl_name, pose)]。"""
        x0, y0, w, h = [int(v) for v in roi]
        crop = image[y0:y0 + h, x0:x0 + w]
        if crop.size == 0:
            return []
        crop = crop.copy()
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        if crop.dtype != np.uint8:
            crop = crop.astype(np.uint8)

        found = []
        for _ in range(max_rounds):
            kp_s, des_s = self.sift.detectAndCompute(crop, None)
            if des_s is None or len(kp_s) < 4:
                break
            round_hits = []
            for name, pose, kp_t, des_t, corners, tpl_area in self.templates:
                raw = self.bf.knnMatch(des_t, des_s, k=2)
                good = [m for m, nn in raw
                        if nn is not None and m.distance < self.ratio * nn.distance]
                if len(good) < 4:
                    continue
                src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac)
                if H is None or mask is None:
                    continue
                inl = int(mask.sum())
                if inl < self.min_inliers:
                    continue
                proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
                xs, ys = proj[:, 0], proj[:, 1]
                if not (xs.min() >= self.margin and ys.min() >= self.margin
                        and xs.max() <= w - self.margin
                        and ys.max() <= h - self.margin):
                    continue
                area = cv2.contourArea(proj.astype(np.float32)) / tpl_area
                if not (self.area_lo <= area <= self.area_hi):
                    continue
                round_hits.append((
                    inl, float(xs.mean()) + x0, float(ys.mean()) + y0,
                    (int(xs.min()) + x0, int(ys.min()) + y0,
                     int(xs.max()) + x0, int(ys.max()) + y0), name, pose))
            if not round_hits:
                break
            round_hits.sort(key=lambda t: -t[0])
            accepted = []
            for hit in round_hits:
                if all(_iou(hit[3], a[3]) <= 0.35 for a in accepted):
                    accepted.append(hit)
            if not accepted:
                break
            found.extend(accepted)
            for _, _, _, box, _, _ in accepted:
                xa = max(0, box[0] - x0 - 5)
                xb = min(crop.shape[1], box[2] - x0 + 5)
                ya = max(0, box[1] - y0 - 5)
                yb = min(crop.shape[0], box[3] - y0 + 5)
                crop[ya:yb, xa:xb] = 0
        return found


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    a1 = (a[2] - a[0]) * (a[3] - a[1])
    a2 = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(max(1, a1 + a2 - inter))



_CHICKEN_TPL = [n for names in POSES.values() for n in names]
_TPL_W, _TPL_H = 102, 93


def _feature_hits(context, image, roi, min_inliers):
    """Android 上没有 cv2.SIFT，改用 pipeline 的 FeatureMatch。

    MaaFramework 的 FeatureMatcher 内部就是 SIFT + RANSAC + 抹除迭代，
    一次调用会返回同一模板的多个实例（源码 FeatureMatcher.cpp 的
    `while (scene.size() >= 4)` + `remove inside points`），正好对上
    原实现的 max_rounds 语义。box 是 ROI 裁剪后的绝对坐标。
    """
    node = "__chicken_feature"
    detail = context.run_recognition(node, image, {
        node: {
            "recognition": "FeatureMatch",
            "template": list(_CHICKEN_TPL),
            "roi": [int(v) for v in roi],
            "count": int(min_inliers),
            "detector": "SIFT",
            "ratio": 0.75,
            "order_by": "Score",
        }
    })
    if detail is None or not detail.hit:
        return []

    raw = list(detail.all_results or [])
    if not raw and detail.best_result is not None:
        raw = [detail.best_result]

    hits = []
    for r in raw:
        b = getattr(r, "box", None)
        if b is None:
            continue
        cnt = int(getattr(r, "count", 0) or 0)
        if cnt < min_inliers:
            continue
        bx0, by0 = int(b.x), int(b.y)
        bx1, by1 = bx0 + int(b.w), by0 + int(b.h)
        area = (int(b.w) * int(b.h)) / float(_TPL_W * _TPL_H)
        if not (0.4 <= area <= 3.0):
            continue
        box = (bx0, by0, bx1, by1)
        if any(_iou(box, h[3]) > 0.35 for h in hits):
            continue
        hits.append((cnt, (bx0 + bx1) / 2.0, (by0 + by1) / 2.0,
                     box, "feature", "?"))
    hits.sort(key=lambda t: -t[0])
    return hits


def _to_result(hits):
    best = max(hits, key=lambda t: t[0])
    # 注意：Maa 的 box 是 (x, y, w, h)，不是 (x0, y0, x1, y1)。
    bx0, by0, bx1, by1 = best[3]
    return CustomRecognition.AnalyzeResult(
        box=(bx0, by0, bx1 - bx0, by1 - by0),
        detail={
            "count": len(hits),
            "chickens": [
                {"inliers": h[0], "cx": round(h[1], 1), "cy": round(h[2], 1),
                 "box": list(h[3]), "template": h[4], "pose": h[5]}
                for h in hits
            ],
            "blocked_x": [[h[3][0], h[3][2]] for h in hits],
        },
    )


_FINDER = None


def get_finder():
    global _FINDER
    if _FINDER is None:
        _FINDER = ChickenFinder()
    return _FINDER


try:
    from maa.custom_recognition import CustomRecognition
    from maa.context import Context

    class ChickenRecognition(CustomRecognition):
        """pipeline 节点用 recognition=Custom / custom_recognition=Chicken。

        Agent 子进程模式需要 register() 把它挂到 AgentServer 上；
        进程内模式用 Resource.register_custom_recognition("Chicken", ...)。
        """

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            node_roi = getattr(argv, "roi", None)
            if node_roi is not None and node_roi.w > 0 and node_roi.h > 0:
                roi = [node_roi.x, node_roi.y, node_roi.w, node_roi.h]
            else:
                roi = param.get("roi") or [0, 250, 720, 810]
            min_inliers = int(param.get("min_inliers", 10))

            if not hasattr(cv2, "SIFT_create"):
                hits = _feature_hits(context, argv.image, roi, min_inliers)
            else:
                finder = get_finder()
                if not finder.ready:
                    return CustomRecognition.AnalyzeResult(
                        box=None, detail={"error": "no chicken template found"})
                hits = finder.find(
                    argv.image, roi, max_rounds=int(param.get("max_rounds", 8)))
                if min_inliers != finder.min_inliers:
                    hits = [h for h in hits if h[0] >= min_inliers]
            if not hits:
                return CustomRecognition.AnalyzeResult(
                    box=None, detail={"count": 0, "chickens": []})
            return _to_result(hits)

    def register(name="Chicken"):
        """注册到 AgentServer（agent/main.py 里调用）。"""
        from maa.agent.agent_server import AgentServer
        return AgentServer.register_custom_recognition(name, ChickenRecognition())

except Exception as _e:  # pragma: no cover - maa 不可用时只保留算法层
    ChickenRecognition = None
    _IMPORT_ERROR = _e

    def register(name="Chicken"):
        raise RuntimeError(f"maa import failed: {_IMPORT_ERROR}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    f = ChickenFinder(tpl_dir=os.path.join(ROOT, "User_screenshot"))
    # User_screenshot 里是 frontage_1.png 这种原名，做一次映射
    f.templates = []
    for pose, names in POSES.items():
        for n in names:
            src = os.path.join(ROOT, "User_screenshot",
                               n.replace("chicken_", ""))
            img = _imread_any(src)
            kp, des = f.sift.detectAndCompute(img, None)
            th, tw = img.shape[:2]
            corners = np.float32(
                [[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
            f.templates.append((n, pose, kp, des, corners, float(tw * th)))
    for name in sys.argv[1:] or ["user_test_1.png", "user_test_2.png"]:
        img = _imread_any(os.path.join(ROOT, "User_screenshot", name))
        for h in f.find(img, (0, 250, 720, 810)):
            print(f"{name:18s} {h[5]:8s} {h[4]:22s} inl={h[0]:3d} "
                  f"cx={h[1]:6.1f} cy={h[2]:6.1f} box={h[3]}")
