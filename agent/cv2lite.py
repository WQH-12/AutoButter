"""cv2 的 numpy + Pillow 子集实现，供 Android 上跑 agent 时替代 opencv。

Android 的 MaaAgentCoreAndroid 只带 CPython 3.13 + numpy，没有 opencv；
桌面端仍用真 cv2。调用方按下面的方式导入，两边都能跑：

    try:
        import cv2
    except ImportError:
        import cv2lite as cv2

只实现 agent/*.py 实际用到的 API，语义对齐 OpenCV 4.x：

  imread / imdecode / imwrite(不支持)
  cvtColor(BGR2GRAY / GRAY2BGR / BGR2HSV) / split
  matchTemplate(TM_CCOEFF_NORMED) / minMaxLoc
  resize(INTER_AREA / INTER_LINEAR)
  morphologyEx(MORPH_CLOSE)
  findContours(RETR_EXTERNAL) / connectedComponentsWithStats
  boundingRect / contourArea
  bitwise_or / absdiff / ones

SIFT / BFMatcher / findHomography / perspectiveTransform 没有实现：
chicken_reco 的这部分改走 pipeline 的 FeatureMatch（见 agent/chicken_reco.py）。
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

IMREAD_GRAYSCALE = 0
IMREAD_COLOR = 1
IMREAD_UNCHANGED = -1

COLOR_BGR2GRAY = 6
COLOR_GRAY2BGR = 8
COLOR_BGR2HSV = 40

TM_CCOEFF_NORMED = 5

INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_AREA = 3

MORPH_CLOSE = 3

RETR_EXTERNAL = 0
CHAIN_APPROX_SIMPLE = 2

uint8 = np.uint8


# --------------------------------------------------------------------------
# 读图
# --------------------------------------------------------------------------

def imdecode(buf, flags=IMREAD_COLOR):
    """把内存里的图片字节解成 ndarray；失败返回 None（与 OpenCV 一致）。"""
    try:
        im = Image.open(BytesIO(bytes(buf)))
        im.load()
    except Exception:  # noqa: BLE001
        return None

    if flags == IMREAD_UNCHANGED:
        if im.mode == "P":
            im = im.convert("RGBA")
        arr = np.asarray(im)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = arr[:, :, [2, 1, 0, 3]]
            elif arr.shape[2] == 3:
                arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)

    if flags == IMREAD_GRAYSCALE:
        return np.ascontiguousarray(np.asarray(im.convert("L")))

    arr = np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(arr[:, :, ::-1])


def imread(path, flags=IMREAD_COLOR):
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    return imdecode(raw, flags)


# --------------------------------------------------------------------------
# 颜色
# --------------------------------------------------------------------------

def cvtColor(img, code):
    if code == COLOR_BGR2GRAY:
        if img.ndim == 2:
            return img
        b = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        r = img[:, :, 2].astype(np.float32)
        return np.clip(b * 0.114 + g * 0.587 + r * 0.299 + 0.5,
                       0, 255).astype(np.uint8)

    if code == COLOR_GRAY2BGR:
        if img.ndim == 3:
            return img
        return np.repeat(img[:, :, None], 3, axis=2)

    if code == COLOR_BGR2HSV:
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError("BGR2HSV needs a 3-channel image")
        b = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        r = img[:, :, 2].astype(np.int16)
        v = np.maximum(np.maximum(b, g), r)
        mn = np.minimum(np.minimum(b, g), r)
        diff = (v - mn).astype(np.int32)

        # OpenCV 8-bit HSV：H in [0,179]，S/V in [0,255]
        s = np.zeros_like(v, dtype=np.uint8)
        nz = v > 0
        s[nz] = np.clip(np.round(diff[nz] * 255.0 / v[nz]), 0, 255).astype(np.uint8)

        h = np.zeros_like(v, dtype=np.int16)
        nz = diff > 0
        if nz.any():
            vv = v[nz].astype(np.int32)
            dd = diff[nz].astype(np.int32)
            rr = r[nz].astype(np.int32)
            gg = g[nz].astype(np.int32)
            bb = b[nz].astype(np.int32)
            hv = np.empty_like(dd)
            is_r = vv == rr
            is_g = (vv == gg) & ~is_r
            is_b = ~(is_r | is_g)
            hv[is_r] = np.round(30.0 * (gg[is_r] - bb[is_r]) / dd[is_r])
            hv[is_g] = np.round(60.0 + 30.0 * (bb[is_g] - rr[is_g]) / dd[is_g])
            hv[is_b] = np.round(120.0 + 30.0 * (rr[is_b] - gg[is_b]) / dd[is_b])
            hv = np.where(hv < 0, hv + 180, hv)
            hv = np.where(hv >= 180, hv - 180, hv)
            h[nz] = hv.astype(np.int16)

        return np.dstack([h.astype(np.uint8), s, v.astype(np.uint8)])

    raise NotImplementedError("cvtColor code %r" % (code,))


def split(img):
    if img.ndim == 2:
        return (img,)
    return tuple(np.ascontiguousarray(img[:, :, i]) for i in range(img.shape[2]))


def bitwise_or(a, b):
    return np.bitwise_or(a, b)


def absdiff(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)


def ones(shape, dtype=None):
    return np.ones(shape, dtype=dtype)


# --------------------------------------------------------------------------
# 模板匹配
# --------------------------------------------------------------------------

def _correlate(img, tpl):
    """valid 区域的 sum(img[y:y+h, x:x+w] * tpl)，FFT 实现。"""
    H, W = img.shape
    h, w = tpl.shape
    F = np.fft.rfft2(img, s=(H, W))
    K = np.fft.rfft2(tpl[::-1, ::-1], s=(H, W))
    out = np.fft.irfft2(F * K, s=(H, W))
    return out[h - 1:H, w - 1:W]


def _window_sums(arr, h, w):
    """每个 h x w 窗口的和，形状 (H-h+1, W-w+1)。"""
    c = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    c[1:, 1:] = arr.cumsum(0).cumsum(1)
    return (c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w])


def matchTemplate(img, tpl, method=TM_CCOEFF_NORMED):
    """TM_CCOEFF_NORMED。多通道按 OpenCV 的口径逐通道累加分子/分母。"""
    if method != TM_CCOEFF_NORMED:
        raise NotImplementedError("matchTemplate method %r" % (method,))
    if img.ndim != tpl.ndim:
        raise ValueError("image and template channel count differ")

    H, W = img.shape[:2]
    h, w = tpl.shape[:2]
    if h > H or w > W:
        raise ValueError("template is larger than image")

    if img.ndim == 2:
        pairs = [(img, tpl)]
    elif img.ndim == 3:
        pairs = [(img[:, :, c], tpl[:, :, c]) for c in range(img.shape[2])]
    else:
        raise ValueError("unsupported ndim %r" % (img.ndim,))

    n = float(h * w)
    num = None
    tss = 0.0
    iss = None

    for src_u8, tp_u8 in pairs:
        src = src_u8.astype(np.float64)
        tp = tp_u8.astype(np.float64)
        tp0 = tp - tp.mean()
        tss += float((tp0 * tp0).sum())

        s1 = _window_sums(src, h, w)
        s2 = _window_sums(src * src, h, w)
        var = s2 - s1 * s1 / n
        np.maximum(var, 0.0, out=var)
        iss = var if iss is None else iss + var

        corr = _correlate(src, tp0)
        num = corr if num is None else num + corr

    den = np.sqrt(tss * iss)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = num / den
    res[~np.isfinite(res)] = 0.0
    return res


def minMaxLoc(arr):
    flat_min = int(np.argmin(arr))
    flat_max = int(np.argmax(arr))
    H, W = arr.shape[:2]
    min_loc = (flat_min % W, flat_min // W)
    max_loc = (flat_max % W, flat_max // W)
    return float(arr.flat[flat_min]), float(arr.flat[flat_max]), min_loc, max_loc


# --------------------------------------------------------------------------
# 缩放 / 形态学
# --------------------------------------------------------------------------

def resize(img, dsize, interpolation=INTER_LINEAR):
    tw, th = int(dsize[0]), int(dsize[1])
    if tw <= 0 or th <= 0:
        raise ValueError("bad dsize %r" % (dsize,))
    H, W = img.shape[:2]
    if (tw, th) == (W, H):
        return img.copy()

    if interpolation == INTER_NEAREST:
        ys = np.clip((np.arange(th) * H // th), 0, H - 1)
        xs = np.clip((np.arange(tw) * W // tw), 0, W - 1)
        return np.ascontiguousarray(img[ys][:, xs])

    # 面积平均退化成双线性：agent 里只拿它缩模板/头像，
    # 分数阈值有足够余量，不值得为此写一套整型面积平均
    ys = (np.arange(th, dtype=np.float64) + 0.5) * H / th - 0.5
    xs = (np.arange(tw, dtype=np.float64) + 0.5) * W / tw - 0.5
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]

    src = img.astype(np.float64)
    if img.ndim == 2:
        top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
        bot = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
        out = top * (1 - wy) + bot * wy
    else:
        wy3 = wy[:, :, None]
        wx3 = wx[:, :, None]
        top = src[y0][:, x0] * (1 - wx3) + src[y0][:, x1] * wx3
        bot = src[y1][:, x0] * (1 - wx3) + src[y1][:, x1] * wx3
        out = top * (1 - wy3) + bot * wy3

    if img.dtype == np.uint8:
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return out.astype(img.dtype)


def _dilate(mask, kh, kw):
    H, W = mask.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(mask, ((ph, kh - 1 - ph), (pw, kw - 1 - pw)),
                 mode="constant", constant_values=0)
    out = np.zeros_like(mask)
    for dy in range(kh):
        for dx in range(kw):
            out |= pad[dy:dy + H, dx:dx + W]
    return out


def _erode(mask, kh, kw):
    H, W = mask.shape
    ph, pw = kh // 2, kw // 2
    fill = 255 if mask.dtype == np.uint8 else 1
    pad = np.pad(mask, ((ph, kh - 1 - ph), (pw, kw - 1 - pw)),
                 mode="constant", constant_values=fill)
    out = np.full_like(mask, fill)
    for dy in range(kh):
        for dx in range(kw):
            out &= pad[dy:dy + H, dx:dx + W]
    return out


def morphologyEx(mask, op, kernel):
    if op != MORPH_CLOSE:
        raise NotImplementedError("morphologyEx op %r" % (op,))
    kh, kw = (int(kernel.shape[0]), int(kernel.shape[1]))
    return _erode(_dilate(mask, kh, kw), kh, kw)


# --------------------------------------------------------------------------
# 连通域
# --------------------------------------------------------------------------

def _row_runs(row):
    padded = np.empty(row.shape[0] + 2, dtype=bool)
    padded[0] = False
    padded[-1] = False
    padded[1:-1] = row
    d = np.diff(padded.astype(np.int8))
    # d[k] 描述 padded[k] -> padded[k+1]；padded 前多一个 False，
    # 所以 start = k 正好是 row 里的起点，end 要 +1 变成不含端
    return [(int(a), int(b))
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def _components(mask):
    """8 连通标记。返回 (labels, stats)，stats 每行 [x, y, w, h, area]，0 号是背景。"""
    H, W = mask.shape
    # 0 是背景，parent[0] = 0；run 的 id 从 1 开始
    parent = [0]

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    labels = np.zeros((H, W), dtype=np.int32)
    runs_prev = []
    all_runs = []
    next_id = 1  # 0 是背景

    for y in range(H):
        cur = []
        row = mask[y]
        if not row.any():
            runs_prev = []
            continue
        for x0, x1 in _row_runs(row):
            idx = next_id
            next_id += 1
            parent.append(idx)
            labels[y, x0:x1] = idx
            cur.append((x0, x1, idx))

        j = 0
        for x0, x1, idx in cur:
            while j < len(runs_prev) and runs_prev[j][1] < x0:
                j += 1
            k = j
            while k < len(runs_prev) and runs_prev[k][0] <= x1:
                union(idx, runs_prev[k][2])
                k += 1

        runs_prev = cur
        all_runs.extend((y, x0, x1, idx) for x0, x1, idx in cur)

    # 把临时 id 收拢成 1..n
    root_to_new = {}
    stats_acc = {}
    for y, x0, x1, idx in all_runs:
        r = find(idx)
        new = root_to_new.get(r)
        if new is None:
            new = len(root_to_new) + 1
            root_to_new[r] = new
            stats_acc[new] = [x0, y, x1, y, 0]
        g = stats_acc[new]
        if x0 < g[0]:
            g[0] = x0
        if y < g[1]:
            g[1] = y
        if x1 > g[2]:
            g[2] = x1
        if y > g[3]:
            g[3] = y
        g[4] += x1 - x0
        labels[y, x0:x1] = new

    n = len(stats_acc)
    stats = np.zeros((n + 1, 5), dtype=np.int32)
    for i in range(1, n + 1):
        x0, y0, x1, y1, area = stats_acc[i]
        stats[i] = (x0, y0, x1 - x0, y1 - y0 + 1, area)

    # labels 里残留的是临时 id，统一映射到新 id
    lut = np.zeros(next_id, dtype=np.int32)
    for old, new in root_to_new.items():
        lut[old] = new
    # 所有 union 过的临时 id 都要映射到根
    for old in range(1, next_id):
        lut[old] = root_to_new[find(old)]
    labels = lut[labels]
    return labels, stats


class _Contour:
    __slots__ = ("x", "y", "w", "h", "area")

    def __init__(self, x, y, w, h, area):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = area


def findContours(mask, mode=RETR_EXTERNAL, method=CHAIN_APPROX_SIMPLE):
    if mode != RETR_EXTERNAL:
        raise NotImplementedError("findContours mode %r" % (mode,))
    _labels, stats = _components(mask.astype(bool))
    out = []
    for i in range(1, stats.shape[0]):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area > 0:
            out.append(_Contour(x, y, w, h, area))
    return out, None


def connectedComponentsWithStats(mask, connectivity=8):
    labels, stats = _components(mask.astype(bool))
    n = stats.shape[0]
    cent = np.zeros((n, 2), dtype=np.float64)
    for i in range(1, n):
        ys, xs = np.nonzero(labels == i)
        if xs.size:
            cent[i] = (xs.mean(), ys.mean())
    return n, labels, stats, cent


def boundingRect(contour):
    return (int(contour.x), int(contour.y), int(contour.w), int(contour.h))


def contourArea(contour, oriented=False):
    return float(contour.area)
