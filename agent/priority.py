"""物品收集优先级：0=不收集，1..7 越大越先。

内部短名与物品模板一致：
  stone=云英, fabric=卷轴, jade=玉璧, honey=蜂蜜,
  silver=饷银, berry=枸杞, alcohol=酒米

当前默认只收集酒米；后续可由 interface.json 的 input/pipeline_override
覆盖成任意 0..7 组合。
"""
from __future__ import annotations

ITEM_ORDER = ["stone", "fabric", "jade", "honey", "silver", "berry", "alcohol"]
ITEM_LABELS = {
    "stone": "云英",
    "fabric": "卷轴",
    "jade": "玉璧",
    "honey": "蜂蜜",
    "silver": "饷银",
    "berry": "枸杞",
    "alcohol": "酒米",
}

# 先只收集酒米；其余资源默认关闭。
DEFAULT_LEVELS = {
    "stone": 0,
    "fabric": 0,
    "jade": 0,
    "honey": 0,
    "silver": 0,
    "berry": 0,
    "alcohol": 7,
}

_LEVELS = dict(DEFAULT_LEVELS)


def _level(value) -> int:
    try:
        level = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(7, level))


def set_levels(values) -> None:
    """设置当前收集等级；缺失项按 0 处理。"""
    values = values or {}
    _LEVELS.clear()
    for item in ITEM_ORDER:
        _LEVELS[item] = _level(values.get(item, 0))


def get_levels() -> dict[str, int]:
    return dict(_LEVELS)


def reset_defaults() -> None:
    _LEVELS.clear()
    _LEVELS.update(DEFAULT_LEVELS)


def order(values=None) -> list[str]:
    """返回从高到低的物品短名；0 分不进入列表。"""
    if values is None:
        values = _LEVELS
    if isinstance(values, (list, tuple)):
        # 兼容旧 DEFAULT_PRIORITY：列表本身就是从高到低。
        return [str(v) for v in values]
    if not isinstance(values, dict):
        values = _LEVELS
    parsed = {}
    for item in ITEM_ORDER:
        raw = values.get(item, 0)
        try:
            parsed[item] = _level(int(raw))
        except (TypeError, ValueError):
            # pipeline_override 没有被替换成数字时（例如仍含
            # "{stone_priority}"），退回当前默认等级，避免全部静默关闭。
            parsed = dict(_LEVELS)
            break
    ranked = []
    for item in ITEM_ORDER:
        level = parsed.get(item, 0)
        if level > 0:
            ranked.append((item, level))
    ranked.sort(key=lambda pair: (-pair[1], ITEM_ORDER.index(pair[0])))
    return [item for item, _level_value in ranked]


def levels_from_recognition_param(raw) -> list[str]:
    """从 pipeline 的 priority_levels 解析排序。"""
    return order(raw)
