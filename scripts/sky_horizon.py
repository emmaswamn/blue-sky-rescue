"""海天分界：按行/按列自顶向下，连续 gap_rows 行非天 → 截断（忽略海面误检孤岛）。"""
from __future__ import annotations

import cv2
import numpy as np

HORIZON_GAP_ROWS = 3
# 行扫描只看画面中央带，避免两侧礁石把天际线拉歪
ROW_SCAN_X0 = 0.20
ROW_SCAN_X1 = 0.80
ROW_MIN_FILL = 0.55   # 行内「像天」
ROW_SEA_FILL = 0.15   # 行内「像海」— 仅连续低于此值才计 gap（云区波动不算）

# A 轮动态 ROI（底向上三组扫描，替换写死 0.65）
ROI_GROUP_COLS = 5
ROI_MARGIN = 0          # y_cap = max(三组 y_hor) + margin
ROI_FALLBACK_FRAC = 0.65  # 三组均读失败时回退
ROI_OUTLIER_SPREAD = 60   # max-min 超过此值 → 用 median 抗单列异常（岸/岛）


def y_hor_column_top_down(col: np.ndarray, gap_rows: int = HORIZON_GAP_ROWS) -> int:
    """单列：自 y=0 向下；连续 gap_rows 行非天则停，返回最后一条天行 y。"""
    gap = 0
    last_sky = -1
    for y in range(len(col)):
        if col[y] > 0:
            last_sky = y
            gap = 0
        else:
            gap += 1
            if gap >= gap_rows:
                break
    return last_sky


def y_hor_global_from_rows(
    mask_a: np.ndarray,
    min_fill: float = ROW_MIN_FILL,
    sea_fill: float = ROW_SEA_FILL,
    gap_rows: int = HORIZON_GAP_ROWS,
    x0_frac: float = ROW_SCAN_X0,
    x1_frac: float = ROW_SCAN_X1,
) -> int:
    """按行自上而下：进入天区后，连续 gap_rows 行中央带 fill < sea_fill → 海天。"""
    h, w = mask_a.shape
    x0, x1 = int(w * x0_frac), int(w * x1_frac)
    band_w = max(x1 - x0, 1)
    gap = 0
    y_hor = 0
    seen_sky = False
    for y in range(h):
        fill = float(np.count_nonzero(mask_a[y, x0:x1])) / band_w
        if fill >= min_fill:
            seen_sky = True
            y_hor = y
            gap = 0
        elif seen_sky and fill < sea_fill:
            gap += 1
            if gap >= gap_rows:
                break
        else:
            gap = 0
    return y_hor


def y_hor_per_column(
    mask_a: np.ndarray,
    y_cap: int,
    smooth: int = 0,
) -> np.ndarray:
    """按列取 cap 以下最底天行（保留云区），不超过行扫描海天。"""
    h, w = mask_a.shape
    y_hor = np.full(w, -1, dtype=np.int32)
    for x in range(w):
        ys = np.where(mask_a[:, x] > 0)[0]
        ys = ys[ys <= y_cap]
        if len(ys):
            y_hor[x] = int(ys.max())
    if (y_hor < 0).any():
        valid = y_hor[y_hor >= 0]
        fallback = int(np.median(valid)) if len(valid) else y_cap
        y_hor[y_hor < 0] = fallback
    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float64) / smooth
        y_hor = np.round(np.convolve(y_hor.astype(np.float64), kernel, mode="same")).astype(np.int32)
        y_hor = np.minimum(y_hor, y_cap)
    return y_hor


def allow_from_y_hor(y_hor: np.ndarray, h: int, w: int, margin: int) -> np.ndarray:
    y_idx = np.arange(h, dtype=np.int32)[:, None]
    return ((y_idx <= y_hor[None, :] + margin).astype(np.uint8) * 255)


def allow_from_mask_a_columns(
    mask_a: np.ndarray,
    margin: int,
    smooth: int,
    gap_rows: int = HORIZON_GAP_ROWS,
) -> np.ndarray:
    """B′ allow：先按行找全局海天，再按列微调（列界不超过行界 + margin）。"""
    h, w = mask_a.shape
    y_global = y_hor_global_from_rows(mask_a, gap_rows=gap_rows)
    y_cap = y_global + margin
    y_hor = y_hor_per_column(mask_a, y_cap, smooth)
    return allow_from_y_hor(y_hor, h, w, margin)


def _row_fill_band(mask: np.ndarray, y: int, x0: int, x1: int) -> float:
    band_w = max(x1 - x0, 1)
    return float(np.count_nonzero(mask[y, x0:x1])) / band_w


def y_hor_bottom_up_band(
    mask: np.ndarray,
    x0: int,
    x1: int,
    sky_fill: float = ROW_MIN_FILL,
    gap_rows: int = HORIZON_GAP_ROWS,
) -> int:
    """单列带自底向上：连续 gap_rows 行填充 >= sky_fill → 海天；返回天区最底行 y。"""
    h, w = mask.shape
    x0, x1 = max(0, x0), min(w, x1)
    if x1 <= x0:
        return -1
    sky_run = 0
    y_hor = -1
    for y in range(h - 1, -1, -1):
        fill = _row_fill_band(mask, y, x0, x1)
        if fill >= sky_fill:
            sky_run += 1
            if sky_run >= gap_rows:
                y_hor = y + gap_rows - 1
                break
        else:
            sky_run = 0
    return y_hor


def y_roi_cap_bottom_up_groups(
    mask: np.ndarray,
    group_cols: int = ROI_GROUP_COLS,
    margin: int = ROI_MARGIN,
    sky_fill: float = ROW_MIN_FILL,
    gap_rows: int = HORIZON_GAP_ROWS,
    fallback_frac: float = ROI_FALLBACK_FRAC,
) -> tuple[int, list[int], bool]:
    """左/中/右各 group_cols 列底向上；取 max(y) 保守 cap。返回 (y_cap, 有效 y 列表, 是否 fallback)。"""
    h, w = mask.shape
    half = group_cols // 2
    cx = w // 2
    groups = (
        (0, group_cols),
        (cx - half, cx - half + group_cols),
        (w - group_cols, w),
    )
    ys: list[int] = []
    for x0, x1 in groups:
        y = y_hor_bottom_up_band(mask, x0, x1, sky_fill=sky_fill, gap_rows=gap_rows)
        if y >= 0:
            ys.append(y)
    if not ys:
        return int(h * fallback_frac), [], True
    med = int(np.median(ys))
    mx = max(ys)
    y_global = med if (len(ys) >= 2 and mx - med > ROI_OUTLIER_SPREAD) else mx
    return min(y_global + margin, h - 1), ys, False


def apply_roi_y_cap(mask: np.ndarray, y_cap: int) -> np.ndarray:
    """保留 y <= y_cap 的 mask 区域。"""
    h, w = mask.shape
    roi = np.zeros((h, w), np.uint8)
    roi[: y_cap + 1, :] = 255
    return cv2.bitwise_and(mask, roi)


def trim_mask_a_by_horizon(
    mask_a: np.ndarray,
    margin: int = 0,
    smooth: int = 0,
    gap_rows: int = HORIZON_GAP_ROWS,
) -> np.ndarray:
    """A″：按行切海天，再按列去掉 cap 以下误检（海面白沫等）。"""
    h, w = mask_a.shape
    y_global = y_hor_global_from_rows(mask_a, gap_rows=gap_rows)
    y_cap = y_global + margin
    y_hor = y_hor_per_column(mask_a, y_cap, smooth)
    allow = allow_from_y_hor(y_hor, h, w, margin=0)
    return cv2.bitwise_and(mask_a, allow)
