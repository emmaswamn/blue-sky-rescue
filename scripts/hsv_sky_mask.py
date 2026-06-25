"""Step 2 A+B′ HSV sky mask — shared by pseudo-labels and HSV RTSP stream."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sky_horizon import (
    HORIZON_GAP_ROWS,
    ROI_FALLBACK_FRAC,
    ROI_MARGIN,
    allow_from_mask_a_columns,
    apply_roi_y_cap,
    trim_mask_a_by_horizon,
    y_roi_cap_bottom_up_groups,
)

DEFAULT_THRESH = Path("output/sky-filter/hsv_threshold_a.json")
DEFAULT_EXCLUDE = Path("output/sky-filter/mask_exclude.json")

CLAMP_H_LOWER = 92
Y_HOR_SMOOTH = 7
HORIZON_MARGIN = 5
LOW_SAT_MAX_S = 28
LOW_SAT_MIN_V = 120
CLOSE_B_ITER = 1


def load_hsv_bounds(thresh_path: Path, clamp_h_lower: int | None = CLAMP_H_LOWER) -> tuple[np.ndarray, np.ndarray]:
    cfg = json.loads(thresh_path.read_text(encoding="utf-8"))
    lower = np.array(cfg["lower"], dtype=np.uint8)
    upper = np.array(cfg["upper"], dtype=np.uint8)
    if clamp_h_lower is not None:
        lower[0] = max(lower[0], clamp_h_lower)
    upper[2] = 255
    return lower, upper


def load_exclude_rects(path: Path) -> list[tuple[int, int, int, int]]:
    if not path.exists():
        return []
    ex = json.loads(path.read_text(encoding="utf-8"))
    rects = []
    for item in ex.get("rects", []):
        if item and len(item) == 4:
            rects.append(tuple(int(v) for v in item))
    return rects


def _apply_roi(mask: np.ndarray, h: int, roi_mode: str) -> np.ndarray:
    if roi_mode == "full":
        return mask
    if roi_mode == "fixed":
        return apply_roi_y_cap(mask, int(h * ROI_FALLBACK_FRAC))
    if roi_mode == "bottom_up":
        y_cap, _, _ = y_roi_cap_bottom_up_groups(mask, margin=ROI_MARGIN, gap_rows=HORIZON_GAP_ROWS)
        return apply_roi_y_cap(mask, y_cap)
    raise ValueError(f"未知 roi_mode={roi_mode!r}")


def compute_mask_a(
    frame: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    exclude_rects: list[tuple[int, int, int, int]],
    roi_mode: str = "bottom_up",
) -> np.ndarray:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = _apply_roi(mask, h, roi_mode)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)
    for x1, y1, x2, y2 in exclude_rects:
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        mask[y1:y2, x1:x2] = 0
    return trim_mask_a_by_horizon(mask, margin=0, smooth=Y_HOR_SMOOTH, gap_rows=HORIZON_GAP_ROWS)


def compute_mask_b(frame: np.ndarray, mask_a: np.ndarray) -> np.ndarray:
    allow = allow_from_mask_a_columns(
        mask_a, HORIZON_MARGIN, Y_HOR_SMOOTH, gap_rows=HORIZON_GAP_ROWS
    )
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask_b = (((s <= LOW_SAT_MAX_S) & (v >= LOW_SAT_MIN_V)).astype(np.uint8) * 255)
    mask_b = cv2.bitwise_and(mask_b, allow)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if CLOSE_B_ITER:
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, k, CLOSE_B_ITER)
        mask_b = cv2.bitwise_and(mask_b, allow)
    return mask_b


def compute_hsv_sky_mask(
    frame: np.ndarray,
    thresh_path: Path = DEFAULT_THRESH,
    exclude_path: Path = DEFAULT_EXCLUDE,
    roi_mode: str = "bottom_up",
) -> np.ndarray:
    """A 轮 + B′ 合并，与 run_mask_hsv_a/b 及伪标注工厂一致。"""
    lower, upper = load_hsv_bounds(thresh_path)
    exclude_rects = load_exclude_rects(exclude_path)
    mask_a = compute_mask_a(frame, lower, upper, exclude_rects, roi_mode)
    return cv2.bitwise_or(mask_a, compute_mask_b(frame, mask_a))
