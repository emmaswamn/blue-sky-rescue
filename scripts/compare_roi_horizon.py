#!/usr/bin/env python3
"""对比 A 轮 ROI：写死 0.65 vs 底向上三组。产出并排 overlay + 打印 y_cap。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sky_horizon import (
    HORIZON_GAP_ROWS,
    ROI_FALLBACK_FRAC,
    ROI_MARGIN,
    apply_roi_y_cap,
    trim_mask_a_by_horizon,
    y_roi_cap_bottom_up_groups,
)

FRAME = Path("output/sky-filter/frame_raw.jpg")
THRESH = Path("output/sky-filter/hsv_threshold_a.json")
OUT_DIR = Path("output/sky-filter/debug/roi_horizon_compare")
CLAMP_H_LOWER = 92
Y_HOR_SMOOTH = 7


def _mask_a_pipeline(mask: np.ndarray, y_cap: int) -> np.ndarray:
    m = apply_roi_y_cap(mask, y_cap)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 2)
    return trim_mask_a_by_horizon(
        m, margin=0, smooth=Y_HOR_SMOOTH, gap_rows=HORIZON_GAP_ROWS
    )


def _overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    o = frame.copy()
    o[mask > 0] = (0, 255, 255)
    return cv2.addWeighted(frame, 0.7, o, 0.3, 0)


def main() -> None:
    cfg = json.loads(THRESH.read_text())
    lower = np.array(cfg["lower"], dtype=np.uint8)
    upper = np.array(cfg["upper"], dtype=np.uint8)
    if CLAMP_H_LOWER is not None:
        lower[0] = max(lower[0], CLAMP_H_LOWER)
    upper[2] = 255

    frame = cv2.imread(str(FRAME))
    if frame is None:
        raise SystemExit(f"读不到 {FRAME}")
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, lower, upper)

    y_fixed = int(h * ROI_FALLBACK_FRAC)
    y_dyn, groups, fallback = y_roi_cap_bottom_up_groups(
        raw, margin=ROI_MARGIN, gap_rows=HORIZON_GAP_ROWS
    )

    mask_fixed = _mask_a_pipeline(raw, y_fixed)
    mask_dyn = _mask_a_pipeline(raw, y_dyn)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "overlay_fixed_0.65.jpg"), _overlay(frame, mask_fixed))
    cv2.imwrite(str(OUT_DIR / "overlay_bottom_up.jpg"), _overlay(frame, mask_dyn))
    diff = cv2.absdiff(mask_fixed, mask_dyn)
    cv2.imwrite(str(OUT_DIR / "mask_diff_x4.png"), np.clip(diff.astype(np.uint16) * 4, 0, 255).astype(np.uint8))

    vis = frame.copy()
    cv2.line(vis, (0, y_fixed), (w - 1, y_fixed), (0, 0, 255), 2)
    cv2.line(vis, (0, y_dyn), (w - 1, y_dyn), (0, 255, 0), 2)
    cv2.putText(vis, f"red=fixed y={y_fixed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(
        vis,
        f"green=bottom_up y={y_dyn} groups={groups}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(OUT_DIR / "cap_lines.jpg"), vis)

    print(f"h={h}  fixed y_cap={y_fixed} ({ROI_FALLBACK_FRAC:.0%})")
    print(f"bottom_up y_cap={y_dyn}  groups={groups}  fallback={fallback}")
    print("saved", OUT_DIR)


if __name__ == "__main__":
    main()
