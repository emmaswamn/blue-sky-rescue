#!/usr/bin/env python3
"""§2.8.2 A 轮：inRange → ROI → morph → A′ exclude → A″ → mask_hsv_a.png"""
import argparse
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
EXCLUDE_JSON = Path("output/sky-filter/mask_exclude.json")
OUT_A = Path("output/sky-filter/mask_hsv_a.png")  # 勿用 jpg，压缩会让 exclude 边缘漏黄
OUT_A_OVERLAY = Path("output/sky-filter/mask_hsv_a_overlay.jpg")
CLAMP_H_LOWER = 92
Y_HOR_SMOOTH = 7
ROI_MODE = "bottom_up"  # bottom_up | fixed | full


def load_exclude_rects(path: Path) -> list[tuple[int, int, int, int]]:
    if not path.exists():
        return []
    ex = json.loads(path.read_text())
    raw = ex.get("rects", [])
    if raw and not isinstance(raw[0], (list, tuple)):
        raise SystemExit(
            "mask_exclude.json 格式错：rects 须为 [[x1,y1,x2,y2], ...]，"
            f"不能写成 [x1,y1,x2,y2]。当前: {raw!r}"
        )
    rects = []
    for i, item in enumerate(raw):
        if not item:
            continue  # [] 或空：本轮不扣
        if not isinstance(item, (list, tuple)):
            raise SystemExit(
                f"rects[{i}] 须为 [x1,y1,x2,y2]，收到: {item!r}"
            )
        if len(item) != 4:
            raise SystemExit(f"rects[{i}] 须 4 个数 [x1,y1,x2,y2]，收到: {item!r}")
        rects.append(tuple(int(v) for v in item))
    return rects


def apply_roi(mask: np.ndarray, h: int, mode: str) -> tuple[np.ndarray, int]:
    if mode == "full":
        print("ROI: full frame (no cap)")
        return mask, h - 1
    if mode == "fixed":
        y_cap = int(h * ROI_FALLBACK_FRAC)
        print(f"ROI fixed: y_cap={y_cap} ({ROI_FALLBACK_FRAC:.0%} of h={h})")
        return apply_roi_y_cap(mask, y_cap), y_cap
    if mode == "bottom_up":
        y_cap, groups, fallback = y_roi_cap_bottom_up_groups(
            mask, margin=ROI_MARGIN, gap_rows=HORIZON_GAP_ROWS
        )
        tag = "fallback" if fallback else f"groups={groups}"
        print(f"ROI bottom_up: y_cap={y_cap} ({tag})")
        return apply_roi_y_cap(mask, y_cap), y_cap
    raise SystemExit(f"未知 ROI_MODE={mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A 轮 HSV mask")
    parser.add_argument(
        "--roi-mode",
        choices=("bottom_up", "fixed", "full"),
        default=ROI_MODE,
        help="bottom_up=三组底向上(默认); fixed=0.65; full=不裁",
    )
    args = parser.parse_args()

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
    mask = cv2.inRange(hsv, lower, upper)
    mask, y_cap = apply_roi(mask, h, args.roi_mode)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)

    rects = load_exclude_rects(EXCLUDE_JSON)
    for x1, y1, x2, y2 in rects:
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        mask[y1:y2, x1:x2] = 0
    if rects:
        print("A′ manual exclude:", len(rects), "rect(s) from", EXCLUDE_JSON)

    mask = trim_mask_a_by_horizon(
        mask, margin=0, smooth=Y_HOR_SMOOTH, gap_rows=HORIZON_GAP_ROWS
    )
    print(f"A″ horizon trim: gap_rows={HORIZON_GAP_ROWS}, smooth={Y_HOR_SMOOTH}")

    OUT_A.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_A), mask)
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 255)
    cv2.imwrite(str(OUT_A_OVERLAY), cv2.addWeighted(frame, 0.7, overlay, 0.3, 0))
    print("A 轮锁定", OUT_A, "+ overlay", OUT_A_OVERLAY)
    print("lower", lower.tolist(), "upper", upper.tolist())
    if rects:
        print("→ 改 exclude 后请再跑: python scripts/run_mask_hsv_b.py")
    else:
        print("→ 满意 A 后跑: python scripts/run_mask_hsv_b.py")


if __name__ == "__main__":
    main()
