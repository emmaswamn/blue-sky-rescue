#!/usr/bin/env python3
"""§2.8.2 B 轮：顶 band 补云 + 与 mask_a OR 合并 → mask_hsv.jpg"""
from pathlib import Path

import cv2
import numpy as np

FRAME = Path("output/sky-filter/frame_raw.jpg")
MASK_A = Path("output/sky-filter/mask_hsv_a.png")
OUT_B = Path("output/sky-filter/mask_hsv_b.png")
OUT = Path("output/sky-filter/mask_hsv.png")
OUT_OVERLAY = Path("output/sky-filter/mask_hsv_overlay.jpg")

TOP_BAND = 0.60       # 画面高比例，非海平线；见 step-02 §2.8.3
LOW_SAT_MAX_S = 28
LOW_SAT_MIN_V = 120
CLOSE_B_ITER = 1
EXCLUDE_RIGHT_ISLAND = False
EXCLUDE_Y_FROM = 0.40
EXCLUDE_X_FROM = 0.72


def main() -> None:
    frame = cv2.imread(str(FRAME))
    mask_a = cv2.imread(str(MASK_A), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask_a is None:
        raise SystemExit(f"需要 {FRAME} 和 {MASK_A} — 先跑 scripts/run_mask_hsv_a.py")

    h, w = frame.shape[:2]
    band_rows = int(h * TOP_BAND)
    print(f"B band: y=0..{band_rows-1} ({TOP_BAND:.0%} of h={h})")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    top = np.zeros((h, w), np.uint8)
    top[: int(h * TOP_BAND), :] = 255
    mask_b = (((s <= LOW_SAT_MAX_S) & (v >= LOW_SAT_MIN_V)).astype(np.uint8) * 255)
    mask_b = cv2.bitwise_and(mask_b, top)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if CLOSE_B_ITER:
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, k, CLOSE_B_ITER)
        mask_b = cv2.bitwise_and(mask_b, top)

    mask = cv2.bitwise_or(mask_a, mask_b)
    if EXCLUDE_RIGHT_ISLAND:
        mask[int(h * EXCLUDE_Y_FROM) :, int(w * EXCLUDE_X_FROM) :] = 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_B), mask_b)
    cv2.imwrite(str(OUT), mask)

    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 255)
    cv2.imwrite(str(OUT_OVERLAY), cv2.addWeighted(frame, 0.7, overlay, 0.3, 0))
    print("saved", OUT_B, "(B 仅顶云)")
    print("saved", OUT, OUT_OVERLAY, "← 最终")


if __name__ == "__main__":
    main()
