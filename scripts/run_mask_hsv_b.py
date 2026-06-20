#!/usr/bin/env python3
"""§2.8.2 B 轮：低 S 补云 + 与 mask_a OR 合并 → mask_hsv.png

HORIZON_MODE:
  mask_a     — 按列扫 mask_a 天际线（B′，见 Step2-A轮取点逻辑与B轮海天约束）
  bottom_cc  — 贴底连通非天区取反
  top_band   — 旧方案：固定画面高比例
"""
from pathlib import Path

import cv2
import numpy as np

FRAME = Path("output/sky-filter/frame_raw.jpg")
MASK_A = Path("output/sky-filter/mask_hsv_a.png")
OUT_B = Path("output/sky-filter/mask_hsv_b.png")
OUT = Path("output/sky-filter/mask_hsv.png")
OUT_OVERLAY = Path("output/sky-filter/mask_hsv_overlay.jpg")

HORIZON_MODE = "mask_a"   # mask_a | bottom_cc | top_band
HORIZON_MARGIN = 5        # y <= y_hor + margin，给天际薄云
Y_HOR_SMOOTH = 7          # 1D 平滑窗口（奇数）；0 = 不平滑
B_ONLY_HOLES = False      # True: 只在 mask_a 为黑处补云

TOP_BAND = 0.47           # 仅 top_band 模式
LOW_SAT_MAX_S = 28
LOW_SAT_MIN_V = 120
CLOSE_B_ITER = 1
EXCLUDE_RIGHT_ISLAND = False
EXCLUDE_Y_FROM = 0.40
EXCLUDE_X_FROM = 0.72


def allow_from_mask_a_columns(mask_a: np.ndarray, margin: int, smooth: int) -> np.ndarray:
    h, w = mask_a.shape
    y_hor = np.full(w, -1, dtype=np.int32)
    for x in range(w):
        ys = np.where(mask_a[:, x] > 0)[0]
        if len(ys) > 0:
            y_hor[x] = int(ys.max())

    if (y_hor < 0).any():
        valid = y_hor[y_hor >= 0]
        fallback = int(np.median(valid)) if len(valid) else int(h * 0.5)
        y_hor[y_hor < 0] = fallback

    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float64) / smooth
        y_hor = np.round(np.convolve(y_hor.astype(np.float64), kernel, mode="same")).astype(np.int32)

    y_idx = np.arange(h, dtype=np.int32)[:, None]
    return ((y_idx <= y_hor[None, :] + margin).astype(np.uint8) * 255)


def allow_from_mask_a_bottom_cc(mask_a: np.ndarray) -> np.ndarray:
    h, w = mask_a.shape
    non_sky = cv2.bitwise_not(mask_a)
    n, labels = cv2.connectedComponents(non_sky)
    exclude = np.zeros((h, w), np.uint8)
    bottom_labels = set(labels[h - 1, :].tolist()) - {0}
    for label in bottom_labels:
        exclude[labels == label] = 255
    return cv2.bitwise_not(exclude)


def allow_from_top_band(h: int, w: int, top_band: float) -> np.ndarray:
    allow = np.zeros((h, w), np.uint8)
    allow[: int(h * top_band), :] = 255
    return allow


def main() -> None:
    frame = cv2.imread(str(FRAME))
    mask_a = cv2.imread(str(MASK_A), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask_a is None:
        raise SystemExit(f"需要 {FRAME} 和 {MASK_A} — 先跑 scripts/run_mask_hsv_a.py")

    h, w = frame.shape[:2]
    if mask_a.shape[:2] != (h, w):
        raise SystemExit(f"mask_a 尺寸 {mask_a.shape[:2]} 与 frame {h,w} 不一致")

    if HORIZON_MODE == "mask_a":
        allow = allow_from_mask_a_columns(mask_a, HORIZON_MARGIN, Y_HOR_SMOOTH)
        y_hor = np.full(w, -1, dtype=np.int32)
        for x in range(w):
            ys = np.where(mask_a[:, x] > 0)[0]
            if len(ys) > 0:
                y_hor[x] = int(ys.max())
        y_min, y_max = int(y_hor[y_hor >= 0].min()), int(y_hor.max())
        print(
            f"B′ column: y_hor≈{y_min}..{y_max}, margin={HORIZON_MARGIN}, smooth={Y_HOR_SMOOTH}"
        )
    elif HORIZON_MODE == "bottom_cc":
        allow = allow_from_mask_a_bottom_cc(mask_a)
        print("B′ bottom_cc: exclude 贴底非天连通域")
    elif HORIZON_MODE == "top_band":
        allow = allow_from_top_band(h, w, TOP_BAND)
        print(f"B band: y=0..{int(h * TOP_BAND) - 1} ({TOP_BAND:.0%} of h={h})")
    else:
        raise SystemExit(f"未知 HORIZON_MODE={HORIZON_MODE!r}")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    mask_b = (((s <= LOW_SAT_MAX_S) & (v >= LOW_SAT_MIN_V)).astype(np.uint8) * 255)
    mask_b = cv2.bitwise_and(mask_b, allow)
    if B_ONLY_HOLES:
        holes = cv2.bitwise_not(mask_a)
        mask_b = cv2.bitwise_and(mask_b, holes)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if CLOSE_B_ITER:
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, k, CLOSE_B_ITER)
        mask_b = cv2.bitwise_and(mask_b, allow)
        if B_ONLY_HOLES:
            mask_b = cv2.bitwise_and(mask_b, holes)

    mask = cv2.bitwise_or(mask_a, mask_b)
    if EXCLUDE_RIGHT_ISLAND:
        mask[int(h * EXCLUDE_Y_FROM) :, int(w * EXCLUDE_X_FROM) :] = 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_B), mask_b)
    cv2.imwrite(str(OUT), mask)

    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 255)
    cv2.imwrite(str(OUT_OVERLAY), cv2.addWeighted(frame, 0.7, overlay, 0.3, 0))
    print("saved", OUT_B, "(B 仅补云)")
    print("saved", OUT, OUT_OVERLAY, "← 最终")


if __name__ == "__main__":
    main()
