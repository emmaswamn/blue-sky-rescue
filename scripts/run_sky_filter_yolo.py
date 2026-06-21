#!/usr/bin/env python3
"""Step 3 · 旧方案：整片 mask 统一 H/S 提蓝（step-03 原版）

与 run_sky_filter.py（a+c 分路）分开，方便对照。
输入：frame_raw.jpg + mask_hsv.png
产出：frame_after_unified.jpg
"""
from pathlib import Path

import cv2
import numpy as np

FRAME = Path("output/sky-filter/frame_raw.jpg")
MASK = Path("output/sky-filter/mask_yolo.png")
OUT = Path("output/sky-filter/frame_after_yolo.jpg")

MASK_BLUR_SIGMA = 3
HUE_SHIFT = -8
SAT_SCALE = 1.25
VAL_SCALE = 1.0


def soft_mask(mask_u8: np.ndarray, sigma: float) -> np.ndarray:
    m = mask_u8.astype(np.float32) / 255.0
    if sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.clip(m, 0.0, 1.0)


def apply_unified_sky_filter(
    bgr: np.ndarray,
    mask: np.ndarray,
    hue_shift: float = HUE_SHIFT,
    sat_scale: float = SAT_SCALE,
    val_scale: float = VAL_SCALE,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    m3 = mask[..., None]
    out = bgr.astype(np.float32) * (1.0 - m3) + boosted.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    frame = cv2.imread(str(FRAME))
    mask_u8 = cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask_u8 is None:
        raise SystemExit(
            f"需要 {FRAME} 和 {MASK} — 先做 Step 2（run_mask_hsv_a.py + run_mask_hsv_b.py）"
        )
    if frame.shape[:2] != mask_u8.shape[:2]:
        raise SystemExit(f"frame 与 mask 尺寸不一致")

    m = soft_mask(mask_u8, MASK_BLUR_SIGMA)
    after = apply_unified_sky_filter(frame, m, HUE_SHIFT, SAT_SCALE, VAL_SCALE)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), after)
    print(f"mask {int(np.count_nonzero(mask_u8))}px")
    print(f"hue{HUE_SHIFT} sat×{SAT_SCALE} val×{VAL_SCALE} blurσ={MASK_BLUR_SIGMA}")
    print("saved", OUT, "← 整片统一（旧 step-03）")


if __name__ == "__main__":
    main()
