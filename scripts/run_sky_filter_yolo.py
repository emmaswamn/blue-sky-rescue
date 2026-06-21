#!/usr/bin/env python3
"""Step 4 · YOLO mask + unified 蓝天滤镜 → frame_after_yolo.jpg

输入：frame_raw.jpg + mask_yolo.png
"""
from pathlib import Path

import cv2
import numpy as np

from sky_filter_core import (
    HUE_SHIFT,
    MASK_BLUR_SIGMA,
    SAT_SCALE,
    VAL_SCALE,
    apply_unified_sky_filter,
    soft_mask,
)

FRAME = Path("output/sky-filter/frame_raw.jpg")
MASK = Path("output/sky-filter/mask_yolo.png")
OUT = Path("output/sky-filter/frame_after_yolo.jpg")


def main() -> None:
    frame = cv2.imread(str(FRAME))
    mask_u8 = cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask_u8 is None:
        raise SystemExit(f"需要 {FRAME} 和 {MASK} — 先跑 run_yolo_sky_mask.py")
    if frame.shape[:2] != mask_u8.shape[:2]:
        raise SystemExit("frame 与 mask 尺寸不一致")

    m = soft_mask(mask_u8, MASK_BLUR_SIGMA)
    after = apply_unified_sky_filter(frame, m, HUE_SHIFT, SAT_SCALE, VAL_SCALE)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), after)
    print(f"mask {int(np.count_nonzero(mask_u8))}px")
    print(f"hue{HUE_SHIFT} sat×{SAT_SCALE} val×{VAL_SCALE} blurσ={MASK_BLUR_SIGMA}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
