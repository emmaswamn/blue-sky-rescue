#!/usr/bin/env python3
"""Step 4 · YOLO sky-seg 单帧推理 → mask_yolo.png

前置：weights/sky-seg.pt（见 Step4-自训sky-seg-抽帧标注训练-2026-06-21.md）
"""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from sky_filter_core import extend_sky_mask_to_top

WEIGHTS = Path("weights/sky-seg.pt")
FRAME = Path("output/sky-filter/frame_raw.jpg")
OUT = Path("output/sky-filter/mask_yolo.png")
SKY_CLASS = 0


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"缺 {WEIGHTS} — 先完成 Step 4 训练并 cp best.pt")
    frame = cv2.imread(str(FRAME))
    if frame is None:
        raise SystemExit(f"缺 {FRAME}")

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    model = YOLO(str(WEIGHTS))
    for r in model(frame, verbose=False):
        if r.masks is None:
            continue
        for seg, cls in zip(r.masks.data, r.boxes.cls):
            if int(cls) != SKY_CLASS:
                continue
            m = seg.cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = np.maximum(mask, (m > 0.5).astype(np.uint8) * 255)

    mask = extend_sky_mask_to_top(mask)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), mask)
    print("saved", OUT, "sky px", int(np.count_nonzero(mask)))
    print("→ 改 run_sky_filter_unified.py 的 MASK/OUT 或跑 frame_after_yolo")


if __name__ == "__main__":
    main()
