#!/usr/bin/env python3
"""Step 6 B1 · OpenCV 窗口伪流（YOLO mask + unified 滤镜）

用法:
  python scripts/run_sky_filter_preview.py
  python scripts/run_sky_filter_preview.py --mask-every 1
  python scripts/run_sky_filter_preview.py --no-filter

按键: q / Esc 退出
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from sky_filter_core import (
    MASK_BLUR_SIGMA,
    apply_unified_sky_filter,
    extend_sky_mask_to_top,
    soft_mask,
)

WEIGHTS = Path("weights/sky-seg.pt")
VIDEO = Path("data/sky-filter/pexels-15982565.mp4")
SKY_CLASS = 0
WINDOW = "sky-filter preview"


def yolo_sky_mask(model: YOLO, frame: np.ndarray, sky_class: int = SKY_CLASS) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for r in model(frame, verbose=False):
        if r.masks is None:
            continue
        for seg, cls in zip(r.masks.data, r.boxes.cls):
            if int(cls) != sky_class:
                continue
            m = seg.cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = np.maximum(mask, (m > 0.5).astype(np.uint8) * 255)
    return extend_sky_mask_to_top(mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step6 B1 YOLO mask + unified 滤镜 · 窗口循环预览")
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument(
        "--mask-every",
        type=int,
        default=5,
        help="每 N 帧重算 YOLO mask，中间帧复用（减闪烁可设 1）",
    )
    parser.add_argument("--no-filter", action="store_true", help="只循环播原片，不做滤镜")
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="左右对比显示（原片 | 滤镜后）",
    )
    args = parser.parse_args()

    if not os.environ.get("DISPLAY"):
        raise SystemExit("无 DISPLAY — B1 需要桌面环境；无 GUI 请走 B2 go2rtc")

    if not args.no_filter and not args.weights.exists():
        raise SystemExit(f"缺 {args.weights}")
    if not args.video.exists():
        raise SystemExit(f"缺 {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"打不开 {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay_ms = max(1, int(1000.0 / fps))

    model = YOLO(str(args.weights)) if not args.no_filter else None
    last_mask: np.ndarray | None = None
    frame_idx = 0
    t0 = time.perf_counter()
    shown = 0

    print(f"playing {args.video}  loop  q/Esc quit")
    if args.no_filter:
        print("mode: raw (no filter)")
    else:
        print(f"mode: yolo + unified  mask-every={args.mask_every}")

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            last_mask = None
            continue

        if args.no_filter:
            display = frame
        else:
            assert model is not None
            if frame_idx % args.mask_every == 0 or last_mask is None:
                last_mask = yolo_sky_mask(model, frame)
            m = soft_mask(last_mask, MASK_BLUR_SIGMA)
            after = apply_unified_sky_filter(frame, m)
            display = np.hstack([frame, after]) if args.side_by_side else after

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

        frame_idx += 1
        shown += 1
        if shown % 60 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {shown} frames  {shown / elapsed:.1f} fps")

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t0
    print(f"done {shown} frames in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
