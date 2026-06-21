#!/usr/bin/env python3
"""Step 5 · mp4 蓝天滤镜对比片（YOLO mask + unified）

用法:
  python scripts/run_sky_filter_video.py
  python scripts/run_sky_filter_video.py --seconds 10 --mask-every 5
  python scripts/run_sky_filter_video.py --side-by-side
"""
from __future__ import annotations

import argparse
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
OUT_DIR = Path("output/sky-filter")
SKY_CLASS = 0


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
    parser = argparse.ArgumentParser(description="Step5 YOLO mask + unified 滤镜 → before/after mp4")
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seconds", type=float, default=10.0, help="处理前 N 秒")
    parser.add_argument(
        "--mask-every",
        type=int,
        default=5,
        help="每 N 帧重算 YOLO mask，中间帧复用（减闪烁可设 1）",
    )
    parser.add_argument("--side-by-side", action="store_true", help="额外输出左右对比 mp4")
    parser.add_argument("--start", type=float, default=0.0, help="从第 N 秒开始读")
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"缺 {args.weights}")
    if not args.video.exists():
        raise SystemExit(f"缺 {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"打不开 {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    max_frames = max(1, int(fps * args.seconds))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{int(args.seconds)}s" if args.seconds == int(args.seconds) else f"{args.seconds}s"
    before_path = args.out_dir / f"before_{stem}.mp4"
    after_path = args.out_dir / f"after_{stem}.mp4"
    sbs_path = args.out_dir / f"compare_{stem}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_before = cv2.VideoWriter(str(before_path), fourcc, fps, (w, h))
    w_after = cv2.VideoWriter(str(after_path), fourcc, fps, (w, h))
    w_sbs = (
        cv2.VideoWriter(str(sbs_path), fourcc, fps, (w * 2, h)) if args.side_by_side else None
    )

    model = YOLO(str(args.weights))
    last_mask: np.ndarray | None = None
    n = 0
    t0 = time.perf_counter()

    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if n % args.mask_every == 0 or last_mask is None:
            last_mask = yolo_sky_mask(model, frame)
        m = soft_mask(last_mask, MASK_BLUR_SIGMA)
        after = apply_unified_sky_filter(frame, m)
        w_before.write(frame)
        w_after.write(after)
        if w_sbs is not None:
            w_sbs.write(np.hstack([frame, after]))
        n += 1
        if n % 30 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {n}/{max_frames} frames  {n / elapsed:.1f} fps")

    cap.release()
    w_before.release()
    w_after.release()
    if w_sbs is not None:
        w_sbs.release()

    elapsed = time.perf_counter() - t0
    print(f"done {n} frames in {elapsed:.1f}s ({n / max(elapsed, 0.001):.1f} fps effective)")
    print("saved", before_path)
    print("saved", after_path)
    if w_sbs is not None:
        print("saved", sbs_path)


if __name__ == "__main__":
    main()
