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

WEIGHTS = Path("weights/sky-seg.pt")
VIDEO = Path("data/sky-filter/pexels-15982565.mp4")
OUT_DIR = Path("output/sky-filter")
SKY_CLASS = 0

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
    return mask


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
