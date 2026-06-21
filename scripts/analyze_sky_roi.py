#!/usr/bin/env python3
"""ROI 天空紫带分析 · 从 (x0,y0) 到右上角，不跑全图

默认 ROI：用户标注 (858,6) 起，至画面右上；y 向下延伸 --roi-h 行便于看横带上下文。

用法:
  python scripts/analyze_sky_roi.py
  python scripts/analyze_sky_roi.py --video data/sky-filter/pexels-15982565.mp4 --at 2.0
  python scripts/analyze_sky_roi.py --after-mp4 output/sky-filter/after_10s.mp4 --at 2.0
  python scripts/analyze_sky_roi.py --x0 858 --y0 0 --roi-h 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from sky_filter_core import (
    HUE_FADE_V0,
    HUE_FADE_V1,
    HUE_SHIFT_MIN_S,
    MASK_BLUR_SIGMA,
    apply_unified_sky_filter,
    extend_sky_mask_to_top,
    soft_mask,
)

DEFAULT_FRAME = Path("output/sky-filter/frame_raw.jpg")
DEFAULT_MASK = Path("output/sky-filter/mask_yolo.png")
DEFAULT_WEIGHTS = Path("weights/sky-seg.pt")
OUT_DIR = Path("output/sky-filter/debug/roi_analysis")
SKY_CLASS = 0

# 韩师傅标注：紫带附近起点
DEFAULT_X0 = 858
DEFAULT_Y0 = 0
DEFAULT_ROI_H = 80


def load_frame(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.after_mp4 is not None:
        cap = cv2.VideoCapture(str(args.after_mp4))
        cap.set(cv2.CAP_PROP_POS_MSEC, args.at * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"读不到 {args.after_mp4} @ {args.at}s")
        return frame, f"after_mp4@{args.at}s"

    if args.video is not None:
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_MSEC, args.at * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"读不到 {args.video} @ {args.at}s")
        return frame, f"video@{args.at}s"

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"缺 {args.image}")
    return frame, str(args.image)


def yolo_sky_mask(model: YOLO, frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for r in model(frame, verbose=False):
        if r.masks is None:
            continue
        for seg, cls in zip(r.masks.data, r.boxes.cls):
            if int(cls) != SKY_CLASS:
                continue
            m = seg.cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = np.maximum(mask, (m > 0.5).astype(np.uint8) * 255)
    return extend_sky_mask_to_top(mask)


def load_mask(args: argparse.Namespace, frame: np.ndarray) -> np.ndarray:
    if args.mask is not None:
        mask_u8 = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            raise SystemExit(f"缺 {args.mask}")
        return mask_u8
    if args.yolo:
        if not args.weights.exists():
            raise SystemExit(f"缺 {args.weights}")
        return yolo_sky_mask(YOLO(str(args.weights)), frame)
    mask_u8 = cv2.imread(str(DEFAULT_MASK), cv2.IMREAD_GRAYSCALE)
    if mask_u8 is None:
        raise SystemExit(f"缺 {DEFAULT_MASK}，或加 --yolo / --mask")
    return mask_u8


def roi_slices(h: int, w: int, x0: int, y0: int, roi_h: int) -> tuple[slice, slice]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = min(h, y0 + roi_h)
    return slice(y0, y1), slice(x0, w)


def compute_effect_w(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s, v = hsv[..., 1], hsv[..., 2]
    sat_w = np.clip((s - HUE_SHIFT_MIN_S) / 25.0, 0.0, 1.0)
    bright_w = np.clip((HUE_FADE_V1 - v) / (HUE_FADE_V1 - HUE_FADE_V0), 0.0, 1.0)
    return sat_w * bright_w


def purple_mask(bgr: np.ndarray) -> np.ndarray:
    r, g, b = bgr[..., 2].astype(np.float32), bgr[..., 1], bgr[..., 0]
    return (r > g + 3) & (b > g) & (b > 80)


def row_stats(bgr: np.ndarray, sky: np.ndarray, soft_m: np.ndarray | None = None) -> dict[str, np.ndarray]:
    h = bgr.shape[0]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ew = compute_effect_w(bgr)
    purp = purple_mask(bgr)
    rows = np.arange(h)
    out: dict[str, np.ndarray] = {"y": rows}
    channels: list[tuple[str, np.ndarray]] = [
        ("H", hsv[..., 0]),
        ("S", hsv[..., 1]),
        ("V", hsv[..., 2]),
        ("effect_w", ew),
        ("mask_cov", sky.astype(np.float32) * 100.0),
        ("purple_pct", purp.astype(np.float32) * 100.0),
    ]
    if soft_m is not None:
        channels.append(("soft_m", soft_m * 100.0))
    for name, arr in channels:
        vals = np.zeros(h, np.float32)
        for y in range(h):
            sel = np.ones(arr.shape[1], dtype=bool)  # ROI 内整行，不限 mask
            vals[y] = arr[y][sel].mean()
        out[name] = vals
    return out


def save_row_profile(path: Path, raw_s: dict, aft_s: dict, delta_r: np.ndarray) -> None:
    lines = [
        "y | raw_H raw_S raw_V raw_ew mask% soft% | aft purple% | dR",
        "-" * 78,
    ]
    for i, y in enumerate(raw_s["y"]):
        soft = raw_s.get("soft_m", np.zeros_like(raw_s["H"]))[i]
        lines.append(
            f"{y:3d} | "
            f"{raw_s['H'][i]:5.1f} {raw_s['S'][i]:5.1f} {raw_s['V'][i]:5.1f} {raw_s['effect_w'][i]:4.2f} "
            f"{raw_s['mask_cov'][i]:4.0f} {soft:4.0f} | "
            f"{aft_s['purple_pct'][i]:4.2f} | {delta_r[i]:+5.1f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def heatmap_u8(values: np.ndarray, min_width: int = 64) -> np.ndarray:
    v = np.clip(values, 0, 1)
    gray = (v * 255).astype(np.uint8)
    if gray.ndim == 1:
        gray = np.tile(gray[:, None], (1, min_width))
    if gray.shape[1] < min_width:
        gray = cv2.resize(gray, (min_width, gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def main() -> None:
    parser = argparse.ArgumentParser(description="右上 ROI 天空紫带分析")
    parser.add_argument("--image", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--video", type=Path, default=None, help="原片 mp4")
    parser.add_argument("--after-mp4", type=Path, default=None, help="滤镜后 mp4（只分析不再跑滤镜）")
    parser.add_argument("--at", type=float, default=0.0, help="视频时间点（秒）")
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--yolo", action="store_true", help="本帧重跑 YOLO mask")
    parser.add_argument("--x0", type=int, default=DEFAULT_X0, help="ROI 左缘（默认 858）")
    parser.add_argument("--y0", type=int, default=DEFAULT_Y0, help="ROI 顶缘（默认 0）")
    parser.add_argument("--roi-h", type=int, default=DEFAULT_ROI_H, help="ROI 高度（像素）")
    parser.add_argument("--anchor-y", type=int, default=6, help="标注 y，报告里高亮该行")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    frame, src_label = load_frame(args)
    h, w = frame.shape[:2]
    ys, xs = roi_slices(h, w, args.x0, args.y0, args.roi_h)

    from_after = args.after_mp4 is not None
    if from_after:
        raw = frame  # 无法分离，diff 无意义
        after = frame
        mask_u8 = load_mask(args, frame) if args.mask or not args.yolo else np.ones((h, w), np.uint8) * 255
        if args.yolo:
            mask_u8 = load_mask(args, frame)
    else:
        mask_u8 = load_mask(args, frame)
        m = soft_mask(mask_u8, MASK_BLUR_SIGMA)
        raw = frame
        after = apply_unified_sky_filter(frame, m)
        soft_roi = m[ys, xs]

    roi_raw = raw[ys, xs].copy()
    roi_after = after[ys, xs].copy()
    roi_mask = mask_u8[ys, xs] > 128
    sky_frac = roi_mask.mean() * 100

    diff = cv2.absdiff(roi_raw, roi_after).astype(np.float32)
    diff_amp = np.clip(diff * 4, 0, 255).astype(np.uint8)

    # R 通道增量（滤镜后 - 原片），只在 mask 内
    dr = roi_after[:, :, 2].astype(np.float32) - roi_raw[:, :, 2].astype(np.float32)
    dr_row = dr.mean(axis=1)

    soft_roi = None if from_after else soft_roi
    raw_row = row_stats(roi_raw, roi_mask, soft_roi)
    aft_row = row_stats(roi_after, roi_mask, soft_roi)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = src_label.replace("/", "_").replace("@", "_")
    cv2.imwrite(str(args.out_dir / f"roi_raw_{tag}.jpg"), roi_raw)
    cv2.imwrite(str(args.out_dir / f"roi_after_{tag}.jpg"), roi_after)
    cv2.imwrite(str(args.out_dir / f"roi_diff_x4_{tag}.jpg"), diff_amp)
    cv2.imwrite(str(args.out_dir / f"roi_effect_w_{tag}.png"), heatmap_u8(compute_effect_w(roi_raw)))
    cv2.imwrite(str(args.out_dir / f"roi_purple_{tag}.png"), (purple_mask(roi_after).astype(np.uint8) * 255))

    if not from_after:
        save_row_profile(args.out_dir / f"roi_rows_{tag}.txt", raw_row, aft_row, dr_row)

    # 控制台摘要
    print(f"source: {src_label}")
    print(f"ROI: x=[{xs.start},{xs.stop}) y=[{ys.start},{ys.stop})  sky={sky_frac:.1f}%")
    print(f"saved → {args.out_dir}/")
    if from_after:
        print("(after-mp4 模式：仅 ROI 截图 + purple map，无 raw/filter 对比)")
        return

    ay = args.anchor_y - ys.start
    if 0 <= ay < roi_raw.shape[0]:
        print(f"\nanchor y={args.anchor_y} (ROI 内 row {ay}):")
        print(
            f"  raw  H/S/V/ew = {raw_row['H'][ay]:.1f} / {raw_row['S'][ay]:.1f} / {raw_row['V'][ay]:.1f} / {raw_row['effect_w'][ay]:.3f}"
        )
        print(f"  mask_cov={raw_row['mask_cov'][ay]:.0f}%  soft_m={raw_row.get('soft_m', np.zeros(1))[ay]:.1f}%")
        print(f"  ΔR={dr_row[ay]:+.1f}  purple%={aft_row['purple_pct'][ay]:.2f}")
        if raw_row["mask_cov"][ay] < 50:
            print("  ⚠ 该行 hard mask 覆盖低 — 若有变色，可能是 soft mask blur 从下方 bleed 上来")

    peak_y = int(aft_row["y"][np.argmax(aft_row["purple_pct"])])
    print(f"\nROI 内 purple_pct 最高行: y={peak_y + ys.start} ({aft_row['purple_pct'].max():.2f}%)")
    peak_dr = int(aft_row["y"][np.argmax(np.abs(dr_row))])
    print(f"ROI 内 |ΔR| 最大行: y={peak_dr + ys.start} (ΔR={dr_row[peak_dr]:+.1f})")

    print(f"\n逐行表 → roi_rows_{tag}.txt")


if __name__ == "__main__":
    main()
