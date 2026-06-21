#!/usr/bin/env python3
"""A 轮取色实验：加点 → 扩 hsv_threshold → 看 frame overlay（不写回正式 JSON）

用法:
  python scripts/try_a_sky_sample.py
  python scripts/try_a_sky_sample.py --x 668 --y 78 --name sky_blue2
  python scripts/try_a_sky_sample.py --write   # 满意后写入 hsv_threshold_a_trial.json

默认点：frame_0001 @ (668,78) — 韩师傅标的「很蓝天」
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from generate_pseudo_sky_labels import (
    CLAMP_H_LOWER,
    compute_mask_a,
    compute_mask_b,
    save_overlay,
    save_step_outputs,
)
from sky_horizon import HORIZON_GAP_ROWS

BASE_THRESH = Path("output/sky-filter/hsv_threshold_a.json")
OUT_THRESH = Path("output/sky-filter/hsv_threshold_a_trial.json")
DEFAULT_IMAGE = Path("data/sky-seg/images/train/frame_0001.jpg")
OUT_DIR = Path("output/sky-seg-pseudo/trial-a-sample")

# 相对采样点的扩框（OpenCV HSV）；S 略宽以包住饱和蓝天
MARGIN_H = 12
MARGIN_S = 25
MARGIN_V = 35


def read_pixel_hsv(frame: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = frame.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y, x].astype(np.int32)


def expand_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    sample: np.ndarray,
    mh: int = MARGIN_H,
    ms: int = MARGIN_S,
    mv: int = MARGIN_V,
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.minimum(lower.astype(np.int32), sample - np.array([mh, ms, mv], dtype=np.int32))
    hi = np.maximum(upper.astype(np.int32), sample + np.array([mh, ms, mv], dtype=np.int32))
    lo = np.clip(lo, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    hi = np.clip(hi, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    if CLAMP_H_LOWER is not None:
        lo[0] = max(int(lo[0]), CLAMP_H_LOWER)
    hi[2] = 255
    return lo, hi


def in_current_range(sample: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(np.all(sample >= lower) and np.all(sample <= upper))


def main() -> None:
    parser = argparse.ArgumentParser(description="A 轮单点取色实验")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--x", type=int, default=668)
    parser.add_argument("--y", type=int, default=78)
    parser.add_argument("--name", type=str, default="sky_blue2")
    parser.add_argument("--write", action="store_true", help="写入 hsv_threshold_a_trial.json")
    args = parser.parse_args()

    if not BASE_THRESH.exists():
        raise SystemExit(f"缺 {BASE_THRESH}")
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"读不到 {args.image}")

    cfg = json.loads(BASE_THRESH.read_text(encoding="utf-8"))
    lower = np.array(cfg["lower"], dtype=np.uint8)
    upper = np.array(cfg["upper"], dtype=np.uint8)
    if CLAMP_H_LOWER is not None:
        lower[0] = max(lower[0], CLAMP_H_LOWER)
    upper[2] = 255

    sample = read_pixel_hsv(frame, args.x, args.y)
    picked = in_current_range(sample, lower, upper)

    new_lower, new_upper = expand_bounds(lower, upper, sample)
    stem = Path(args.image).stem

    print(f"image {args.image}  sample {args.name} @ ({args.x},{args.y})")
    print(f"  OpenCV HSV = {sample.tolist()}  (H×2≈{sample[0]*2:.0f} 若工具用 0–360°)")
    print(f"  当前 A 范围已包住此点: {'是' if picked else '否'}")
    if not picked:
        for i, ch in enumerate("HSV"):
            if sample[i] < lower[i] or sample[i] > upper[i]:
                print(f"    超出 {ch}: sample={sample[i]} 不在 [{lower[i]}, {upper[i]}]")
    print(f"  原 lower/upper: {lower.tolist()} / {upper.tolist()}")
    print(f"  新 lower/upper: {new_lower.tolist()} / {new_upper.tolist()}")

    mask_a_old = compute_mask_a(frame, lower, upper)
    mask_b_old = compute_mask_b(frame, mask_a_old)
    mask_old = cv2.bitwise_or(mask_a_old, mask_b_old)

    mask_a_new = compute_mask_a(frame, new_lower, new_upper)
    mask_b_new = compute_mask_b(frame, mask_a_new)
    mask_new = cv2.bitwise_or(mask_a_new, mask_b_new)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{stem}_{args.name}"
    save_step_outputs(frame, mask_a_new, mask_b_new, mask_new, tag, OUT_DIR)
    save_overlay(frame, mask_old, OUT_DIR / f"{tag}_overlay_before.jpg")
    save_overlay(frame, mask_new, OUT_DIR / f"{tag}_overlay_after.jpg")

    a0, a1 = int(np.count_nonzero(mask_a_old)), int(np.count_nonzero(mask_a_new))
    s0, s1 = int(np.count_nonzero(mask_old)), int(np.count_nonzero(mask_new))
    print(f"  A px: {a0} → {a1} ({a1 - a0:+d})")
    print(f"  final px: {s0} → {s1} ({s1 - s0:+d})")
    print(f"  → {OUT_DIR}/{tag}_overlay.jpg  (及 mask_a/b before/after)")

    if args.write:
        payload = {
            **cfg,
            "lower": new_lower.tolist(),
            "upper": new_upper.tolist(),
            "sky_labels": list(cfg.get("sky_labels", [])) + [args.name],
            "trial_from": str(args.image),
            "trial_xy": [args.x, args.y],
        }
        OUT_THRESH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  已写 {OUT_THRESH}（未动 {BASE_THRESH.name}）")


if __name__ == "__main__":
    main()
