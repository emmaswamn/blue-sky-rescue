#!/usr/bin/env python3
"""用 Step 2 A+B′ 批量生成 YOLO sky 伪标注（mask → 多边形 txt）

若 overlay 海天 OK，可跳过 Roboflow 手标，直接训 YOLO。
见 .internal/docs/Step4-自训sky-seg-抽帧标注训练-2026-06-21.md §4b

用法:
  python scripts/generate_pseudo_sky_labels.py --limit 5
  python scripts/generate_pseudo_sky_labels.py          # 全部 train 图
  python scripts/generate_pseudo_sky_labels.py --only frame_0001 --save-steps  # 单帧 + A 轮中间步
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sky_horizon import HORIZON_GAP_ROWS, allow_from_mask_a_columns, trim_mask_a_by_horizon

THRESH = Path("output/sky-filter/hsv_threshold_a.json")
EXCLUDE_JSON = Path("output/sky-filter/mask_exclude.json")
IMAGES_DIR = Path("data/sky-seg/images/train")
LABELS_DIR = Path("data/sky-seg/labels/train")
OVERLAY_DIR = Path("output/sky-seg-pseudo")

CLAMP_H_LOWER = 92
ROI_HEIGHT = 0.65
HORIZON_MARGIN = 5
Y_HOR_SMOOTH = 7
LOW_SAT_MAX_S = 28
LOW_SAT_MIN_V = 120
CLOSE_B_ITER = 1
MIN_CONTOUR_AREA = 500
CLASS_ID = 0


def load_exclude_rects(path: Path) -> list[tuple[int, int, int, int]]:
    if not path.exists():
        return []
    ex = json.loads(path.read_text(encoding="utf-8"))
    rects = []
    for item in ex.get("rects", []):
        if item and len(item) == 4:
            rects.append(tuple(int(v) for v in item))
    return rects


def compute_mask_a(frame: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    roi = np.zeros((h, w), np.uint8)
    roi[: int(h * ROI_HEIGHT), :] = 255
    mask = cv2.bitwise_and(mask, roi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)
    for x1, y1, x2, y2 in load_exclude_rects(EXCLUDE_JSON):
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        mask[y1:y2, x1:x2] = 0
    return trim_mask_a_by_horizon(mask, margin=0, smooth=Y_HOR_SMOOTH, gap_rows=HORIZON_GAP_ROWS)


def compute_mask_b(frame: np.ndarray, mask_a: np.ndarray) -> np.ndarray:
    allow = allow_from_mask_a_columns(
        mask_a, HORIZON_MARGIN, Y_HOR_SMOOTH, gap_rows=HORIZON_GAP_ROWS
    )
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask_b = (((s <= LOW_SAT_MAX_S) & (v >= LOW_SAT_MIN_V)).astype(np.uint8) * 255)
    mask_b = cv2.bitwise_and(mask_b, allow)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if CLOSE_B_ITER:
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, k, CLOSE_B_ITER)
        mask_b = cv2.bitwise_and(mask_b, allow)
    return mask_b


def compute_mask_final(frame: np.ndarray, mask_a: np.ndarray) -> np.ndarray:
    return cv2.bitwise_or(mask_a, compute_mask_b(frame, mask_a))


def mask_to_yolo_lines(mask: np.ndarray, class_id: int = CLASS_ID) -> list[str]:
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        eps = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        pts = np.clip(pts, 0.0, 1.0)
        coords = " ".join(f"{p:.6f}" for xy in pts for p in xy)
        lines.append(f"{class_id} {coords}")
    return lines


def save_overlay(
    frame: np.ndarray, mask: np.ndarray, path: Path, color: tuple[int, int, int] = (0, 255, 255)
) -> None:
    vis = frame.copy()
    vis[mask > 0] = color
    cv2.imwrite(str(path), cv2.addWeighted(frame, 0.7, vis, 0.3, 0))


def save_step_outputs(
    frame: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    mask_final: np.ndarray,
    stem: str,
    out_dir: Path,
) -> None:
    """A 轮 mask + overlay；B 轮、最终 overlay 分色，方便对照调参。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_mask_a.png"), mask_a)
    save_overlay(frame, mask_a, out_dir / f"{stem}_mask_a_overlay.jpg")
    save_overlay(frame, mask_b, out_dir / f"{stem}_mask_b_overlay.jpg", color=(255, 0, 255))
    save_overlay(frame, mask_final, out_dir / f"{stem}_overlay.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step2 mask → YOLO sky 伪标注")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张（试跑用 5）")
    parser.add_argument("--only", type=str, default="", help="只处理指定 stem，如 frame_0001")
    parser.add_argument(
        "--save-steps",
        action="store_true",
        help="保存 A/B 中间步：mask_a.png、mask_a_overlay、mask_b_overlay、overlay",
    )
    parser.add_argument("--images", type=Path, default=IMAGES_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_DIR)
    parser.add_argument("--overlays", type=Path, default=OVERLAY_DIR)
    parser.add_argument(
        "--thresh",
        type=Path,
        default=THRESH,
        help="HSV 阈值 JSON（默认 hsv_threshold_a.json；实验可用 *_trial.json）",
    )
    args = parser.parse_args()

    if not args.thresh.exists():
        raise SystemExit(f"缺 {args.thresh} — 先完成 Step 2 标定")
    cfg = json.loads(args.thresh.read_text(encoding="utf-8"))
    lower = np.array(cfg["lower"], dtype=np.uint8)
    upper = np.array(cfg["upper"], dtype=np.uint8)
    if CLAMP_H_LOWER is not None:
        lower[0] = max(lower[0], CLAMP_H_LOWER)
    upper[2] = 255

    images = sorted(args.images.glob("*.jpg"))
    if not images:
        raise SystemExit(f"无图片: {args.images}")
    if args.only:
        images = [p for p in images if p.stem == args.only]
        if not images:
            raise SystemExit(f"找不到 stem={args.only!r} in {args.images}")
    elif args.limit > 0:
        images = images[: args.limit]

    args.labels.mkdir(parents=True, exist_ok=True)
    args.overlays.mkdir(parents=True, exist_ok=True)

    ok, empty = 0, 0
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("skip read fail", img_path.name)
            continue
        mask_a = compute_mask_a(frame, lower, upper)
        mask_b = compute_mask_b(frame, mask_a)
        mask = cv2.bitwise_or(mask_a, mask_b)
        lines = mask_to_yolo_lines(mask)
        label_path = args.labels / f"{img_path.stem}.txt"
        if lines:
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            ok += 1
        else:
            label_path.write_text("", encoding="utf-8")
            empty += 1
        if args.save_steps:
            save_step_outputs(frame, mask_a, mask_b, mask, img_path.stem, args.overlays)
        else:
            save_overlay(frame, mask, args.overlays / f"{img_path.stem}_overlay.jpg")
        sky_px = int(np.count_nonzero(mask))
        a_px = int(np.count_nonzero(mask_a))
        b_only_px = int(np.count_nonzero(cv2.bitwise_and(mask_b, cv2.bitwise_not(mask_a))))
        print(
            f"{img_path.name}: sky {sky_px}px (A {a_px} + B-only {b_only_px}), "
            f"poly {len(lines)} → {label_path.name}"
        )

    print(f"done {ok} labeled, {empty} empty | overlays → {args.overlays}")


if __name__ == "__main__":
    main()
