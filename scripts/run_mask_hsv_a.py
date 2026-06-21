#!/usr/bin/env python3
"""§2.8.2 A 轮：inRange → morph → A′ 手动 exclude → mask_hsv_a.png"""
import json
from pathlib import Path

import cv2
import numpy as np

from sky_horizon import HORIZON_GAP_ROWS, trim_mask_a_by_horizon

FRAME = Path("output/sky-filter/frame_raw.jpg")
THRESH = Path("output/sky-filter/hsv_threshold_a.json")
EXCLUDE_JSON = Path("output/sky-filter/mask_exclude.json")
OUT_A = Path("output/sky-filter/mask_hsv_a.png")  # 勿用 jpg，压缩会让 exclude 边缘漏黄
OUT_A_OVERLAY = Path("output/sky-filter/mask_hsv_a_overlay.jpg")
CLAMP_H_LOWER = 92
Y_HOR_SMOOTH = 7


def load_exclude_rects(path: Path) -> list[tuple[int, int, int, int]]:
    if not path.exists():
        return []
    ex = json.loads(path.read_text())
    raw = ex.get("rects", [])
    if raw and not isinstance(raw[0], (list, tuple)):
        raise SystemExit(
            "mask_exclude.json 格式错：rects 须为 [[x1,y1,x2,y2], ...]，"
            f"不能写成 [x1,y1,x2,y2]。当前: {raw!r}"
        )
    rects = []
    for i, item in enumerate(raw):
        if not item:
            continue  # [] 或空：本轮不扣
        if not isinstance(item, (list, tuple)):
            raise SystemExit(
                f"rects[{i}] 须为 [x1,y1,x2,y2]，收到: {item!r}"
            )
        if len(item) != 4:
            raise SystemExit(f"rects[{i}] 须 4 个数 [x1,y1,x2,y2]，收到: {item!r}")
        rects.append(tuple(int(v) for v in item))
    return rects


def main() -> None:
    cfg = json.loads(THRESH.read_text())
    lower = np.array(cfg["lower"], dtype=np.uint8)
    upper = np.array(cfg["upper"], dtype=np.uint8)
    if CLAMP_H_LOWER is not None:
        lower[0] = max(lower[0], CLAMP_H_LOWER)
    upper[2] = 255

    frame = cv2.imread(str(FRAME))
    if frame is None:
        raise SystemExit(f"读不到 {FRAME}")
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    roi = np.zeros((h, w), np.uint8)
    roi[: int(h * 0.65), :] = 255
    mask = cv2.bitwise_and(mask, roi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)

    rects = load_exclude_rects(EXCLUDE_JSON)
    for x1, y1, x2, y2 in rects:
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        mask[y1:y2, x1:x2] = 0
    if rects:
        print("A′ manual exclude:", len(rects), "rect(s) from", EXCLUDE_JSON)

    mask = trim_mask_a_by_horizon(
        mask, margin=0, smooth=Y_HOR_SMOOTH, gap_rows=HORIZON_GAP_ROWS
    )
    print(f"A″ horizon trim: gap_rows={HORIZON_GAP_ROWS}, smooth={Y_HOR_SMOOTH}")

    OUT_A.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_A), mask)
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 255)
    cv2.imwrite(str(OUT_A_OVERLAY), cv2.addWeighted(frame, 0.7, overlay, 0.3, 0))
    print("A 轮锁定", OUT_A, "+ overlay", OUT_A_OVERLAY)
    print("lower", lower.tolist(), "upper", upper.tolist())
    if rects:
        print("→ 改 exclude 后请再跑: python scripts/run_mask_hsv_b.py")
    else:
        print("→ 满意 A 后跑: python scripts/run_mask_hsv_b.py")


if __name__ == "__main__":
    main()
