#!/usr/bin/env python3
"""§2.8.2 C 轮：云取色 inRange + mask_a 洞/allow 约束 → mask_hsv_c.png

见 .internal/docs/Step2-C轮云取色与Step3分路滤镜-2026-06-20.md
先跑 run_mask_hsv_a.py；改 CLOUD_SAMPLES 或 hsv_threshold_c.json 后再跑本脚本。
"""
import json
from pathlib import Path

import cv2
import numpy as np

from sky_horizon import HORIZON_GAP_ROWS, allow_from_mask_a_columns

FRAME = Path("output/sky-filter/frame_raw.jpg")
MASK_A = Path("output/sky-filter/mask_hsv_a.png")
THRESH = Path("output/sky-filter/hsv_threshold_c.json")
OUT_C = Path("output/sky-filter/mask_hsv_c.png")
OUT_C_OVERLAY = Path("output/sky-filter/mask_hsv_c_overlay.jpg")

# ---------- §2.5C 云表：在 mask_a overlay 上改 (x,y)，须落在 A 的「洞里」----------
CLOUD_SAMPLES = [
    ("cloud_top", 501, 31),       # 顶区厚云心
    ("cloud_mid", 542, 67),       # 中上薄云
    ("cloud_bright", 682, 164),   # 较亮云块
    ("cloud_edge", 380, 95),      # 云边/丝（可选第 4 点）
]

HORIZON_MARGIN = 5
Y_HOR_SMOOTH = 7
INTERSECT_HOLES = True   # 只在 mask_a 为黑处（洞里补云）
CLOSE_C_ITER = 1
WRITE_THRESH_FROM_SAMPLES = False  # True: 用上面 CLOUD_SAMPLES 覆盖写 JSON


def compute_cloud_lower_upper(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(samples.min(0) - [20, 15, 30], [0, 0, 0])
    upper = np.minimum(samples.max(0) + [20, 30, 15], [179, 255, 255])
    lower[0], upper[0] = 0, 179  # 云 H 离群，不信窄 H 带
    lower[1] = 0
    upper[2] = 255
    return lower.astype(np.uint8), upper.astype(np.uint8)


def write_thresh_from_samples(frame_hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array([frame_hsv[y, x] for _, x, y in CLOUD_SAMPLES])
    lower, upper = compute_cloud_lower_upper(pts)
    payload = {
        "role": "c",
        "round": 1,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "cloud_labels": [n for n, _, _ in CLOUD_SAMPLES],
    }
    THRESH.parent.mkdir(parents=True, exist_ok=True)
    THRESH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("=== C 表采样 ===")
    for name, x, y in CLOUD_SAMPLES:
        print(f"  {name:16s} @ ({x:4d},{y:4d})  HSV={frame_hsv[y, x].tolist()}")
    print("lower", lower.tolist(), "upper", upper.tolist())
    print("saved", THRESH)
    return lower, upper


def main() -> None:
    frame = cv2.imread(str(FRAME))
    mask_a = cv2.imread(str(MASK_A), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask_a is None:
        raise SystemExit(f"需要 {FRAME} 和 {MASK_A} — 先跑 scripts/run_mask_hsv_a.py")

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if WRITE_THRESH_FROM_SAMPLES or not THRESH.exists():
        lower, upper = write_thresh_from_samples(hsv)
    else:
        cfg = json.loads(THRESH.read_text(encoding="utf-8"))
        lower = np.array(cfg["lower"], dtype=np.uint8)
        upper = np.array(cfg["upper"], dtype=np.uint8)
        print("read", THRESH, "lower", lower.tolist(), "upper", upper.tolist())

    allow = allow_from_mask_a_columns(
        mask_a, HORIZON_MARGIN, Y_HOR_SMOOTH, gap_rows=HORIZON_GAP_ROWS
    )
    mask_c = cv2.inRange(hsv, lower, upper)
    mask_c = cv2.bitwise_and(mask_c, allow)
    if INTERSECT_HOLES:
        mask_c = cv2.bitwise_and(mask_c, cv2.bitwise_not(mask_a))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if CLOSE_C_ITER:
        mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_CLOSE, k, CLOSE_C_ITER)
        mask_c = cv2.bitwise_and(mask_c, allow)
        if INTERSECT_HOLES:
            mask_c = cv2.bitwise_and(mask_c, cv2.bitwise_not(mask_a))

    OUT_C.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_C), mask_c)
    overlay = frame.copy()
    overlay[mask_c > 0] = (0, 200, 255)  # 橙青区分 A overlay
    cv2.imwrite(str(OUT_C_OVERLAY), cv2.addWeighted(frame, 0.7, overlay, 0.3, 0))
    px = int(np.count_nonzero(mask_c))
    print(f"C 轮 mask 像素 {px} saved {OUT_C} + overlay {OUT_C_OVERLAY}")
    print("→ Step 3 分路：mask_a 蓝天 + mask_c 洗云（见 C 轮文档 §4）")


if __name__ == "__main__":
    main()
