#!/usr/bin/env python3
"""Step 3 · 分路蓝天滤镜：mask_a 提蓝 + mask_c / mask_a 低 S 洗云

v2：mask_a 内低饱和走 cloud 路（左侧灰蒙天）；Lab 去色偏 + 暗云自适应提亮
见 .internal/docs/Step2-C轮云取色与Step3分路滤镜-2026-06-20.md §4
"""
from pathlib import Path

import cv2
import numpy as np

FRAME = Path("output/sky-filter/frame_raw.jpg")
MASK_SKY = Path("output/sky-filter/mask_hsv_a.png")
MASK_CLOUD = Path("output/sky-filter/mask_hsv_c.png")
OUT_DIR = Path("output/sky-filter")
OUT_BEFORE = OUT_DIR / "frame_before.jpg"
OUT_AFTER = OUT_DIR / "frame_after.jpg"

MASK_BLUR_SIGMA = 3
# 仅 mask_a 里 S 极低才走洗云；其余低 S 仍跟 unified 一样偏蓝（更自然）
GRAY_IN_A_SAT = 12

SKY_HUE_SHIFT = -8
SKY_SAT_SCALE = 1.25
SKY_VAL_SCALE = 1.0

CLOUD_SAT_SCALE = 0.95          # 略收饱和，别压太狠
CLOUD_LAB_CHROMA_PULL = 0.28
CLOUD_VAL_TARGET = 192
CLOUD_VAL_SCALE_MAX = 1.12      # 暗云提亮上限（原 1.35 易假）

# 浅灰云 mask_c 内：轻微微拉 L，避免 unified 那种脏，也避免 v2 过曝
PALE_CLOUD_MAX_S = 10
PALE_CLOUD_MIN_V = 165
PALE_CLOUD_L_TARGET = 228
PALE_CLOUD_LIFT = 0.10


def soft_mask(mask_u8: np.ndarray, sigma: float) -> np.ndarray:
    m = mask_u8.astype(np.float32) / 255.0
    if sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.clip(m, 0.0, 1.0)


def apply_sky_blue(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + SKY_HUE_SHIFT) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * SKY_SAT_SCALE, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * SKY_VAL_SCALE, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_cloud_clean(bgr: np.ndarray) -> np.ndarray:
    """分两类云：暗云提亮；浅灰云(S极低/V已高)往白拉 L，不压饱和"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s, v = hsv[..., 1], hsv[..., 2]

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    pale = (s <= PALE_CLOUD_MAX_S) & (v >= PALE_CLOUD_MIN_V)
    dark = v < 160

    val_scale = np.clip(CLOUD_VAL_TARGET / np.maximum(v, 1.0), 1.0, CLOUD_VAL_SCALE_MAX)
    L = np.where(dark, np.clip(L * val_scale, 0, 255), L)
    L = np.where(
        pale,
        np.clip(L + (PALE_CLOUD_L_TARGET - L) * PALE_CLOUD_LIFT, 0, 255),
        L,
    )

    pull = np.where(pale, 0.25, CLOUD_LAB_CHROMA_PULL)
    a = 128 + (a - 128) * (1.0 - pull)
    b = 128 + (b - 128) * (1.0 - pull)
    lab[..., 0], lab[..., 1], lab[..., 2] = L, a, b

    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    hsv2 = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    s2 = hsv2[..., 1]
    # 浅灰云 S 已经贴 0，再 ×0.85 只会更闷
    hsv2[..., 1] = np.where(
        s2 <= PALE_CLOUD_MAX_S,
        s2,
        np.clip(s2 * CLOUD_SAT_SCALE, 0, 255),
    )
    return cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2BGR)


def split_sky_cloud_masks(
    m_a: np.ndarray,
    m_c: np.ndarray,
    sat: np.ndarray,
    gray_sat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """mask_a 高 S → 蓝天；mask_c + mask_a 低 S → 洗云"""
    s_ratio = np.clip(sat / gray_sat, 0.0, 1.0)
    m_blue = m_a * s_ratio
    m_gray_a = m_a * (1.0 - s_ratio)
    m_cloud = np.clip(m_c + m_gray_a * (1.0 - m_c), 0.0, 1.0)
    return m_blue, m_cloud


def blend_region(
    base: np.ndarray,
    effect: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    m3 = mask[..., None]
    out = base.astype(np.float32) * (1.0 - m3) + effect.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    frame = cv2.imread(str(FRAME))
    mask_sky_u8 = cv2.imread(str(MASK_SKY), cv2.IMREAD_GRAYSCALE)
    mask_cloud_u8 = cv2.imread(str(MASK_CLOUD), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask_sky_u8 is None or mask_cloud_u8 is None:
        raise SystemExit(
            f"需要 {FRAME}、{MASK_SKY}、{MASK_CLOUD} — 先跑 A 与 C 轮脚本"
        )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    m_a = soft_mask(mask_sky_u8, MASK_BLUR_SIGMA)
    m_c = soft_mask(mask_cloud_u8, MASK_BLUR_SIGMA)
    m_blue, m_cloud = split_sky_cloud_masks(m_a, m_c, sat, GRAY_IN_A_SAT)

    sky_fx = apply_sky_blue(frame)
    cloud_fx = apply_cloud_clean(frame)

    after = blend_region(frame, sky_fx, m_blue)
    after = blend_region(after, cloud_fx, m_cloud)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_BEFORE), frame)
    cv2.imwrite(str(OUT_AFTER), after)

    gray_a_px = int(np.count_nonzero((mask_sky_u8 > 0) & (sat < GRAY_IN_A_SAT)))
    print(f"mask_a 内低S(走洗云) ≈ {gray_a_px}px | mask_c {np.count_nonzero(mask_cloud_u8)}px")
    print(f"GRAY_IN_A_SAT={GRAY_IN_A_SAT} | pale lift={PALE_CLOUD_LIFT}→{PALE_CLOUD_L_TARGET}")
    print("saved", OUT_AFTER, "← 分路 a+c（整片旧方案见 run_sky_filter_unified.py）")


if __name__ == "__main__":
    main()
