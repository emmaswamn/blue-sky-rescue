"""Unified sky filter shared by Step 3–6 scripts."""
from __future__ import annotations

import cv2
import numpy as np

MASK_BLUR_SIGMA = 3
HUE_SHIFT = -8
SAT_SCALE = 1.25
VAL_SCALE = 1.0

# S 低于此值不做 H/S 增强（低 S 提饱和会把随机色相放大成紫/品红）
HUE_SHIFT_MIN_S = 30
# V 高于 HUE_FADE_V1 不增强；V0～V1 之间线性衰减（高亮区：顶/右侧浅蓝天）
HUE_FADE_V0 = 200
HUE_FADE_V1 = 245
# shift 后 H 限制在 OpenCV 蓝段，防止 wrap 到 magenta
H_SKY_LO = 98
H_SKY_HI = 118


def extend_sky_mask_to_top(mask_u8: np.ndarray, thresh: int = 128) -> np.ndarray:
    """YOLO seg 常在帧顶留空：按列把首个 sky 像素以上补满，避免顶部未滤镜横带。"""
    out = mask_u8.copy()
    binary = mask_u8 >= thresh
    if not binary.any():
        return out
    first = binary.argmax(axis=0)
    has = binary.any(axis=0)
    rows = np.arange(binary.shape[0])[:, None]
    fill = (rows < first[None, :]) & has[None, :]
    out[fill] = 255
    return out


def soft_mask(mask_u8: np.ndarray, sigma: float = MASK_BLUR_SIGMA) -> np.ndarray:
    m = mask_u8.astype(np.float32) / 255.0
    if sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.clip(m, 0.0, 1.0)


def _suppress_sky_purple(bgr: np.ndarray) -> np.ndarray:
    """蓝空 B≥G≥R；压异常偏紫（R 过高）。"""
    out = bgr.astype(np.float32)
    b, g, r = out[..., 0], out[..., 1], out[..., 2]
    bad = (r > g + 3) & (b > g) & (b > 80)
    r2 = np.where(bad, np.minimum(r, g - 2), r)
    return np.stack([b, g, r2], axis=-1)


def apply_unified_sky_filter(
    bgr: np.ndarray,
    mask: np.ndarray,
    hue_shift: float = HUE_SHIFT,
    sat_scale: float = SAT_SCALE,
    val_scale: float = VAL_SCALE,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1]
    v = hsv[..., 2]

    sat_w = np.clip((s - HUE_SHIFT_MIN_S) / 25.0, 0.0, 1.0)
    bright_w = np.clip((HUE_FADE_V1 - v) / (HUE_FADE_V1 - HUE_FADE_V0), 0.0, 1.0)
    effect_w = sat_w * bright_w

    h0 = hsv[..., 0]
    h1 = np.clip((h0 + hue_shift) % 180, H_SKY_LO, H_SKY_HI)
    hsv[..., 0] = h0 * (1.0 - effect_w) + h1 * effect_w

    sat_boost = 1.0 + (sat_scale - 1.0) * effect_w
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_boost, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)

    boosted = _suppress_sky_purple(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    m3 = mask[..., None]
    out = bgr.astype(np.float32) * (1.0 - m3) + boosted * m3
    return np.clip(out, 0, 255).astype(np.uint8)
