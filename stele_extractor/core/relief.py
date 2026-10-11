"""높이 지도에서 글자를 드러내는 처리.

1. 배경 제거: 비석의 큰 굴곡/기울기를 빼서 글자 깊이만 남긴다.
2. 글자 신호: 음각이면 '얼마나 파였는지', 양각이면 '얼마나 솟았는지'.
3. 이미지: 탁본, 음영 기복, 깊이 색상, 글자 강조, 이진(흑백) 글자.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

INVALID_GRAY = 200


def _blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """큰 시그마도 빠르게 처리하는 가우시안 블러 (축소 → 블러 → 확대)."""
    img = img.astype(np.float32)
    if sigma_px <= 0.3:
        return img
    if sigma_px <= 12:
        return cv2.GaussianBlur(img, (0, 0), sigma_px, borderType=cv2.BORDER_REFLECT)
    factor = sigma_px / 6.0
    h, w = img.shape
    sh, sw = max(2, int(round(h / factor))), max(2, int(round(w / factor)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6.0, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def masked_blur(values: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    m = mask.astype(np.float32)
    num = _blur(values * m, sigma_px)
    den = _blur(m, sigma_px)
    out = np.zeros_like(num)
    ok = den > 1e-4
    out[ok] = num[ok] / den[ok]
    return out


def robust_std(x: np.ndarray) -> float:
    if x.size == 0:
        return 1.0
    med = np.median(x)
    s = 1.4826 * float(np.median(np.abs(x - med)))
    return s if s > 1e-9 else float(np.std(x)) or 1.0


def remove_background(
    values: np.ndarray, valid: np.ndarray, sigma_px: float, iterations: int = 3
) -> np.ndarray:
    """큰 굴곡을 빼고 남은 잔차를 돌려준다. 글자(이상치)는 배경 추정에서 제외한다."""
    weights = valid.copy()
    residual = np.zeros_like(values, dtype=np.float32)
    for _ in range(iterations):
        bg = masked_blur(values, weights, sigma_px)
        residual = (values - bg).astype(np.float32)
        r = residual[valid]
        if r.size == 0:
            break
        med = float(np.median(r))
        s = robust_std(r)
        weights = valid & (np.abs(residual - med) < 2.0 * s)
        if weights.sum() < 0.2 * valid.sum():
            weights = valid
    residual[~valid] = 0
    return residual


def detect_polarity(residual: np.ndarray, valid: np.ndarray) -> str:
    """글자가 파였는지(음각) 솟았는지(양각) 꼬리 분포로 판단한다."""
    r = residual[valid]
    if r.size < 10:
        return "engraved"
    med = np.median(r)
    low = med - np.percentile(r, 1)
    high = np.percentile(r, 99) - med
    return "engraved" if low >= high else "raised"


def color_signal(
    color: np.ndarray, valid: np.ndarray, sigma_px: float, reference: np.ndarray
) -> np.ndarray:
    """색상(밝기)에서 배경을 뺀 값. 글자 신호와 같은 방향·크기로 맞춘다."""
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY).astype(np.float32)
    res = remove_background(gray, valid, sigma_px, iterations=2)
    a = res[valid]
    b = reference[valid]
    if a.size < 10:
        return np.zeros_like(res)
    corr = np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0.0
    sign = -1.0 if corr < 0 else 1.0
    scale = robust_std(b) / robust_std(a)
    return (res * sign * scale).astype(np.float32)


def otsu_threshold(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    lo, hi = np.percentile(values, [0.5, 99.8])
    if hi <= lo:
        return float(hi)
    hist, edges = np.histogram(np.clip(values, lo, hi), bins=256, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1e-12)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between[:-1]))])


def binarize(
    signal: np.ndarray,
    valid: np.ndarray,
    pixel_mm: float,
    sensitivity: float = 50.0,
    smooth_mm: float = 0.2,
    min_area_mm2: float = 1.0,
    edge_margin_px: int = 2,
) -> tuple[np.ndarray, float]:
    """글자 신호를 흑백 글자 영역으로 바꾼다. (mask, 사용한 임계값) 반환."""
    inner = valid
    if edge_margin_px > 0:
        inner = ndimage.binary_erosion(valid, iterations=edge_margin_px)
    vals = signal[inner]
    base = otsu_threshold(vals)
    noise_floor = 2.5 * robust_std(vals) if vals.size else 0.0
    base = max(base, noise_floor * 0.5)
    # 감도 50 = 자동값, 100 = 절반 임계값(더 많이 잡음), 0 = 두 배(덜 잡음)
    thr = base * 2.0 ** ((50.0 - sensitivity) / 50.0)
    mask = (signal > thr) & inner

    r = int(round(smooth_mm / pixel_mm))
    if r >= 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m8 = mask.astype(np.uint8)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
        mask = m8.astype(bool)

    min_px = min_area_mm2 / (pixel_mm * pixel_mm)
    if min_px >= 1:
        mask = remove_small(mask, min_px)
        mask = ~remove_small(~mask, min_px) & inner
    return mask, float(thr)


def remove_small(mask: np.ndarray, min_px: float) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    keep = stats[:, cv2.CC_STAT_AREA] >= min_px
    keep[0] = False
    return keep[lab]


# ---------------------------------------------------------------------------
# 이미지 만들기 (모두 uint8)
# ---------------------------------------------------------------------------


def _normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def rubbing_image(height_res: np.ndarray, valid: np.ndarray, contrast: float = 1.0,
                  ref: np.ndarray | None = None) -> np.ndarray:
    """탁본: 종이에 먹을 두드리면 높은 면은 검게, 파인 곳은 하얗게 남는다.

    ref: 밝기 기준으로 쓸 높이 (없으면 height_res 자신).
    """
    r = (height_res if ref is None else ref)[valid]
    if r.size == 0:
        return np.full(height_res.shape, 255, np.uint8)
    med = float(np.median(r))
    s = robust_std(r)
    low = float(np.percentile(r, 1))
    depth = max(med - low, 3 * s, 1e-6) / max(contrast, 0.05)
    # 표면 높이(med 근처) → 먹색, 표면보다 깊을수록 → 종이색
    t = _normalize(med + 0.6 * s / max(contrast, 0.05) - height_res, 0, depth)
    t = t ** 0.8
    ink, paper = 25.0, 245.0
    img = ink + (paper - ink) * t
    img[~valid] = 255
    return img.astype(np.uint8)


def hillshade_image(
    height_res: np.ndarray,
    valid: np.ndarray,
    pixel_mm: float,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 35.0,
    exaggeration: float = 3.0,
) -> np.ndarray:
    """낮게 비춘 빛으로 만든 음영. 마모된 글자도 잘 보인다."""
    z = height_res * exaggeration
    gy, gx = np.gradient(z, pixel_mm)
    az = np.deg2rad(360.0 - azimuth_deg + 90.0)
    alt = np.deg2rad(altitude_deg)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    img = np.clip(shade, 0, 1) * 255
    img[~valid] = 255
    return img.astype(np.uint8)


def depth_color_image(height_res: np.ndarray, valid: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """잔차 깊이를 색으로 표시 (파랑=깊음, 빨강=높음). RGB."""
    r = (height_res if ref is None else ref)[valid]
    if r.size == 0:
        return np.full(height_res.shape + (3,), INVALID_GRAY, np.uint8)
    lim = max(abs(np.percentile(r, 1)), abs(np.percentile(r, 99)), 1e-6)
    t = (_normalize(height_res, -lim, lim) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(t, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = INVALID_GRAY
    return rgb


def enhanced_image(signal: np.ndarray, valid: np.ndarray, contrast: float = 1.0,
                   ref: np.ndarray | None = None) -> np.ndarray:
    """글자 강조: 흰 바탕에 글자가 진하게 (OCR·판독용 회색조)."""
    s = (signal if ref is None else ref)[valid]
    if s.size == 0:
        return np.full(signal.shape, 255, np.uint8)
    hi = float(np.percentile(s, 99.5)) / max(contrast, 0.05)
    lo = float(np.median(s))
    t = _normalize(signal, lo, max(hi, lo + 1e-6))
    img = 255 - t * 255
    img[~valid] = 255
    return img.astype(np.uint8)


def mask_image(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    img = np.full(mask.shape, 255, np.uint8)
    img[~valid] = 235
    img[mask] = 0
    return img
