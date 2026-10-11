"""전체 처리 흐름: 스캔 → 높이 지도 → (자르기) → 글자 이미지/마스크."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import relief
from .scan import ScanData
from .surface import HeightMap, Projection, auto_pixel_size, project, rasterize, rotate_uv

# 결과 이미지 이름 (GUI 표시 순서)
IMAGE_NAMES = {
    "rubbing": "탁본",
    "hillshade": "음영 기복",
    "enhanced": "글자 강조",
    "mask": "글자 추출(흑백)",
    "depth": "깊이 색상",
    "color": "스캔 색상",
}


@dataclass
class GeometryParams:
    """다시 계산하는 데 시간이 걸리는 단계(3D → 2D)의 설정."""

    surface_mode: str = "auto"  # auto(자동) / plane(평면) / cylinder(원통 펼치기)
    pixel_mm: float = 0.0  # 0이면 자동
    flip: bool = False  # 앞뒤 뒤집기
    rotation_deg: float = 0.0  # 화면에서 돌리기
    auto_align: bool = True  # 글자 획 방향을 보고 기울어짐을 자동으로 바로잡기


@dataclass
class ReliefParams:
    """슬라이더로 바로 바뀌는 단계의 설정."""

    letter_type: str = "auto"  # auto / engraved(음각) / raised(양각)
    background_mm: float = 0.0  # 배경 제거 크기, 0이면 자동
    denoise_mm: float = 0.1  # 잡음 제거
    color_weight: float = 0.0  # 0~1, 색상 보조 비율
    contrast: float = 1.0
    light_azimuth: float = 315.0
    light_altitude: float = 35.0
    exaggeration: float = 3.0
    sensitivity: float = 50.0  # 0~100
    smooth_mm: float = 0.2
    min_area_mm2: float = 1.0


@dataclass
class ReliefResult:
    heightmap: HeightMap
    residual: np.ndarray  # 배경 뺀 실제 높이 (mm, 바깥쪽 +)
    signal: np.ndarray  # 글자 신호 (클수록 글자)
    mask: np.ndarray  # 글자 영역 bool
    polarity: str  # engraved / raised
    threshold: float
    background_mm: float
    images: dict = field(default_factory=dict)
    # residual = height_sign × (스캔 표면 바깥쪽 높이). 점군에서 앞뒤를 뒤집어 보여 줄 때 -1
    height_sign: float = 1.0
    # 깊이 편집으로 더한 높이 변화 (residual과 같은 방향, mm). 편집하지 않았으면 None
    delta: np.ndarray | None = None

    def stats(self) -> dict:
        hm = self.heightmap
        v = hm.valid
        depth = float(np.percentile(self.signal[self.mask], 90)) if self.mask.any() else 0.0
        return {
            "크기(mm)": f"{hm.size_mm[0]:.1f} x {hm.size_mm[1]:.1f}",
            "해상도": f"{hm.pixel_mm:.3f} mm/픽셀 ({hm.dpi:.0f} DPI)",
            "글자 형태": "음각" if self.polarity == "engraved" else "양각",
            "글자 깊이(추정)": f"{depth:.2f} mm",
            "글자 면적 비율": f"{100 * self.mask.sum() / max(v.sum(), 1):.1f} %",
        }


def auto_background_mm(hm: HeightMap) -> float:
    """배경 제거 크기 자동값: 영역 크기의 1/8 정도, 3~40mm."""
    w, h = hm.size_mm
    return float(np.clip(min(w, h) / 8.0, 3.0, 40.0))


def estimate_text_angle(proj: Projection, pixel_mm: float, spacing: float) -> float:
    """글자 획(가로획·세로획)이 화면과 나란해지도록 돌릴 각도(도, -45~45)를 추정한다."""
    import cv2

    hm = rasterize(proj, pixel_mm, point_spacing=spacing)
    if min(hm.shape) < 16:
        return 0.0
    bg = auto_background_mm(hm)
    res = relief.remove_background(hm.height, hm.valid, bg / pixel_mm / 2.0, iterations=2)
    res = relief.masked_blur(res, hm.valid, 1.0)
    gx = cv2.Sobel(res, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(res, cv2.CV_32F, 0, 1, ksize=3)
    inner = cv2.erode(hm.valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    w = (gx * gx + gy * gy) * inner
    if w.sum() <= 0:
        return 0.0
    # 강한 경계(글자 가장자리)만 쓴다
    w = np.where(w > np.percentile(w[inner], 90), w, 0)
    theta = np.arctan2(gy, gx)
    s4, c4 = (w * np.sin(4 * theta)).sum(), (w * np.cos(4 * theta)).sum()
    if np.hypot(s4, c4) < 0.05 * w.sum():  # 뚜렷한 방향이 없음
        return 0.0
    phi = np.degrees(np.arctan2(s4, c4) / 4.0)
    return float(round(phi, 2))


def build_geometry(scan: ScanData, gp: GeometryParams) -> tuple[Projection, HeightMap]:
    """3D 스캔을 펼쳐 높이 지도를 만든다. 펼친 좌표(Projection)도 함께 돌려준다."""
    proj = project(scan, gp.surface_mode, gp.flip, 0.0)
    spacing = scan.point_spacing()
    pixel_mm = gp.pixel_mm or auto_pixel_size(scan, proj)
    align = 0.0
    if gp.auto_align:
        align = estimate_text_angle(proj, max(pixel_mm, spacing) * 1.5, spacing)
    total = align + gp.rotation_deg
    if total:
        proj.u, proj.v = rotate_uv(proj.u, proj.v, total)
    proj.info["auto_align_deg"] = align
    hm = rasterize(proj, pixel_mm, colors=scan.colors, attributes=scan.attributes, point_spacing=spacing)
    hm.info["point_spacing_mm"] = spacing
    return proj, hm


def build_heightmap(scan: ScanData, gp: GeometryParams) -> HeightMap:
    return build_geometry(scan, gp)[1]


def process_relief(hm: HeightMap, rp: ReliefParams) -> ReliefResult:
    px = hm.pixel_mm
    valid = hm.valid
    bg_mm = rp.background_mm or auto_background_mm(hm)
    residual = relief.remove_background(hm.height, valid, sigma_px=bg_mm / px / 2.0)
    if rp.denoise_mm > 0:
        residual = relief.masked_blur(residual, valid, rp.denoise_mm / px)
        residual[~valid] = 0

    detected = relief.detect_polarity(residual, valid)
    polarity = detected if rp.letter_type == "auto" else rp.letter_type
    geom_signal = -residual if polarity == "engraved" else residual

    # 법선으로 앞뒤를 알 수 없으면(점군) 자동 판정 결과에 맞춰 높이 방향을 정한다.
    height_res = residual
    if not hm.info.get("oriented_by_normals", True) and rp.letter_type == "auto" and detected == "raised":
        # 대부분의 비석은 음각이므로 '음각'으로 보이도록 뒤집는다.
        height_res = -residual
        geom_signal = residual
        polarity = "engraved"

    signal = geom_signal
    if hm.color is not None and rp.color_weight > 0:
        csig = relief.color_signal(hm.color, valid, bg_mm / px / 2.0, geom_signal)
        signal = (1 - rp.color_weight) * geom_signal + rp.color_weight * csig
    signal = signal.astype(np.float32)
    signal[~valid] = 0

    mask, thr = relief.binarize(
        signal,
        valid,
        px,
        sensitivity=rp.sensitivity,
        smooth_mm=rp.smooth_mm,
        min_area_mm2=rp.min_area_mm2,
    )

    images = render_images(hm, height_res, signal, mask, rp)

    return ReliefResult(
        heightmap=hm,
        residual=height_res,
        signal=signal,
        mask=mask,
        polarity=polarity,
        threshold=thr,
        background_mm=bg_mm,
        images=images,
        height_sign=-1.0 if height_res is not residual else 1.0,
    )


def render_images(
    hm: HeightMap,
    height_res: np.ndarray,
    signal: np.ndarray,
    mask: np.ndarray,
    rp: ReliefParams,
    ref_res: np.ndarray | None = None,
    ref_signal: np.ndarray | None = None,
) -> dict:
    """결과 이미지들을 만든다.

    ref_res / ref_signal을 주면 밝기 기준을 그 값(편집 전)에 맞춰, 깊이를 바꾼 만큼 이미지도 달라 보이게 한다.
    """
    valid = hm.valid
    images = {
        "rubbing": relief.rubbing_image(height_res, valid, rp.contrast, ref=ref_res),
        "hillshade": relief.hillshade_image(
            height_res, valid, hm.pixel_mm, rp.light_azimuth, rp.light_altitude, rp.exaggeration
        ),
        "enhanced": relief.enhanced_image(signal, valid, rp.contrast, ref=ref_signal),
        "mask": relief.mask_image(mask, valid),
        "depth": relief.depth_color_image(height_res, valid, ref=ref_res),
    }
    if hm.color is not None:
        images["color"] = hm.color
    return images


def params_to_dict(gp: GeometryParams, rp: ReliefParams) -> dict:
    return {"geometry": asdict(gp), "relief": asdict(rp)}
