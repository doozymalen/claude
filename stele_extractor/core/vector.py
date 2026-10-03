"""흑백 글자 마스크를 벡터 외곽선(SVG/DXF, mm 단위)으로 만든다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

UPSAMPLE = 4  # 매끈한 외곽선을 위해 확대해서 윤곽을 딴다


@dataclass
class Outline:
    points: np.ndarray  # (K, 2) mm 좌표, x 오른쪽 / y 아래쪽 (이미지 기준)
    is_hole: bool  # 글자 안쪽 구멍 (예: 'ㅇ'의 가운데)
    parent: int = -1  # 구멍이면 바깥 외곽선의 번호


def mask_to_outlines(
    mask: np.ndarray,
    pixel_mm: float,
    simplify_mm: float = 0.05,
    smooth_mm: float = 0.1,
) -> list[Outline]:
    """마스크 → 외곽선 목록. 계단 모양을 줄이려고 부드럽게 한 뒤 확대해서 윤곽을 딴다."""
    m = mask.astype(np.float32)
    if not m.any():
        return []
    m = np.pad(m, 2)
    sigma = max(0.5, smooth_mm / pixel_mm)
    big = cv2.resize(m, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_LINEAR)
    big = cv2.GaussianBlur(big, (0, 0), sigma * UPSAMPLE)
    binary = (big >= 0.5).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    eps = simplify_mm / pixel_mm * UPSAMPLE
    scale = pixel_mm / UPSAMPLE
    offset = 2 * pixel_mm  # 위에서 붙인 여백
    outlines: list[Outline] = []
    index_of: dict[int, int] = {}
    # 바깥 외곽선을 먼저 넣어야 구멍이 부모 번호를 찾을 수 있다.
    order = sorted(range(len(contours)), key=lambda i: hierarchy[i][3] >= 0)
    for i in order:
        cnt, h = contours[i], hierarchy[i]
        is_hole = h[3] >= 0
        if is_hole and h[3] not in index_of:
            continue
        if eps > 0:
            cnt = cv2.approxPolyDP(cnt, eps, True)
        if len(cnt) < 3:
            continue
        # 픽셀 중심 보정(+0.5) 후 mm로
        pts = (cnt.reshape(-1, 2).astype(np.float64) + 0.5) * scale - offset
        index_of[i] = len(outlines)
        outlines.append(Outline(points=pts, is_hole=is_hole, parent=index_of[h[3]] if is_hole else -1))
    return outlines


def outlines_to_polygons(outlines: list[Outline], height_mm: float | None = None, flip_y: bool = True):
    """shapely 폴리곤 목록(구멍 포함)으로 변환. flip_y면 y를 위쪽(+)으로 바꾼다."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    def conv(p):
        if flip_y and height_mm is not None:
            return np.column_stack([p[:, 0], height_mm - p[:, 1]])
        return p

    pieces = []
    for i, o in enumerate(outlines):
        if o.is_hole:
            continue
        shell = Polygon(conv(o.points)).buffer(0)
        holes = [Polygon(conv(h.points)).buffer(0) for h in outlines if h.is_hole and h.parent == i]
        if holes:
            shell = shell.difference(unary_union(holes))
        pieces.append(shell)
    if not pieces:
        return []
    shape = unary_union(pieces)
    geoms = getattr(shape, "geoms", [shape])
    return [g for g in geoms if g.area > 0]


def _path_d(points: np.ndarray) -> str:
    parts = [f"M{points[0, 0]:.3f},{points[0, 1]:.3f}"]
    parts += [f"L{x:.3f},{y:.3f}" for x, y in points[1:]]
    return "".join(parts) + "Z"


def save_svg(outlines: list[Outline], path: str | Path, width_mm: float, height_mm: float,
             fill: bool = True) -> Path:
    """실제 크기(mm)로 열리는 SVG. 일러스트레이터/잉크스케이프에서 1:1."""
    d = "".join(_path_d(o.points) for o in outlines)
    style = 'fill="#000000" fill-rule="evenodd" stroke="none"' if fill else \
        'fill="none" stroke="#000000" stroke-width="0.1"'
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{width_mm:.3f}mm" height="{height_mm:.3f}mm" '
        f'viewBox="0 0 {width_mm:.3f} {height_mm:.3f}">\n'
        '  <title>비석 글자 외곽선 (단위: mm)</title>\n'
        f'  <rect x="0" y="0" width="{width_mm:.3f}" height="{height_mm:.3f}" '
        'fill="none" stroke="#bbbbbb" stroke-width="0.1"/>\n'
        f'  <path {style} d="{d}"/>\n'
        "</svg>\n"
    )
    Path(path).write_text(svg, encoding="utf-8")
    return Path(path)


def save_dxf(outlines: list[Outline], path: str | Path, width_mm: float, height_mm: float) -> Path:
    """CAD용 DXF (단위 mm, y축 위쪽). 바깥 외곽선/구멍/테두리를 레이어로 나눈다."""
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    doc.layers.add("LETTERS", color=7)
    doc.layers.add("HOLES", color=1)
    doc.layers.add("FRAME", color=8)
    msp = doc.modelspace()
    for o in outlines:
        pts = [(float(x), float(height_mm - y)) for x, y in o.points]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "HOLES" if o.is_hole else "LETTERS"})
    msp.add_lwpolyline(
        [(0, 0), (width_mm, 0), (width_mm, height_mm), (0, height_mm)],
        close=True,
        dxfattribs={"layer": "FRAME"},
    )
    doc.saveas(str(path))
    return Path(path)
