"""글자 깊이 편집: 음각·양각 글자의 깊이를 늘리거나 줄이고, 뒤집는다.

원리
- 처리 결과의 잔차(residual)는 '배경 굴곡을 뺀 높이'라서 돌 표면은 약 0, 글자는 파이거나(음수) 솟아(양수) 있다.
- 글자 부분의 잔차에 배율을 곱하면 원래 모양(V자 홈, 마모 흔적)은 그대로 둔 채 깊이만 바뀐다.
    배율 2 = 두 배 깊게, 0.5 = 절반, 0 = 평평하게, -1 = 음각↔양각 뒤집기
- 글자마다 배율을 따로 줄 수 있다. 글자 경계는 부드럽게 이어 붙인다.
- 배경(돌 표면)은 그대로 두거나 자잘한 거칠기를 줄일 수 있다.
- 바뀐 높이(delta)를 원본 스캔 정점에도 적용해 3D 수정본을 만든다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import cv2
import numpy as np
from scipy import ndimage

from .pipeline import ReliefParams, ReliefResult, render_images
from .scan import ScanData, compute_vertex_normals
from .surface import HeightMap, Projection


@dataclass
class DepthEdit:
    enabled: bool = False
    global_scale: float = 1.0  # 모든 글자에 곱할 깊이 배율
    # 글자별 배율: [u_mm, v_mm, 배율] — 글자 안쪽 한 점(펼친 면 좌표)으로 글자를 기억한다.
    overrides: list = field(default_factory=list)
    merge_mm: float = 0.0  # 획을 한 글자로 묶는 거리, 0이면 자동
    feather_mm: float = 0.3  # 글자 경계를 부드럽게 이을 폭
    bg_mode: str = "keep"  # keep(그대로) / smooth(매끈하게)
    bg_smooth: float = 0.7  # 0~1, 배경 거칠기를 얼마나 줄일지

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "DepthEdit":
        d = d or {}
        names = set(DepthEdit.__dataclass_fields__)
        return DepthEdit(**{k: v for k, v in d.items() if k in names})

    def is_identity(self) -> bool:
        no_letters = abs(self.global_scale - 1.0) < 1e-6 and all(abs(o[2] - 1.0) < 1e-6 for o in self.overrides)
        return no_letters and (self.bg_mode != "smooth" or self.bg_smooth <= 0)


# ---------------------------------------------------------------------------
# 글자 나누기
# ---------------------------------------------------------------------------


def stroke_width_mm(mask: np.ndarray, pixel_mm: float) -> float:
    """획 굵기(mm) 추정: 글자 안쪽 거리의 대표값 × 2."""
    if not mask.any():
        return 1.0
    dist = ndimage.distance_transform_edt(mask)
    skeleton_vals = dist[(dist >= ndimage.maximum_filter(dist, 3)) & mask]
    return float(2 * np.median(skeleton_vals) * pixel_mm) if skeleton_vals.size else 1.0


def auto_merge_mm(mask: np.ndarray, pixel_mm: float) -> float:
    """한 글자 안에서 떨어진 획을 이어 줄 거리 (획 굵기의 약 1.5배)."""
    return max(1.5 * stroke_width_mm(mask, pixel_mm), 2 * pixel_mm)


def label_letters(mask: np.ndarray, pixel_mm: float, merge_mm: float = 0.0) -> tuple[np.ndarray, int]:
    """글자 마스크를 글자 단위로 번호 매긴다. (labels, 글자 수). 0은 글자 아님."""
    if not mask.any():
        return np.zeros(mask.shape, np.int32), 0
    merge = merge_mm or auto_merge_mm(mask, pixel_mm)
    r = max(1, int(round(merge / 2 / pixel_mm)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    grown = cv2.dilate(mask.astype(np.uint8), k)
    n, lab = cv2.connectedComponents(grown, connectivity=8)
    labels = np.where(mask, lab, 0).astype(np.int32)
    return labels, int(n - 1)


def letter_anchor(labels: np.ndarray, label: int) -> tuple[int, int] | None:
    """글자 안쪽에서 가장 깊숙한 픽셀 (ㅇ처럼 가운데가 빈 글자도 글자 위의 점이 된다)."""
    m = labels == label
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    d = ndimage.distance_transform_edt(np.pad(m[r0:r1, c0:c1], 1))[1:-1, 1:-1]
    i = int(np.argmax(d))
    return r0 + i // (c1 - c0), c0 + i % (c1 - c0)


def pixel_to_uv(hm: HeightMap, row: float, col: float) -> tuple[float, float]:
    return hm.u0 + (col + 0.5) * hm.pixel_mm, hm.v0 - (row + 0.5) * hm.pixel_mm


def uv_to_pixel(hm: HeightMap, u: float, v: float) -> tuple[int, int]:
    return int(np.floor((hm.v0 - v) / hm.pixel_mm)), int(np.floor((u - hm.u0) / hm.pixel_mm))


def label_at(labels: np.ndarray, row: int, col: int, search_px: int = 0) -> int:
    """(row, col)에 있는 글자 번호. 비어 있으면 search_px 안에서 가장 가까운 글자."""
    h, w = labels.shape
    if 0 <= row < h and 0 <= col < w and labels[row, col]:
        return int(labels[row, col])
    if search_px <= 0:
        return 0
    r0, r1 = max(0, row - search_px), min(h, row + search_px + 1)
    c0, c1 = max(0, col - search_px), min(w, col + search_px + 1)
    win = labels[r0:r1, c0:c1]
    if win.size == 0 or not win.any():
        return 0
    ys, xs = np.nonzero(win)
    d = (ys + r0 - row) ** 2 + (xs + c0 - col) ** 2
    i = int(np.argmin(d))
    return int(win[ys[i], xs[i]])


def labels_in_rect(labels: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> list[int]:
    win = labels[max(0, r0):r1, max(0, c0):c1]
    return sorted(int(x) for x in np.unique(win) if x)


def letter_scales(edit: DepthEdit, labels: np.ndarray, n: int, hm: HeightMap) -> np.ndarray:
    """글자 번호 → 배율 표 (0번은 배경용 1.0)."""
    table = np.full(n + 1, edit.global_scale, np.float32)
    table[0] = 1.0
    search = max(1, int(round(1.0 / hm.pixel_mm)))
    for u, v, s in edit.overrides:
        r, c = uv_to_pixel(hm, u, v)
        lab = label_at(labels, r, c, search)
        if lab:
            table[lab] = s
    return table


def set_letter_scale(edit: DepthEdit, labels: np.ndarray, hm: HeightMap, ids: list[int], scale: float | None) -> None:
    """고른 글자의 배율을 정한다. scale=None이면 글자별 설정을 지워 전체 배율을 따르게 한다."""
    search = max(1, int(round(1.0 / hm.pixel_mm)))
    ids = set(ids)
    kept = []
    for o in edit.overrides:
        r, c = uv_to_pixel(hm, o[0], o[1])
        if label_at(labels, r, c, search) not in ids:
            kept.append(o)
    if scale is not None:
        for lab in ids:
            a = letter_anchor(labels, lab)
            if a is not None:
                u, v = pixel_to_uv(hm, *a)
                kept.append([round(u, 4), round(v, 4), float(scale)])
    edit.overrides = kept


# ---------------------------------------------------------------------------
# 높이 바꾸기
# ---------------------------------------------------------------------------


def compute_delta(result: ReliefResult, labels: np.ndarray, n: int, edit: DepthEdit) -> np.ndarray:
    """편집으로 더할 높이(mm, residual과 같은 방향)."""
    hm = result.heightmap
    px = hm.pixel_mm
    valid = hm.valid
    res = np.where(valid, result.residual, 0).astype(np.float32)
    table = letter_scales(edit, labels, n, hm)

    # 글자 영역을 조금 넓혀 홈의 벽면까지 포함하고, 각 칸은 가장 가까운 글자의 배율을 쓴다.
    feather_px = max(1.0, edit.feather_mm / px)
    region = labels > 0
    if region.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~region, return_indices=True)
        nearest = labels[iy, ix]
        grow = int(np.ceil(feather_px))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        zone = cv2.dilate(region.astype(np.uint8), k).astype(np.float32)
        weight = cv2.GaussianBlur(zone, (0, 0), feather_px / 2) if feather_px > 1 else zone
        weight = np.clip(weight, 0, 1) * valid
        factor = 1.0 + weight * (table[nearest] - 1.0)
    else:
        weight = np.zeros_like(res)
        factor = np.ones_like(res)

    delta = (factor - 1.0) * res
    if edit.bg_mode == "smooth" and edit.bg_smooth > 0:
        delta -= edit.bg_smooth * (1.0 - weight) * res
    delta[~valid] = 0
    return delta.astype(np.float32)


def apply_edit(
    result: ReliefResult,
    labels: np.ndarray,
    n: int,
    edit: DepthEdit,
    rp: ReliefParams,
) -> ReliefResult:
    """편집을 적용한 결과(이미지, 잔차, 글자 신호 포함)를 새로 만든다. 원래 결과는 그대로 둔다."""
    if not edit.enabled or edit.is_identity():
        return result
    delta = compute_delta(result, labels, n, edit)
    residual = result.residual + delta
    # 글자 신호는 '글자일수록 큰 값': 음각이면 높이의 반대 방향
    sign = -1.0 if result.polarity == "engraved" else 1.0
    signal = (result.signal + sign * delta).astype(np.float32)
    images = render_images(result.heightmap, residual, signal, result.mask, rp,
                           ref_res=result.residual, ref_signal=result.signal)
    return replace(result, residual=residual, signal=signal, images=images, delta=delta)


def letter_depth_mm(result: ReliefResult, labels: np.ndarray, ids: list[int] | None = None) -> float:
    """글자(들)의 대표 깊이(mm, 90% 지점). 원래 형태(음각/양각) 방향이면 +, 뒤집혔으면 -."""
    sign = -1.0 if result.polarity == "engraved" else 1.0
    m = labels > 0 if not ids else np.isin(labels, ids)
    if not m.any():
        return 0.0
    vals = sign * result.residual[m]
    if np.median(vals) >= 0:
        return float(np.percentile(vals, 90))
    return -float(np.percentile(-vals, 90))  # 뒤집혀서 반대 방향이 된 경우 음수


# ---------------------------------------------------------------------------
# 원본 스캔에 적용
# ---------------------------------------------------------------------------


def height_directions(proj: Projection, vertices: np.ndarray) -> np.ndarray:
    """정점마다 '높이(+)' 방향 단위 벡터. 펼칠 때 쓴 평면 법선 또는 원통 반지름 방향."""
    info = proj.info
    if proj.mode == "cylinder":
        c = np.asarray(info["center"], float)
        a = np.asarray(info["axis"], float)
        w = vertices - c
        radial = w - np.outer(w @ a, a)
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        return radial if info.get("convex", True) else -radial
    n = np.asarray(info["normal"], float)
    return np.broadcast_to(n, vertices.shape)


def sample_grid(grid: np.ndarray, proj: Projection, hm: HeightMap) -> np.ndarray:
    """정점 위치에서 격자 값을 부드럽게(쌍선형) 읽는다. 격자 밖은 0."""
    rows = (hm.v0 - proj.v) / hm.pixel_mm - 0.5
    cols = (proj.u - hm.u0) / hm.pixel_mm - 0.5
    return ndimage.map_coordinates(grid.astype(np.float32), [rows, cols], order=1, mode="constant", cval=0.0)


def displace_scan(scan: ScanData, proj: Projection, edited: ReliefResult) -> ScanData:
    """깊이 편집을 원본 스캔 정점에 적용한 새 스캔."""
    if edited.delta is None:
        return scan
    hm = edited.heightmap
    d = sample_grid(edited.delta, proj, hm) * edited.height_sign
    dirs = height_directions(proj, scan.vertices)
    vertices = scan.vertices + dirs * d[:, None]
    return ScanData(
        vertices=vertices,
        faces=scan.faces,
        colors=scan.colors,
        vertex_normals=compute_vertex_normals(vertices, scan.faces) if scan.has_faces else scan.vertex_normals,
        source_path=scan.source_path,
        attributes=scan.attributes,
    )
