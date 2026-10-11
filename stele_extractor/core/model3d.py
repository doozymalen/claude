"""글자만 분리한 3D 모델 만들기.

- 기록용: 원래 스캔 메쉬에서 글자 부분만 잘라낸다 (실제 표면 그대로, PLY/OBJ).
- 프린팅용: 두께가 있는 닫힌(watertight) STL.
    * 글자판: 바닥판 위에 글자가 솟은 판 (또는 글자가 파인 판)
    * 글자만: 글자 외곽선을 위로 밀어 올린 입체 글자
    * 표면 복제: 실제 비석 표면의 굴곡을 그대로 살린 판
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from .scan import ScanData
from .surface import HeightMap, Projection

MAX_PRINT_CELLS = 1_500_000


# ---------------------------------------------------------------------------
# 기록용: 원본 메쉬에서 글자 부분 잘라내기
# ---------------------------------------------------------------------------


def vertex_pixels(proj: Projection, hm: HeightMap) -> tuple[np.ndarray, np.ndarray]:
    """각 정점이 높이 지도의 몇 행/열에 해당하는지."""
    col = np.floor((proj.u - hm.u0) / hm.pixel_mm).astype(np.int64)
    row = np.floor((hm.v0 - proj.v) / hm.pixel_mm).astype(np.int64)
    return row, col


def cut_letters_from_scan(
    scan: ScanData,
    proj: Projection,
    hm: HeightMap,
    mask: np.ndarray,
    margin_mm: float = 0.5,
) -> ScanData:
    """글자 마스크(여백 포함) 안에 있는 원본 삼각형만 남긴다."""
    if not scan.has_faces:
        raise ValueError("점군(면이 없는 스캔)은 잘라낼 수 없습니다. 메쉬로 내보낸 파일을 쓰세요.")
    m = mask
    r = int(round(margin_mm / hm.pixel_mm))
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    row, col = vertex_pixels(proj, hm)
    h, w = m.shape
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    sel = np.zeros(scan.n_points, bool)
    sel[inside] = m[row[inside], col[inside]]
    faces = scan.faces[sel[scan.faces].all(axis=1)]
    if len(faces) == 0:
        raise ValueError("잘라낼 글자 영역이 없습니다. 감도를 조정해 보세요.")
    used = np.unique(faces)
    remap = -np.ones(scan.n_points, np.int64)
    remap[used] = np.arange(len(used))
    return ScanData(
        vertices=scan.vertices[used],
        faces=remap[faces],
        colors=None if scan.colors is None else scan.colors[used],
        vertex_normals=None if scan.vertex_normals is None else scan.vertex_normals[used],
    )


def save_scan_mesh(scan: ScanData, path: str | Path) -> Path:
    """스캔(메쉬 또는 점군)을 PLY/OBJ/STL로 저장한다."""
    import trimesh

    colors = None
    if scan.colors is not None:
        colors = np.column_stack([scan.colors, np.full(len(scan.colors), 255, np.uint8)])
    if scan.has_faces:
        geom = trimesh.Trimesh(scan.vertices, scan.faces, vertex_colors=colors, process=False)
    else:
        if Path(path).suffix.lower() == ".stl":
            raise ValueError("점군(면이 없는 스캔)은 STL로 저장할 수 없습니다. PLY를 고르세요.")
        geom = trimesh.PointCloud(scan.vertices, colors=colors)
    geom.export(str(path))
    return Path(path)


# ---------------------------------------------------------------------------
# 프린팅용
# ---------------------------------------------------------------------------


@dataclass
class PrintOptions:
    kind: str = "plate"  # plate(글자판) / letters(글자만) / relief(표면 복제)
    base_mm: float = 3.0  # 바닥판 두께
    letter_mm: float = 2.0  # 글자 높이(깊이)
    engraved_plate: bool = False  # 글자판: 글자를 파낸 판으로
    relief_scale: float = 1.0  # 표면 복제: 높이 배율
    border_mm: float = 0.0  # 글자판 둘레 여백 (0이면 스캔 영역 그대로)
    mirror: bool = False  # 좌우 반전 (도장/찍기용)


def heightfield_solid(z: np.ndarray, pixel_mm: float):
    """높이 격자(모두 > 0)로 바닥이 평평한 닫힌 입체를 만든다."""
    import trimesh

    h, w = z.shape
    if h < 2 or w < 2:
        raise ValueError("영역이 너무 작습니다.")
    xs = np.arange(w) * pixel_mm
    ys = (h - 1 - np.arange(h)) * pixel_mm  # 행 0이 위쪽(+y)
    X, Y = np.meshgrid(xs, ys)
    top = np.column_stack([X.ravel(), Y.ravel(), z.ravel()])
    idx = np.arange(h * w).reshape(h, w)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    # 위에서 보았을 때 반시계 방향 → 법선이 위(+z)
    top_faces = np.concatenate([np.column_stack([a, c, b]), np.column_stack([b, c, d])])

    # 둘레 (위에서 보아 시계 방향으로 한 바퀴). 옆면/바닥 법선은 바깥을 향한다.
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    n_top = h * w
    bottom = top[ring].copy()
    bottom[:, 2] = 0.0
    center = np.array([[xs.mean(), ys.mean(), 0.0]])
    verts = np.vstack([top, bottom, center])
    nb = len(ring)
    bi = n_top + np.arange(nb)
    bj = n_top + (np.arange(nb) + 1) % nb
    ti = ring
    tj = ring[(np.arange(nb) + 1) % nb]
    sides = np.concatenate([np.column_stack([ti, tj, bi]), np.column_stack([tj, bj, bi])])
    ci = n_top + nb
    bottom_faces = np.column_stack([np.full(nb, ci), bi, bj])
    faces = np.concatenate([top_faces, sides, bottom_faces])
    mesh = trimesh.Trimesh(verts, faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _downsample(arr: np.ndarray, pixel_mm: float, max_cells: int, interp=cv2.INTER_AREA):
    h, w = arr.shape
    if h * w <= max_cells:
        return arr, pixel_mm
    f = np.sqrt(max_cells / (h * w))
    nw, nh = max(2, int(w * f)), max(2, int(h * f))
    out = cv2.resize(arr.astype(np.float32), (nw, nh), interpolation=interp)
    return out, pixel_mm * w / nw


def make_print_model(
    mask: np.ndarray,
    valid: np.ndarray,
    pixel_mm: float,
    opts: PrintOptions,
    residual: np.ndarray | None = None,
    polygons=None,
):
    """프린팅용 닫힌 메쉬(trimesh.Trimesh)를 만든다."""
    import trimesh

    if opts.kind == "letters":
        if not polygons:
            raise ValueError("글자 외곽선이 없습니다. 감도를 조정해 보세요.")
        parts = [
            trimesh.creation.extrude_polygon(p, opts.letter_mm)
            for p in polygons
            if p.area > 0.01
        ]
        if not parts:
            raise ValueError("글자 외곽선이 너무 작습니다.")
        mesh = trimesh.util.concatenate(parts)
        if opts.mirror:
            mesh.apply_transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
            if mesh.volume < 0:
                mesh.invert()
        return mesh

    if opts.kind == "relief":
        if residual is None:
            raise ValueError("표면 높이 정보가 없습니다.")
        r = np.where(valid, residual, 0.0).astype(np.float32)
        lo = float(np.percentile(r[valid], 0.5)) if valid.any() else 0.0
        z = opts.base_mm + (np.clip(r - lo, 0, None)) * opts.relief_scale
        z = np.where(valid, z, opts.base_mm)
    else:  # plate
        m = mask.astype(np.float32)
        # 계단을 줄이려고 살짝 부드럽게
        m = cv2.GaussianBlur(m, (0, 0), 0.6)
        if opts.engraved_plate:
            z = opts.base_mm + opts.letter_mm * (1.0 - m)
        else:
            z = opts.base_mm + opts.letter_mm * m

    if opts.border_mm > 0:
        b = int(round(opts.border_mm / pixel_mm))
        edge_val = opts.base_mm + (opts.letter_mm if opts.kind == "plate" and opts.engraved_plate else 0)
        z = np.pad(z, b, constant_values=edge_val)

    z, px = _downsample(z, pixel_mm, MAX_PRINT_CELLS)
    if opts.mirror:
        z = z[:, ::-1]
    z = np.maximum(z, 0.2)
    return heightfield_solid(np.ascontiguousarray(z), px)


def save_mesh(mesh, path: str | Path) -> Path:
    mesh.export(str(path))
    return Path(path)


def mesh_report(mesh) -> dict:
    ext = mesh.extents
    return {
        "삼각형 수": f"{len(mesh.faces):,}",
        "크기(mm)": f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}",
        "닫힌 메쉬(watertight)": "예" if mesh.is_watertight else "아니오",
    }


def label_count(mask: np.ndarray) -> int:
    return int(ndimage.label(mask)[1])
