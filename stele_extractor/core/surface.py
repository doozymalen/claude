"""스캔 표면을 2D 평면으로 펼치고 높이 지도(height map)로 만든다.

- 평면 투영: 표면에 가장 잘 맞는 평면(PCA)을 기준으로 위에서 내려다본 높이.
- 원통 펼치기: 둥근 비석처럼 휘어진 면을 원통으로 근사해 종이처럼 펼친다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .scan import ScanData

MAX_GRID_PIXELS = 40_000_000


@dataclass
class Projection:
    """3D 점을 펼친 좌표 (u, v: 면 위 mm 좌표, h: 면에서 바깥쪽 높이 mm)."""

    u: np.ndarray
    v: np.ndarray
    h: np.ndarray
    mode: str
    info: dict = field(default_factory=dict)


@dataclass
class HeightMap:
    """펼친 표면을 일정 간격 격자로 만든 결과.

    행(row) 0이 위쪽(v 최대), 열(col) 0이 왼쪽(u 최소)이다.
    """

    height: np.ndarray  # (H, W) float32, mm. 유효하지 않은 곳은 0
    valid: np.ndarray  # (H, W) bool
    pixel_mm: float
    u0: float  # 왼쪽 경계의 u 값
    v0: float  # 위쪽 경계의 v 값
    color: np.ndarray | None = None  # (H, W, 3) uint8
    layers: dict = field(default_factory=dict)  # 부가 정보 격자 (예: 정답 라벨)
    info: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height.shape

    @property
    def size_mm(self) -> tuple[float, float]:
        h, w = self.shape
        return w * self.pixel_mm, h * self.pixel_mm

    @property
    def dpi(self) -> float:
        return 25.4 / self.pixel_mm

    def crop(self, r0: int, r1: int, c0: int, c1: int) -> "HeightMap":
        h, w = self.shape
        r0, r1 = max(0, r0), min(h, r1)
        c0, c1 = max(0, c0), min(w, c1)
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise ValueError("선택한 영역이 너무 작습니다.")
        return HeightMap(
            height=self.height[r0:r1, c0:c1].copy(),
            valid=self.valid[r0:r1, c0:c1].copy(),
            pixel_mm=self.pixel_mm,
            u0=self.u0 + c0 * self.pixel_mm,
            v0=self.v0 - r0 * self.pixel_mm,
            color=None if self.color is None else self.color[r0:r1, c0:c1].copy(),
            layers={k: a[r0:r1, c0:c1].copy() for k, a in self.layers.items()},
            info=dict(self.info),
        )


# ---------------------------------------------------------------------------
# 기준 좌표계
# ---------------------------------------------------------------------------


def _subsample(n: int, k: int, seed: int = 0) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, k, replace=False)


def fit_plane_frame(scan: ScanData, flip: bool = False):
    """PCA로 기준 평면을 구한다. (원점, e1, e2, n) 을 돌려준다.

    n은 스캐너 쪽(표면 바깥)을 향하도록 정점 법선으로 방향을 맞춘다.
    e1 x e2 = n 인 오른손 좌표계라서 앞에서 보았을 때 글자가 뒤집히지 않는다.
    """
    pts = scan.vertices
    idx = _subsample(len(pts), 200_000)
    sample = pts[idx]
    origin = sample.mean(axis=0)
    cov = np.cov((sample - origin).T)
    evals, evecs = np.linalg.eigh(cov)
    e1 = evecs[:, 2]
    n = evecs[:, 0]
    if scan.vertex_normals is not None:
        mean_normal = scan.vertex_normals[idx].mean(axis=0)
        if np.dot(mean_normal, n) < 0:
            n = -n
    if flip:
        n = -n
    # 가로축(e1)이 대체로 +x 쪽을 향하도록 해 매번 같은 방향으로 보이게 한다.
    if e1[np.argmax(np.abs(e1))] < 0:
        e1 = -e1
    e2 = np.cross(n, e1)
    return origin, e1, e2, n, bool(scan.vertex_normals is not None)


def rotate_uv(u: np.ndarray, v: np.ndarray, degrees: float):
    if not degrees:
        return u, v
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    return c * u - s * v, s * u + c * v


def project_plane(scan: ScanData, flip: bool = False, rotation_deg: float = 0.0) -> Projection:
    origin, e1, e2, n, oriented = fit_plane_frame(scan, flip)
    d = scan.vertices - origin
    u, v = rotate_uv(d @ e1, d @ e2, rotation_deg)
    return Projection(
        u=u,
        v=v,
        h=d @ n,
        mode="plane",
        info={"normal": n.tolist(), "oriented_by_normals": oriented},
    )


# ---------------------------------------------------------------------------
# 원통 펼치기
# ---------------------------------------------------------------------------


def _axis_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def fit_cylinder(points: np.ndarray, normal_hint: np.ndarray):
    """점들에 원통을 맞춘다. (축 위의 점, 축 방향, 반지름) 을 돌려준다."""
    from scipy.optimize import least_squares

    origin = points.mean(axis=0)
    d = points - origin
    cov = np.cov(d.T)
    _, evecs = np.linalg.eigh(cov)
    n = normal_hint / np.linalg.norm(normal_hint)
    e1 = evecs[:, 2]
    e1 = e1 - np.dot(e1, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    x, y, z = d @ e1, d @ e2, d @ n

    # 2차 곡면 z = a x² + b xy + c y² + d x + e y + f 로 곡률 방향을 추정
    A = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    hess = np.array([[2 * coef[0], coef[1]], [coef[1], 2 * coef[2]]])
    kvals, kvecs = np.linalg.eigh(hess)
    i_curv = int(np.argmax(np.abs(kvals)))
    k = float(kvals[i_curv])
    extent = float(np.ptp(x) + np.ptp(y)) / 2
    if abs(k) < 1e-9:
        k = 1e-9
    radius0 = 1.0 / abs(k)
    if radius0 > 100 * max(extent, 1.0):
        raise ValueError("표면이 거의 평평해서 원통으로 펼칠 수 없습니다. '평면'을 사용하세요.")
    axis_dir_2d = kvecs[:, 1 - i_curv]
    axis0 = axis_dir_2d[0] * e1 + axis_dir_2d[1] * e2
    # 포물선 꼭짓점 근처에서 법선 방향으로 반지름만큼 떨어진 곳이 축
    center0 = origin + n * (np.sign(k) * radius0)

    # 축 방향을 구면 좌표로, 축 위의 점은 축에 수직인 2자유도로 표현
    theta0 = np.arccos(np.clip(axis0[2], -1, 1))
    phi0 = np.arctan2(axis0[1], axis0[0])

    def unpack(p):
        a = _axis_from_angles(p[0], p[1])
        # 축에 수직인 기저
        tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        b1 = np.cross(a, tmp)
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(a, b1)
        c = center0 + p[2] * b1 + p[3] * b2
        return c, a, p[4]

    def residual(p):
        c, a, r = unpack(p)
        w = points - c
        radial = w - np.outer(w @ a, a)
        return np.linalg.norm(radial, axis=1) - r

    p0 = np.array([theta0, phi0, 0.0, 0.0, radius0])
    sol = least_squares(residual, p0, loss="soft_l1", f_scale=max(radius0 * 1e-3, 0.05))
    c, a, r = unpack(sol.x)
    return c, a / np.linalg.norm(a), float(abs(r))


def project_cylinder(scan: ScanData, flip: bool = False, rotation_deg: float = 0.0) -> Projection:
    origin, e1, e2, n, oriented = fit_plane_frame(scan, flip)
    idx = _subsample(scan.n_points, 30_000, seed=1)
    center, axis, radius = fit_cylinder(scan.vertices[idx], n)

    # 기준 방향: 축에서 표면 중심을 향하는 방향
    w0 = origin - center
    r0 = w0 - np.dot(w0, axis) * axis
    r0 /= np.linalg.norm(r0)
    convex = np.dot(n, r0) > 0  # 바깥에서 볼록한 면을 보는 경우
    b0 = np.cross(axis, r0)

    w = scan.vertices - center
    t = w @ axis
    radial = w - np.outer(t, axis)
    rho = np.linalg.norm(radial, axis=1)
    ang = np.arctan2(radial @ b0, radial @ r0)
    if convex:
        u, h = radius * ang, rho - radius
    else:
        u, h = -radius * ang, radius - rho
    v = t
    # 원통 축은 세로(v), 펼친 호는 가로(u)로 둔다. 둥근 비석은 보통 축이 세로다.
    # 축 방향 부호는 정할 수 없으므로 평면 기준 e2(세로)와 맞춘다 (180° 회전만, 좌우 반전 없음).
    if np.dot(axis, e2) < 0:
        u, v = -u, -v
    u, v = rotate_uv(u, v, rotation_deg)
    return Projection(
        u=u,
        v=v,
        h=h,
        mode="cylinder",
        info={
            "radius_mm": radius,
            "axis": axis.tolist(),
            "convex": bool(convex),
            "oriented_by_normals": oriented,
        },
    )


def curvature_sagitta(proj: Projection) -> float:
    """평면 투영 결과가 얼마나 휘었는지: 가운데와 가장자리의 높이 차(mm)."""
    idx = _subsample(len(proj.u), 50_000, seed=2)
    x, y, z = proj.u[idx], proj.v[idx], proj.h[idx]
    A = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    hess = np.array([[2 * coef[0], coef[1]], [coef[1], 2 * coef[2]]])
    kvals, kvecs = np.linalg.eigh(hess)
    i = int(np.argmax(np.abs(kvals)))
    direction = kvecs[:, i]
    span = float(np.ptp(x * direction[0] + y * direction[1]))
    return abs(float(kvals[i])) * (span / 2) ** 2 / 2


def project(scan: ScanData, mode: str = "plane", flip: bool = False, rotation_deg: float = 0.0) -> Projection:
    """mode: plane(평면) / cylinder(원통 펼치기) / auto(휘어짐을 보고 자동 선택)."""
    if mode == "cylinder":
        return project_cylinder(scan, flip, rotation_deg)
    plane = project_plane(scan, flip, rotation_deg)
    if mode == "auto":
        sag = curvature_sagitta(plane)
        plane.info["sagitta_mm"] = sag
        if sag > 2.0:
            try:
                cyl = project_cylinder(scan, flip, rotation_deg)
                cyl.info["sagitta_mm"] = sag
                cyl.info["auto"] = True
                return cyl
            except Exception:
                pass
        plane.info["auto"] = True
    return plane


# ---------------------------------------------------------------------------
# 격자로 만들기
# ---------------------------------------------------------------------------


def auto_pixel_size(scan: ScanData, proj: Projection | None = None) -> float:
    """스캔 점 간격에 맞는 해상도(mm/픽셀)를 고른다."""
    spacing = scan.point_spacing()
    px = max(spacing, 0.02)
    if proj is not None:
        w = float(np.ptp(proj.u))
        h = float(np.ptp(proj.v))
        while (w / px) * (h / px) > MAX_GRID_PIXELS:
            px *= 1.25
    return float(round(px, 4))


def _masked_blur(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    m = mask.astype(np.float32)
    num = ndimage.gaussian_filter(values * m, sigma)
    den = ndimage.gaussian_filter(m, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 1e-6, out, 0).astype(np.float32)


def rasterize(
    proj: Projection,
    pixel_mm: float,
    colors: np.ndarray | None = None,
    attributes: dict | None = None,
    point_spacing: float | None = None,
    fill_holes_mm2: float = 25.0,
) -> HeightMap:
    """펼친 점들을 격자에 모아 평균 높이를 만든다. 빈 칸은 주변 값으로 채운다."""
    if pixel_mm <= 0:
        raise ValueError("해상도는 0보다 커야 합니다.")
    u, v, h = proj.u, proj.v, proj.h
    u0, u1 = float(u.min()), float(u.max())
    v0, v1 = float(v.min()), float(v.max())
    w = int(np.floor((u1 - u0) / pixel_mm)) + 1
    hgt = int(np.floor((v1 - v0) / pixel_mm)) + 1
    if w * hgt > MAX_GRID_PIXELS * 1.05:
        raise ValueError(
            f"해상도가 너무 높습니다 ({w}x{hgt} 픽셀). mm/픽셀 값을 더 크게 설정하세요."
        )
    col = np.clip(((u - u0) / pixel_mm).astype(np.int64), 0, w - 1)
    row = np.clip(((v1 - v) / pixel_mm).astype(np.int64), 0, hgt - 1)
    flat = row * w + col
    n_cells = w * hgt

    counts = np.bincount(flat, minlength=n_cells).astype(np.float64)
    occ = counts > 0

    def cell_mean(values):
        s = np.bincount(flat, weights=values, minlength=n_cells)
        out = np.zeros(n_cells, dtype=np.float64)
        out[occ] = s[occ] / counts[occ]
        return out.reshape(hgt, w)

    height = cell_mean(h).astype(np.float32)
    occ2d = occ.reshape(hgt, w)

    # 유효 영역: 점이 있는 칸을 조금 닫아서(closing) 점 사이 틈을 메운다.
    spacing = point_spacing if point_spacing else pixel_mm
    gap_px = max(1, int(np.ceil(1.5 * spacing / pixel_mm)))
    struct = ndimage.generate_binary_structure(2, 1)
    valid = ndimage.binary_closing(occ2d, structure=struct, iterations=gap_px, border_value=0)
    valid |= occ2d
    # 작은 구멍(스캔 누락)도 메운다.
    filled = ndimage.binary_fill_holes(valid)
    holes = filled & ~valid
    if holes.any():
        lab, nlab = ndimage.label(holes)
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
        max_px = fill_holes_mm2 / (pixel_mm * pixel_mm)
        small = np.zeros(nlab + 1, dtype=bool)
        small[1:] = sizes <= max_px
        valid |= small[lab]

    # 빈 칸 채우기: 가장 가까운 값 → 부드럽게 보간
    need = valid & ~occ2d
    nearest_idx = None
    if need.any():
        _, nearest_idx = ndimage.distance_transform_edt(~occ2d, return_indices=True)

    sigma = max(1.0, gap_px / 2)
    far = None
    if nearest_idx is not None:
        # 블러가 닿지 않는 큰 빈칸은 가장 가까운 값을 쓴다.
        far = need & (ndimage.gaussian_filter(occ2d.astype(np.float32), sigma) < 1e-3)

    def fill(arr):
        if nearest_idx is None:
            return arr
        smooth = _masked_blur(arr.astype(np.float32), occ2d, sigma)
        out = np.where(need, smooth, arr)
        return np.where(far, arr[nearest_idx[0], nearest_idx[1]], out)

    height = fill(height).astype(np.float32)
    height[~valid] = 0

    color_grid = None
    if colors is not None:
        chans = [fill(cell_mean(colors[:, c].astype(np.float64))) for c in range(3)]
        color_grid = np.clip(np.stack(chans, axis=-1), 0, 255).astype(np.uint8)
        color_grid[~valid] = 0

    layers = {}
    for name, values in (attributes or {}).items():
        layers[name] = fill(cell_mean(np.asarray(values, dtype=np.float64))).astype(np.float32)

    return HeightMap(
        height=height,
        valid=valid,
        pixel_mm=float(pixel_mm),
        u0=u0,
        v0=v1,
        color=color_grid,
        layers=layers,
        info=dict(proj.info, mode=proj.mode),
    )
