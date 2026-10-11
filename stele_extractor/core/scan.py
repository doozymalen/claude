"""스캔 파일(PLY/OBJ/STL) 불러오기와 공통 데이터 구조."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = (".ply", ".obj", ".stl")

# 단위 이름 → mm 배율
UNIT_SCALES = {"mm": 1.0, "cm": 10.0, "m": 1000.0}


@dataclass
class ScanData:
    """스캔 한 개의 정점/면/색상 정보. 좌표 단위는 항상 mm."""

    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray | None = None  # (M, 3) int64, 점군이면 None
    colors: np.ndarray | None = None  # (N, 3) uint8
    vertex_normals: np.ndarray | None = None  # (N, 3) float64
    source_path: str | None = None
    # 가상 데이터의 정답 등 정점별 부가 정보 (예: {"label": (N,) float})
    attributes: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.vertices)

    @property
    def has_faces(self) -> bool:
        return self.faces is not None and len(self.faces) > 0

    @property
    def extent_mm(self) -> np.ndarray:
        return self.vertices.max(axis=0) - self.vertices.min(axis=0)

    def scaled(self, factor: float) -> "ScanData":
        if factor == 1.0:
            return self
        return ScanData(
            vertices=self.vertices * factor,
            faces=self.faces,
            colors=self.colors,
            vertex_normals=self.vertex_normals,
            source_path=self.source_path,
            attributes=self.attributes,
        )

    def point_spacing(self) -> float:
        """정점 간 평균 간격(mm) 추정."""
        if self.has_faces:
            f = self.faces
            if len(f) > 200_000:
                f = f[np.random.default_rng(0).choice(len(f), 200_000, replace=False)]
            v = self.vertices
            e = np.concatenate(
                [
                    np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1),
                    np.linalg.norm(v[f[:, 1]] - v[f[:, 2]], axis=1),
                    np.linalg.norm(v[f[:, 2]] - v[f[:, 0]], axis=1),
                ]
            )
            e = e[e > 0]
            if len(e):
                return float(np.median(e))
        from scipy.spatial import cKDTree

        v = self.vertices
        sample = v
        if len(v) > 20_000:
            sample = v[np.random.default_rng(0).choice(len(v), 20_000, replace=False)]
        tree = cKDTree(v)
        d, _ = tree.query(sample, k=2)
        return float(np.median(d[:, 1]))


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """면 법선을 모아 정점 법선을 만든다."""
    v = vertices
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def _vertex_colors_from_visual(mesh) -> np.ndarray | None:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    try:
        kind = visual.kind
    except Exception:
        kind = None
    if kind is None:
        return None
    try:
        if kind == "texture":
            visual = visual.to_color()
        colors = np.asarray(visual.vertex_colors)
    except Exception:
        return None
    if colors.ndim != 2 or len(colors) != len(mesh.vertices):
        return None
    colors = colors[:, :3].astype(np.uint8)
    # 모든 정점이 같은 기본색이면 색 정보가 없는 것으로 본다.
    if np.all(colors == colors[0]):
        return None
    return colors


def load_scan(path: str | Path, unit: str = "mm") -> ScanData:
    """PLY/OBJ/STL 파일을 읽어 ScanData로 돌려준다."""
    import trimesh

    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.dump() if hasattr(g, "vertices")]
        if not geoms:
            raise ValueError("파일에 3D 데이터가 없습니다.")
        meshes = [g for g in geoms if isinstance(g, trimesh.Trimesh)]
        if meshes:
            loaded = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        else:
            loaded = geoms[0]

    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError("파일에 정점이 없습니다.")

    faces = None
    normals = None
    if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
        faces = np.asarray(loaded.faces, dtype=np.int64)
        try:
            normals = np.asarray(loaded.vertex_normals, dtype=np.float64)
        except Exception:
            normals = None
        colors = _vertex_colors_from_visual(loaded)
    else:
        colors = None
        raw = getattr(loaded, "colors", None)
        if raw is not None and len(raw) == len(vertices):
            colors = np.asarray(raw)[:, :3].astype(np.uint8)
            if np.all(colors == colors[0]):
                colors = None

    scan = ScanData(
        vertices=vertices,
        faces=faces,
        colors=colors,
        vertex_normals=normals,
        source_path=str(path),
    )
    return scan.scaled(UNIT_SCALES.get(unit, 1.0))


def guess_unit(scan: ScanData) -> str:
    """크기를 보고 단위를 추측한다 (Revo Scan 기본은 mm)."""
    size = float(scan.extent_mm.max())
    if size < 2.0:
        return "m"
    return "mm"
