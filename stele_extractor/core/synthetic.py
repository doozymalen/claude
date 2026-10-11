"""시험용 가상 비석 스캔 데이터 만들기.

실제 POP 4 스캔이 없을 때 음각/양각 글자가 새겨진 돌 표면을 흉내 낸다.
정점마다 정답 라벨(attributes["label"], 글자=1)을 같이 넣어 정확도 검사에 쓴다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .scan import ScanData, compute_vertex_normals

FONT_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\batang.ttc",
    r"C:\Windows\Fonts\gulim.ttc",
    r"C:\Windows\Fonts\msjh.ttc",  # 번체 한자
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    # Linux
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
]


@dataclass
class SyntheticOptions:
    text: str = "大韓\n民國"
    letter_type: str = "engraved"  # engraved(음각) / raised(양각)
    surface: str = "flat"  # flat(평면) / curved(곡면)
    width_mm: float = 80.0
    height_mm: float = 80.0
    spacing_mm: float = 0.25  # 정점 간격
    depth_mm: float = 1.2  # 글자 깊이(높이)
    curve_radius_mm: float = 150.0
    weathering: float = 0.3  # 0~1, 마모 정도
    roughness_mm: float = 0.08  # 돌 표면 거칠기
    noise_mm: float = 0.02  # 스캐너 잡음
    painted: bool = False  # 글자에 색을 칠했는지
    seed: int = 0
    font_path: str | None = None


def _font_has(font_path: str, text: str) -> bool:
    """글꼴에 text의 글자가 모두 들어 있는지 (없는 글자는 빈 네모 등 같은 모양으로 그려진다)."""
    chars = {c for c in text if not c.isspace()}
    if not chars:
        return True
    try:
        font = ImageFont.truetype(font_path, 48)
    except Exception:
        return False

    def render(ch):
        img = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(img).text((4, 4), ch, fill=255, font=font)
        return np.asarray(img)

    missing = render("\U0010fffd")
    for c in chars:
        g = render(c)
        if not g.any() or np.array_equal(g, missing):
            return False
    return True


def find_font(path: str | None = None, text: str = "") -> str | None:
    """text를 그릴 수 있는 글꼴을 찾는다."""
    if path and os.path.exists(path):
        return path
    existing = [p for p in FONT_CANDIDATES if os.path.exists(p)]
    for p in existing:
        if _font_has(p, text):
            return p
    return existing[0] if existing else None


def render_text_mask(text: str, width_px: int, height_px: int, font_path: str | None = None) -> np.ndarray:
    """글자 모양 마스크(bool)를 만든다. 글자가 영역의 약 80%를 채우도록 크기를 맞춘다."""
    lines = text.split("\n") or [" "]
    fp = find_font(font_path, text)

    def make_font(size):
        if fp:
            return ImageFont.truetype(fp, size)
        try:
            return ImageFont.load_default(size)
        except TypeError:
            return ImageFont.load_default()

    size = max(8, int(height_px / max(len(lines), 1) * 0.8))
    for _ in range(40):
        font = make_font(size)
        probe = Image.new("L", (10, 10))
        bbox = ImageDraw.Draw(probe).multiline_textbbox((0, 0), text, font=font, spacing=size // 6, align="center")
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= width_px * 0.8 and th <= height_px * 0.8:
            break
        size = int(size * 0.9)
    img = Image.new("L", (width_px, height_px), 0)
    draw = ImageDraw.Draw(img)
    x = (width_px - tw) / 2 - bbox[0]
    y = (height_px - th) / 2 - bbox[1]
    draw.multiline_text((x, y), text, fill=255, font=font, spacing=size // 6, align="center")
    return np.asarray(img) > 127


def make_synthetic_scan(opts: SyntheticOptions | None = None) -> ScanData:
    opts = opts or SyntheticOptions()
    rng = np.random.default_rng(opts.seed)
    sp = opts.spacing_mm
    nx = int(round(opts.width_mm / sp)) + 1
    ny = int(round(opts.height_mm / sp)) + 1

    # 글자 마스크 (행 0 = 위쪽)
    mask = render_text_mask(opts.text, nx, ny, opts.font_path)

    # 획 단면: V자 홈 (중심이 가장 깊다)
    dist_in = ndimage.distance_transform_edt(mask) * sp
    half = max(np.percentile(dist_in[mask], 95) if mask.any() else 1.0, sp)
    profile = np.clip(dist_in / (0.6 * half), 0, 1)
    profile = np.sqrt(profile)  # 가장자리는 가파르고 가운데는 평평하게
    # 마모: 위치마다 깊이가 다르고 모서리가 무뎌진다
    if opts.weathering > 0:
        wear = ndimage.gaussian_filter(rng.standard_normal((ny, nx)), 15.0 / sp)
        wear = (wear - wear.min()) / (np.ptp(wear) + 1e-9)
        profile = profile * (1 - opts.weathering * 0.7 * wear)
        profile = ndimage.gaussian_filter(profile, opts.weathering * 0.4 / sp)
    sign = -1.0 if opts.letter_type == "engraved" else 1.0
    letters = sign * opts.depth_mm * profile

    # 돌 표면: 거칠기 + 완만한 굴곡
    rough = ndimage.gaussian_filter(rng.standard_normal((ny, nx)), 0.4 / sp)
    rough *= opts.roughness_mm / (rough.std() + 1e-9)
    yy, xx = np.mgrid[0:ny, 0:nx] * sp
    wave = 0.4 * np.sin(xx / opts.width_mm * np.pi * 1.3 + 0.5) * np.cos(yy / opts.height_mm * np.pi * 0.8)
    z = letters + rough + wave + rng.normal(0, opts.noise_mm, (ny, nx))

    x = xx - opts.width_mm / 2
    y = (opts.height_mm / 2) - yy  # 위쪽이 +y
    if opts.surface == "curved":
        # 세로축 원통에 감는다 (x → 호 길이)
        r = opts.curve_radius_mm
        ang = x / r
        px = (r + z) * np.sin(ang)
        pz = (r + z) * np.cos(ang) - r
        py = y
    else:
        px, py, pz = x, y, z

    vertices = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])

    # 격자 → 삼각형 면
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.column_stack([a, c, b]), np.column_stack([b, c, d])])

    # 군데군데 스캔 누락(구멍)
    a = np.concatenate([a, b])
    keep = np.ones(len(faces), dtype=bool)
    for _ in range(3):
        cx, cy = rng.uniform(0, nx), rng.uniform(0, ny)
        rad = rng.uniform(1.0, 2.5) / sp
        keep &= np.hypot(a % nx - cx, a // nx - cy) > rad
    faces = faces[keep]

    # 색상: 회색 돌 + 반점, 칠한 글자는 흰색
    base = 120 + 25 * ndimage.gaussian_filter(rng.standard_normal((ny, nx)), 3.0 / sp)
    base += rng.normal(0, 8, (ny, nx))
    rgb = np.stack([base * 1.02, base, base * 0.95], axis=-1)
    if opts.painted:
        rgb[mask] = [235, 232, 220]
    else:
        # 파인 곳은 그림자/먼지로 약간 어둡다
        rgb *= (1 - 0.25 * profile)[..., None]
    colors = np.clip(rgb, 0, 255).astype(np.uint8).reshape(-1, 3)

    # 스캐너 좌표계처럼 임의로 돌리고 옮긴다
    angles = rng.uniform(-0.6, 0.6, 3)
    rot = _rotation_matrix(*angles)
    vertices = vertices @ rot.T + rng.uniform(-200, 200, 3)

    normals = compute_vertex_normals(vertices, faces)
    used = np.zeros(len(vertices), bool)
    used[faces.ravel()] = True
    remap = -np.ones(len(vertices), np.int64)
    remap[used] = np.arange(used.sum())

    return ScanData(
        vertices=vertices[used],
        faces=remap[faces],
        colors=colors[used],
        vertex_normals=normals[used],
        source_path=None,
        attributes={"label": mask.ravel().astype(np.float32)[used]},
    )


def _rotation_matrix(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def save_scan_ply(scan: ScanData, path: str | Path) -> None:
    """Revo Scan 내보내기와 같은 형식(색상 포함 바이너리 PLY)으로 저장."""
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=scan.vertices,
        faces=scan.faces,
        vertex_colors=None if scan.colors is None else np.column_stack(
            [scan.colors, np.full(len(scan.colors), 255, np.uint8)]
        ),
        process=False,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
