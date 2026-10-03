"""핵심 처리 시험 (가상 비석 데이터 사용)."""

import json

import numpy as np
import pytest

from stele_extractor.core import export, model3d, vector
from stele_extractor.core.pipeline import GeometryParams, ReliefParams, build_geometry, process_relief
from stele_extractor.core.project import Project, ProjectNotes, crop_to_pixels, load_project, pixels_to_crop, save_project
from stele_extractor.core.scan import load_scan
from stele_extractor.core.synthetic import SyntheticOptions, make_synthetic_scan, save_scan_ply


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def run(opts: SyntheticOptions, gp: GeometryParams | None = None, rp: ReliefParams | None = None):
    scan = make_synthetic_scan(opts)
    proj, hm = build_geometry(scan, gp or GeometryParams())
    result = process_relief(hm, rp or ReliefParams())
    truth = (hm.layers["label"] > 0.5) & hm.valid
    return scan, proj, hm, result, truth


@pytest.mark.parametrize("letter_type", ["engraved", "raised"])
@pytest.mark.parametrize("surface", ["flat", "curved"])
def test_letters_found(letter_type, surface):
    _, _, hm, result, truth = run(
        SyntheticOptions(letter_type=letter_type, surface=surface, curve_radius_mm=90, seed=3)
    )
    assert result.polarity == letter_type
    assert iou(result.mask, truth) > 0.85
    if surface == "curved":
        assert hm.info["mode"] == "cylinder"
        assert abs(hm.info["radius_mm"] - 90) < 5
    else:
        assert hm.info["mode"] == "plane"


def test_curved_unrolled_size_is_true_length():
    """원통 펼치기 후 가로 길이가 실제 호 길이(80mm)와 같아야 한다."""
    _, _, hm, _, _ = run(
        SyntheticOptions(surface="curved", curve_radius_mm=70, width_mm=80, height_mm=50),
        GeometryParams(auto_align=False),
    )
    w, h = hm.size_mm
    assert abs(max(w, h) - 80) < 3
    assert abs(min(w, h) - 50) < 3


def test_weathered_shallow_letters():
    _, _, _, result, truth = run(SyntheticOptions(depth_mm=0.4, weathering=0.7, roughness_mm=0.06, seed=5))
    assert iou(result.mask, truth) > 0.6


def test_color_assist_with_painted_letters():
    _, _, _, result, truth = run(SyntheticOptions(depth_mm=0.3, painted=True), rp=ReliefParams(color_weight=0.6))
    assert iou(result.mask, truth) > 0.7


def test_flip_reverses_polarity():
    _, _, _, result, _ = run(SyntheticOptions(), GeometryParams(flip=True))
    assert result.polarity == "raised"  # 앞뒤가 바뀌면 음각이 양각으로 보인다


def test_ply_roundtrip(tmp_path):
    scan = make_synthetic_scan(SyntheticOptions(width_mm=40, height_mm=30))
    p = tmp_path / "s.ply"
    save_scan_ply(scan, p)
    loaded = load_scan(p)
    assert loaded.n_points == scan.n_points
    assert loaded.colors is not None and loaded.has_faces
    np.testing.assert_allclose(loaded.vertices, scan.vertices, atol=1e-4)
    cm = load_scan(p, unit="cm")
    np.testing.assert_allclose(cm.vertices, scan.vertices * 10, atol=1e-3)


def test_export_images_have_dpi(tmp_path):
    from PIL import Image

    _, _, hm, result, _ = run(SyntheticOptions(width_mm=40, height_mm=30))
    files = export.export_result(result, tmp_path, "t", ["rubbing", "mask"], "png", scale_bar=False)
    img = Image.open(files[0])
    assert img.size == (hm.shape[1], hm.shape[0])
    assert abs(img.info["dpi"][0] - 25.4 / hm.pixel_mm) < 0.5
    meta = json.loads((tmp_path / "t_정보.json").read_text(encoding="utf-8"))
    assert meta["해상도_mm_per_pixel"] == pytest.approx(hm.pixel_mm)
    depth = np.array(Image.open(tmp_path / "t_깊이값_mm.tif"))
    assert depth.dtype == np.float32


def test_vector_area_matches_mask(tmp_path):
    _, _, hm, result, _ = run(SyntheticOptions())
    outlines = vector.mask_to_outlines(result.mask, hm.pixel_mm)
    polys = vector.outlines_to_polygons(outlines, hm.size_mm[1])
    area = sum(p.area for p in polys)
    mask_area = result.mask.sum() * hm.pixel_mm ** 2
    assert abs(area - mask_area) / mask_area < 0.08
    w, h = hm.size_mm
    svg = vector.save_svg(outlines, tmp_path / "a.svg", w, h)
    assert f'width="{w:.3f}mm"' in svg.read_text(encoding="utf-8")
    import ezdxf

    doc = ezdxf.readfile(str(vector.save_dxf(outlines, tmp_path / "a.dxf", w, h)))
    assert doc.units == ezdxf.units.MM
    assert len(doc.modelspace().query("LWPOLYLINE[layer=='LETTERS']")) > 0


@pytest.mark.parametrize("kind", ["plate", "letters", "relief"])
def test_print_models_are_watertight(kind, tmp_path):
    _, _, hm, result, _ = run(SyntheticOptions(width_mm=50, height_mm=40))
    polys = None
    if kind == "letters":
        polys = vector.outlines_to_polygons(vector.mask_to_outlines(result.mask, hm.pixel_mm), hm.size_mm[1])
    for mirror in (False, True):
        opts = model3d.PrintOptions(kind=kind, mirror=mirror, border_mm=2)
        mesh = model3d.make_print_model(result.mask, hm.valid, hm.pixel_mm, opts, result.residual, polys)
        assert mesh.is_watertight and mesh.is_volume
    model3d.save_mesh(mesh, tmp_path / "m.stl")
    assert (tmp_path / "m.stl").stat().st_size > 1000


def test_cut_letters_from_scan(tmp_path):
    scan, proj, hm, result, _ = run(SyntheticOptions(width_mm=50, height_mm=40))
    part = model3d.cut_letters_from_scan(scan, proj, hm, result.mask, margin_mm=0.3)
    assert 0 < len(part.faces) < len(scan.faces) * 0.5
    # 잘라낸 정점은 대부분 글자 정답 위에 있어야 한다
    model3d.save_scan_mesh(part, tmp_path / "c.ply")
    assert load_scan(tmp_path / "c.ply").n_points == part.n_points


def test_crop_roundtrip_and_project(tmp_path):
    scan = make_synthetic_scan(SyntheticOptions(width_mm=40, height_mm=30))
    ply = tmp_path / "scan.ply"
    save_scan_ply(scan, ply)
    _, hm = build_geometry(scan, GeometryParams())
    crop = pixels_to_crop(10, 60, 20, 90, hm.u0, hm.v0, hm.pixel_mm)
    assert crop_to_pixels(crop, hm.u0, hm.v0, hm.pixel_mm) == (10, 60, 20, 90)

    p = Project(folder=tmp_path / "프로젝트", scan_path=str(ply), crop_mm=crop,
                notes=ProjectNotes(name="시험 비석", location="서울"),
                geometry=GeometryParams(rotation_deg=90), relief=ReliefParams(sensitivity=70))
    p.ocr_results.append({"engine": "x", "text": "大韓", "lines": [], "created": "", "options": {}})
    save_project(p, preview=np.zeros((10, 10), np.uint8), copy_scan=True)
    q = load_project(tmp_path / "프로젝트")
    assert q.notes.name == "시험 비석"
    assert q.geometry.rotation_deg == 90 and q.relief.sensitivity == 70
    assert q.crop_mm == pytest.approx(crop)
    assert q.resolve_scan_path() == tmp_path / "프로젝트" / "scan" / "scan.ply"
    assert q.ocr_results[0]["text"] == "大韓"
