"""글자 깊이 편집 시험."""

import numpy as np
import pytest

from stele_extractor.core import depth_edit as de
from stele_extractor.core import model3d
from stele_extractor.core.pipeline import GeometryParams, ReliefParams, build_geometry, process_relief
from stele_extractor.core.project import Project, load_project, save_project
from stele_extractor.core.scan import load_scan
from stele_extractor.core.synthetic import SyntheticOptions, make_synthetic_scan


def prepare(surface="flat", letter_type="engraved"):
    scan = make_synthetic_scan(SyntheticOptions(surface=surface, letter_type=letter_type, curve_radius_mm=90))
    gp, rp = GeometryParams(), ReliefParams()
    proj, hm = build_geometry(scan, gp)
    result = process_relief(hm, rp)
    labels, n = de.label_letters(result.mask, hm.pixel_mm)
    return scan, gp, rp, proj, hm, result, labels, n


def rescan_depth(scan, proj, edited, gp, rp):
    """깊이를 바꾼 스캔을 처음부터 다시 처리해 글자 깊이를 잰다."""
    moved = de.displace_scan(scan, proj, edited)
    _, hm2 = build_geometry(moved, gp)
    r2 = process_relief(hm2, rp)
    lab2, n2 = de.label_letters(r2.mask, hm2.pixel_mm)
    return r2, de.letter_depth_mm(r2, lab2) if n2 else 0.0


def test_letters_are_grouped_per_character():
    *_, labels, n = prepare()
    assert n == 4  # 大韓民國


@pytest.mark.parametrize("surface", ["flat", "curved"])
@pytest.mark.parametrize("scale", [2.0, 0.5])
def test_scale_changes_real_depth(surface, scale):
    scan, gp, rp, proj, hm, result, labels, n = prepare(surface)
    before = de.letter_depth_mm(result, labels)
    edited = de.apply_edit(result, labels, n, de.DepthEdit(enabled=True, global_scale=scale), rp)
    assert de.letter_depth_mm(edited, labels) == pytest.approx(before * scale, rel=0.05)
    r2, after = rescan_depth(scan, proj, edited, gp, rp)
    assert r2.polarity == "engraved"
    assert after == pytest.approx(before * scale, rel=0.12)


@pytest.mark.parametrize("letter_type,flipped", [("engraved", "raised"), ("raised", "engraved")])
def test_negative_scale_flips_engraved_and_raised(letter_type, flipped):
    scan, gp, rp, proj, hm, result, labels, n = prepare(letter_type=letter_type)
    edited = de.apply_edit(result, labels, n, de.DepthEdit(enabled=True, global_scale=-1.0), rp)
    assert de.letter_depth_mm(edited, labels) < 0
    r2, after = rescan_depth(scan, proj, edited, gp, rp)
    assert r2.polarity == flipped
    assert after == pytest.approx(de.letter_depth_mm(result, labels), rel=0.15)


def test_background_untouched_in_keep_mode():
    _, _, rp, _, hm, result, labels, n = prepare()
    edited = de.apply_edit(result, labels, n, de.DepthEdit(enabled=True, global_scale=3.0), rp)
    far = ~np.isin(labels, list(range(1, n + 1)))
    import cv2

    far = cv2.erode(far.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & hm.valid
    assert np.abs(edited.delta[far]).max() < 1e-6


def test_background_smoothing_reduces_roughness():
    _, _, rp, _, hm, result, labels, n = prepare()
    edited = de.apply_edit(result, labels, n, de.DepthEdit(enabled=True, bg_mode="smooth", bg_smooth=0.8), rp)
    bg = (labels == 0) & hm.valid
    assert np.std(edited.residual[bg]) < 0.5 * np.std(result.residual[bg])


def test_per_letter_scale_and_restore():
    _, _, rp, _, hm, result, labels, n = prepare()
    edit = de.DepthEdit(enabled=True, global_scale=1.0)
    de.set_letter_scale(edit, labels, hm, [2], 3.0)
    edited = de.apply_edit(result, labels, n, edit, rp)
    depths = [de.letter_depth_mm(edited, labels, [i]) for i in range(1, n + 1)]
    base = [de.letter_depth_mm(result, labels, [i]) for i in range(1, n + 1)]
    assert depths[1] == pytest.approx(base[1] * 3, rel=0.1)
    for i in (0, 2, 3):
        assert depths[i] == pytest.approx(base[i], rel=0.1)
    # 다시 처리해 글자 번호가 새로 매겨져도 같은 글자를 찾는다
    table = de.letter_scales(edit, labels, n, hm)
    assert table[2] == 3.0 and table[1] == 1.0
    # 글자별 설정 지우기
    de.set_letter_scale(edit, labels, hm, [2], None)
    assert edit.overrides == []


def test_edited_print_and_cut_models(tmp_path):
    scan, gp, rp, proj, hm, result, labels, n = prepare()
    edited = de.apply_edit(result, labels, n, de.DepthEdit(enabled=True, global_scale=2.0), rp)
    mesh = model3d.make_print_model(edited.mask, hm.valid, hm.pixel_mm, model3d.PrintOptions(kind="relief"),
                                    edited.residual)
    assert mesh.is_watertight and mesh.is_volume
    moved = de.displace_scan(scan, proj, edited)
    model3d.save_scan_mesh(moved, tmp_path / "m.ply")
    back = load_scan(tmp_path / "m.ply")
    assert back.n_points == scan.n_points and back.colors is not None


def test_edit_saved_in_project(tmp_path):
    e = de.DepthEdit(enabled=True, global_scale=2.5, overrides=[[1.0, 2.0, -1.0]], bg_mode="smooth")
    p = Project(folder=tmp_path / "p", depth_edit=e.to_dict())
    save_project(p)
    q = de.DepthEdit.from_dict(load_project(tmp_path / "p").depth_edit)
    assert q == e


def test_identity_returns_original():
    _, _, rp, _, _, result, labels, n = prepare()
    assert de.apply_edit(result, labels, n, de.DepthEdit(enabled=True), rp) is result
    assert de.apply_edit(result, labels, n, de.DepthEdit(enabled=False, global_scale=2), rp) is result
