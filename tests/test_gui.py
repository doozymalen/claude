"""화면(GUI) 전체 흐름 시험: 창을 띄우지 않고(offscreen) 실행한다."""

import json
import os
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("STELE_NO_3D", "1")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from stele_extractor.core.synthetic import SyntheticOptions, make_synthetic_scan, save_scan_ply  # noqa: E402
from stele_extractor.gui import main_window as mw  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def window(app, monkeypatch):
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))
    w = mw.MainWindow()
    w.errors = []
    w._error = lambda t, m, d="": w.errors.append((t, m, d))
    w.show()
    yield w
    w.dirty = False
    w.close()


def wait(app, w, minimum=0.2, limit=180):
    t = time.time()
    while True:
        app.processEvents()
        QThreadPool.globalInstance().waitForDone(50)
        app.processEvents()
        idle = w._busy == 0 and not w._geo_timer.isActive() and not w._rel_timer.isActive()
        if idle and time.time() - t > minimum:
            return
        if time.time() - t > limit:
            raise TimeoutError("처리가 끝나지 않았습니다")


def test_full_workflow(app, window, tmp_path, monkeypatch):
    w = window
    ply = tmp_path / "비석.ply"
    save_scan_ply(make_synthetic_scan(SyntheticOptions(surface="curved", curve_radius_mm=90)), ply)

    w.load_scan_file(str(ply))
    wait(app, w)
    assert w.result is not None and not w.errors
    assert w.hm.info["mode"] == "cylinder"
    for i in range(w.image_combo.count()):
        w.image_combo.setCurrentIndex(i)
        app.processEvents()

    w.params.sensitivity.setValue(60)
    w.params._rotate(90)
    wait(app, w)
    w._on_region(10, 150, 10, 150)
    wait(app, w)
    assert w.crop_mm is not None and w.hm.shape == (140, 140)
    w.clear_crop()
    wait(app, w)
    assert w.crop_mm is None

    # 내보내기 (대화상자는 기본값으로 바로 확인)
    out = tmp_path / "out"
    for D in (mw.ExportImagesDialog, mw.VectorExportDialog, mw.Model3DDialog):
        monkeypatch.setattr(D, "exec", lambda self: 1)
    monkeypatch.setattr(w, "_default_out_dir", lambda: str(out))
    w.export_images()
    wait(app, w)
    w.export_vector()
    wait(app, w)
    orig_init = mw.Model3DDialog.__init__
    for btn in ("r_cut", "r_plate", "r_letters", "r_relief"):
        def init(self, *a, _b=btn, **k):
            orig_init(self, *a, **k)
            getattr(self, _b).setChecked(True)

        monkeypatch.setattr(mw.Model3DDialog, "__init__", init)
        w.export_3d()
        wait(app, w)
    names = {p.name for p in out.iterdir()}
    for expected in ("비석_탁본.png", "비석_외곽선.svg", "비석_외곽선.dxf", "비석_글자판.stl",
                     "비석_입체글자.stl", "비석_표면복제.stl", "비석_글자_원본표면.ply"):
        assert expected in names, expected
    assert not w.errors, w.errors

    # 오프라인 판독
    w.run_offline_ocr({"lang": "hanja", "direction": "horizontal", "auto_rotate": True, "source": "mask"})
    wait(app, w)
    assert len(w.ocr_panel.results()) == 1
    w.show_ocr_boxes()
    wait(app, w)

    # 프로젝트 저장 후 다시 열기
    proj_dir = tmp_path / "프로젝트" / "시험"
    monkeypatch.setattr(mw.NewProjectDialog, "exec", lambda self: 1)
    monkeypatch.setattr(mw.NewProjectDialog, "folder", lambda self: proj_dir)
    w.project.notes.name = "시험"
    assert w.save_project()
    data = json.loads((proj_dir / "project.json").read_text(encoding="utf-8"))
    assert data["notes"]["name"] == "시험"
    assert data["relief"]["sensitivity"] == 60
    assert (proj_dir / "scan" / "비석.ply").exists()

    w2 = mw.MainWindow()
    w2.errors = []
    w2._error = lambda t, m, d="": w2.errors.append((t, m, d))
    w2.open_project(str(proj_dir / "project.json"))
    wait(app, w2)
    assert w2.result is not None and not w2.errors
    assert w2.params.sensitivity.value() == 60
    assert len(w2.ocr_panel.results()) == 1
    assert not w2.dirty
    w2.close()


def test_synthetic_dialog_options(app):
    from stele_extractor.gui.dialogs import SyntheticDialog

    d = SyntheticDialog()
    d.surface.setCurrentIndex(1)
    o = d.options()
    assert o.surface == "curved" and o.text
