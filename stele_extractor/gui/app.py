"""프로그램 시작점."""

from __future__ import annotations

import os
import sys


def selftest() -> int:
    """화면 없이 핵심 기능을 한 번씩 실행해 본다 (설치 확인용: --selftest)."""
    import tempfile
    from pathlib import Path

    from ..core import depth_edit, export, model3d, ocr, vector
    from ..core.pipeline import GeometryParams, ReliefParams, build_geometry, process_relief
    from ..core.scan import load_scan
    from ..core.synthetic import SyntheticOptions, make_synthetic_scan, save_scan_ply

    out = Path(tempfile.mkdtemp(prefix="stele_selftest_"))
    scan = make_synthetic_scan(SyntheticOptions(text="忠孝\n仁義"))
    save_scan_ply(scan, out / "s.ply")
    scan = load_scan(out / "s.ply")
    proj, hm = build_geometry(scan, GeometryParams())
    r = process_relief(hm, ReliefParams())
    export.export_result(r, out, "t")
    outlines = vector.mask_to_outlines(r.mask, hm.pixel_mm)
    vector.save_dxf(outlines, out / "t.dxf", *hm.size_mm)
    polys = vector.outlines_to_polygons(outlines, hm.size_mm[1])
    mesh = model3d.make_print_model(r.mask, hm.valid, hm.pixel_mm, model3d.PrintOptions(kind="letters"),
                                    r.residual, polys)
    res = ocr.run_offline(r.images["mask"], r.mask, "hanja", "horizontal", auto_rotate=True)
    labels, n = depth_edit.label_letters(r.mask, hm.pixel_mm)
    edited = depth_edit.apply_edit(r, labels, n, depth_edit.DepthEdit(enabled=True, global_scale=2.0),
                                   ReliefParams())
    model3d.save_scan_mesh(depth_edit.displace_scan(scan, proj, edited), out / "edited.ply")
    import anthropic  # noqa: F401 - AI 판독 구성요소가 들어 있는지 확인
    import numpy as np
    import pyvista as pv
    import pyvista.plotting  # noqa: F401 - 3D 보기에 필요한 VTK 모듈이 들어 있는지 확인
    from pyvistaqt import QtInteractor  # noqa: F401

    pv.PolyData(np.random.rand(10, 3)).delaunay_2d()

    return (f"selftest OK: {hm.shape}, watertight={mesh.is_watertight}, ocr={res.text!r}, "
            f"letters={n}, out={out}")


def _run_selftest() -> int:
    """창 없는 실행 파일에서는 print가 안 보이므로 결과를 파일에도 쓴다 (--selftest-log=경로)."""
    import traceback

    log = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--selftest-log=")), None)
    try:
        msg, code = selftest(), 0
    except Exception:  # noqa: BLE001
        msg, code = "selftest FAILED\n" + traceback.format_exc(), 1
    if sys.stdout is not None:
        print(msg)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            f.write(msg + "\n")
    return code


def main():
    if "--selftest" in sys.argv:
        sys.exit(_run_selftest())
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("비석 글자 추출기")
    app.setOrganizationName("SteleExtractor")
    if sys.platform == "win32":
        app.setFont(QFont("Malgun Gothic", 9))
    win = MainWindow()
    win.show()
    # 명령줄로 파일을 주면 바로 연다 (예: 파일을 실행 파일 아이콘에 끌어다 놓기)
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            continue
        if arg.lower().endswith("project.json"):
            win.open_project(arg)
            break
        if os.path.splitext(arg)[1].lower() in (".ply", ".obj", ".stl"):
            win.load_scan_file(arg)
            break
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
