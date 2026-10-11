"""3D 스캔 보기 (PyVista). 3D 그래픽을 쓸 수 없는 환경이면 안내 문구만 보여준다."""

from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..core.scan import ScanData

MAX_DISPLAY_POINTS = 3_000_000


class Viewer3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = None
        self._error = None
        if os.environ.get("STELE_NO_3D") != "1":
            try:
                os.environ.setdefault("QT_API", "pyside6")
                from pyvistaqt import QtInteractor

                self.plotter = QtInteractor(self)
                self.plotter.set_background("#3a3f44", top="#8a9199")
                layout.addWidget(self.plotter.interactor)
            except Exception as e:  # OpenGL이 없는 PC 등
                self._error = str(e)
                self.plotter = None
        self.label = QLabel(self._placeholder_text())
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        if self.plotter is None:
            layout.addWidget(self.label)

    def _placeholder_text(self) -> str:
        if self._error:
            return f"3D 보기를 사용할 수 없습니다.\n(그래픽 드라이버를 확인하세요)\n\n{self._error}"
        return "3D 보기를 사용할 수 없습니다. 2D 결과 탭은 정상적으로 쓸 수 있습니다."

    def show_scan(self, scan: ScanData, reset_camera: bool = True, use_color: bool = True):
        """스캔을 보여 준다.

        reset_camera=False면 보던 시점을 그대로 둔다 (깊이 편집 미리보기).
        use_color=False면 색 없이 음영만 보여 주어 굴곡(깊이)이 잘 보인다.
        """
        if self.plotter is None:
            return
        import pyvista as pv

        camera = None if reset_camera else self.plotter.camera_position
        self.plotter.clear()
        if scan.has_faces:
            faces = np.column_stack([np.full(len(scan.faces), 3), scan.faces]).ravel()
            mesh = pv.PolyData(scan.vertices, faces)
        else:
            pts = scan.vertices
            idx = None
            if len(pts) > MAX_DISPLAY_POINTS:
                idx = np.random.default_rng(0).choice(len(pts), MAX_DISPLAY_POINTS, replace=False)
                pts = pts[idx]
            mesh = pv.PolyData(pts)
        if scan.colors is not None and use_color:
            colors = scan.colors if scan.has_faces or idx is None else scan.colors[idx]
            mesh.point_data["RGB"] = colors
            self.plotter.add_mesh(mesh, scalars="RGB", rgb=True, point_size=2)
        else:
            self.plotter.add_mesh(mesh, color="#d8d2c4", smooth_shading=True, point_size=2)
        self.plotter.add_axes()
        if camera is None:
            self.plotter.reset_camera()
        else:
            self.plotter.camera_position = camera
        self.plotter.render()

    @property
    def available(self) -> bool:
        return self.plotter is not None

    def clear(self):
        if self.plotter is not None:
            self.plotter.clear()

    def close_viewer(self):
        if self.plotter is not None:
            self.plotter.close()
