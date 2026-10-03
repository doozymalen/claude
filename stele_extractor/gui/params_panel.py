"""오른쪽 설정 패널."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.pipeline import GeometryParams, ReliefParams
from .widgets import FloatSlider

SURFACE_MODES = [("auto", "자동 판단"), ("plane", "평면"), ("cylinder", "원통 펼치기 (둥근 면)")]
LETTER_TYPES = [("auto", "자동 판단"), ("engraved", "음각 (파인 글자)"), ("raised", "양각 (솟은 글자)")]
UNITS = [("mm", "mm (Revo Scan 기본)"), ("cm", "cm"), ("m", "m")]


def _combo(items):
    c = QComboBox()
    for key, text in items:
        c.addItem(text, key)
    return c


def _set_combo(c: QComboBox, key):
    i = c.findData(key)
    if i >= 0:
        c.setCurrentIndex(i)


class ParamsPanel(QWidget):
    geometryChanged = Signal()
    reliefChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)

        # --- 3D → 2D ---------------------------------------------------
        g = QGroupBox("1. 스캔 펼치기 (3D → 2D)")
        f = QFormLayout(g)
        self.unit = _combo(UNITS)
        f.addRow("파일 단위", self.unit)
        self.surface = _combo(SURFACE_MODES)
        self.surface.setToolTip("휘어진 비석은 '원통 펼치기'를 쓰면 글자가 왜곡 없이 펴집니다.")
        f.addRow("표면 형태", self.surface)
        res_row = QHBoxLayout()
        self.auto_res = QCheckBox("자동")
        self.auto_res.setChecked(True)
        self.pixel = QDoubleSpinBox()
        self.pixel.setRange(0.01, 5.0)
        self.pixel.setDecimals(3)
        self.pixel.setSingleStep(0.05)
        self.pixel.setValue(0.2)
        self.pixel.setSuffix(" mm/픽셀")
        self.pixel.setEnabled(False)
        res_row.addWidget(self.auto_res)
        res_row.addWidget(self.pixel, 1)
        f.addRow("해상도", res_row)
        self.flip = QCheckBox("앞뒤 뒤집기 (음각/양각이 반대로 보일 때)")
        f.addRow(self.flip)
        rot_row = QHBoxLayout()
        self.rot_left = QPushButton("↺ 90°")
        self.rot_right = QPushButton("↻ 90°")
        self.rotation = QDoubleSpinBox()
        self.rotation.setRange(-360, 360)
        self.rotation.setDecimals(1)
        self.rotation.setSuffix("°")
        self.rotation.setWrapping(True)
        rot_row.addWidget(self.rot_left)
        rot_row.addWidget(self.rot_right)
        rot_row.addWidget(self.rotation, 1)
        f.addRow("회전", rot_row)
        self.auto_align = QCheckBox("기울기 자동 보정 (글자 획을 수평·수직으로)")
        self.auto_align.setChecked(True)
        f.addRow(self.auto_align)
        root.addWidget(g)

        # --- 글자 찾기 ---------------------------------------------------
        g = QGroupBox("2. 글자 찾기")
        f = QFormLayout(g)
        self.letter = _combo(LETTER_TYPES)
        f.addRow("글자 형태", self.letter)
        bg_row = QHBoxLayout()
        self.auto_bg = QCheckBox("자동")
        self.auto_bg.setChecked(True)
        self.background = FloatSlider(2, 80, 0.5, 10, "{:.1f}", " mm")
        self.background.setEnabled(False)
        self.background.setToolTip("글자보다 큰 굴곡을 지웁니다. 글자 크기의 1~2배 정도가 좋습니다.")
        bg_row.addWidget(self.auto_bg)
        bg_row.addWidget(self.background, 1)
        f.addRow("배경 제거", bg_row)
        self.denoise = FloatSlider(0, 1.0, 0.02, 0.1, "{:.2f}", " mm")
        self.denoise.setToolTip("돌 표면의 자잘한 거칠기를 줄입니다.")
        f.addRow("잡음 제거", self.denoise)
        self.color_weight = FloatSlider(0, 100, 5, 0, "{:.0f}", " %")
        self.color_weight.setToolTip("글자에 색을 칠했거나 색 차이가 있을 때 색상 정보를 함께 씁니다.")
        f.addRow("색상 보조", self.color_weight)
        root.addWidget(g)

        # --- 이미지 ------------------------------------------------------
        g = QGroupBox("3. 이미지 보기")
        f = QFormLayout(g)
        self.contrast = FloatSlider(0.3, 3.0, 0.05, 1.0, "{:.2f}", " 배")
        f.addRow("대비", self.contrast)
        self.azimuth = FloatSlider(0, 360, 5, 315, "{:.0f}", "°")
        f.addRow("빛 방향", self.azimuth)
        self.altitude = FloatSlider(5, 80, 1, 35, "{:.0f}", "°")
        f.addRow("빛 높이", self.altitude)
        self.exaggeration = FloatSlider(1, 15, 0.5, 3, "{:.1f}", " 배")
        f.addRow("음영 강조", self.exaggeration)
        root.addWidget(g)

        # --- 흑백 글자 ----------------------------------------------------
        g = QGroupBox("4. 흑백 글자 추출")
        f = QFormLayout(g)
        self.sensitivity = FloatSlider(0, 100, 1, 50, "{:.0f}")
        self.sensitivity.setToolTip("높이면 얕은 글자까지 잡고, 낮추면 깊은 부분만 잡습니다.")
        f.addRow("감도", self.sensitivity)
        self.smooth = FloatSlider(0, 1.5, 0.05, 0.2, "{:.2f}", " mm")
        f.addRow("다듬기", self.smooth)
        self.min_area = FloatSlider(0, 30, 0.5, 1.0, "{:.1f}", " mm²")
        self.min_area.setToolTip("이보다 작은 점·얼룩은 지웁니다.")
        f.addRow("최소 크기", self.min_area)
        root.addWidget(g)

        self.reset_btn = QPushButton("기본값으로 되돌리기")
        root.addWidget(self.reset_btn)
        root.addStretch(1)

        # 신호 연결
        for w in (self.unit, self.surface):
            w.currentIndexChanged.connect(lambda _: self.geometryChanged.emit())
        self.auto_res.toggled.connect(self._on_auto_res)
        self.pixel.valueChanged.connect(lambda _: self.geometryChanged.emit())
        self.flip.toggled.connect(lambda _: self.geometryChanged.emit())
        self.auto_align.toggled.connect(lambda _: self.geometryChanged.emit())
        self.rotation.valueChanged.connect(lambda _: self.geometryChanged.emit())
        self.rot_left.clicked.connect(lambda: self._rotate(90))
        self.rot_right.clicked.connect(lambda: self._rotate(-90))

        self.letter.currentIndexChanged.connect(lambda _: self.reliefChanged.emit())
        self.auto_bg.toggled.connect(self._on_auto_bg)
        for s in (self.background, self.denoise, self.color_weight, self.contrast, self.azimuth,
                  self.altitude, self.exaggeration, self.sensitivity, self.smooth, self.min_area):
            s.valueChanged.connect(lambda _: self.reliefChanged.emit())
        self.reset_btn.clicked.connect(self.reset)

    # ------------------------------------------------------------------
    def _rotate(self, deg):
        v = self.rotation.value() + deg
        v = ((v + 180) % 360) - 180
        self.rotation.setValue(v)

    def _on_auto_res(self, on):
        self.pixel.setEnabled(not on)
        self.geometryChanged.emit()

    def _on_auto_bg(self, on):
        self.background.setEnabled(not on)
        self.reliefChanged.emit()

    def set_color_available(self, available: bool):
        self.color_weight.setEnabled(available)
        if not available:
            self.color_weight.setToolTip("이 스캔에는 색상 정보가 없습니다.")

    def set_pixel_size(self, px: float):
        """자동 해상도로 정해진 값을 화면에 보여준다 (신호 없이)."""
        self.pixel.blockSignals(True)
        self.pixel.setValue(px)
        self.pixel.blockSignals(False)

    def set_background_value(self, mm: float):
        if self.auto_bg.isChecked():
            self.background.blockSignals(True)
            self.background.setValue(mm)
            self.background.blockSignals(False)

    def unit_key(self) -> str:
        return self.unit.currentData()

    def set_unit(self, unit: str):
        self.unit.blockSignals(True)
        _set_combo(self.unit, unit)
        self.unit.blockSignals(False)

    def geometry_params(self) -> GeometryParams:
        return GeometryParams(
            surface_mode=self.surface.currentData(),
            pixel_mm=0.0 if self.auto_res.isChecked() else self.pixel.value(),
            flip=self.flip.isChecked(),
            rotation_deg=self.rotation.value(),
            auto_align=self.auto_align.isChecked(),
        )

    def relief_params(self) -> ReliefParams:
        return ReliefParams(
            letter_type=self.letter.currentData(),
            background_mm=0.0 if self.auto_bg.isChecked() else self.background.value(),
            denoise_mm=self.denoise.value(),
            color_weight=self.color_weight.value() / 100.0 if self.color_weight.isEnabled() else 0.0,
            contrast=self.contrast.value(),
            light_azimuth=self.azimuth.value(),
            light_altitude=self.altitude.value(),
            exaggeration=self.exaggeration.value(),
            sensitivity=self.sensitivity.value(),
            smooth_mm=self.smooth.value(),
            min_area_mm2=self.min_area.value(),
        )

    def set_params(self, gp: GeometryParams, rp: ReliefParams, unit: str | None = None):
        """저장된 설정을 화면에 적용한다 (신호는 마지막에 한 번만)."""
        widgets = [self.unit, self.surface, self.auto_res, self.pixel, self.flip, self.rotation, self.auto_align,
                   self.letter, self.auto_bg, self.background, self.denoise, self.color_weight,
                   self.contrast, self.azimuth, self.altitude, self.exaggeration, self.sensitivity,
                   self.smooth, self.min_area]
        for w in widgets:
            w.blockSignals(True)
        try:
            if unit:
                _set_combo(self.unit, unit)
            _set_combo(self.surface, gp.surface_mode)
            self.auto_res.setChecked(not gp.pixel_mm)
            self.pixel.setEnabled(bool(gp.pixel_mm))
            if gp.pixel_mm:
                self.pixel.setValue(gp.pixel_mm)
            self.flip.setChecked(gp.flip)
            self.rotation.setValue(gp.rotation_deg)
            self.auto_align.setChecked(gp.auto_align)
            _set_combo(self.letter, rp.letter_type)
            self.auto_bg.setChecked(not rp.background_mm)
            self.background.setEnabled(bool(rp.background_mm))
            if rp.background_mm:
                self.background.setValue(rp.background_mm)
            self.denoise.setValue(rp.denoise_mm)
            self.color_weight.setValue(rp.color_weight * 100)
            self.contrast.setValue(rp.contrast)
            self.azimuth.setValue(rp.light_azimuth)
            self.altitude.setValue(rp.light_altitude)
            self.exaggeration.setValue(rp.exaggeration)
            self.sensitivity.setValue(rp.sensitivity)
            self.smooth.setValue(rp.smooth_mm)
            self.min_area.setValue(rp.min_area_mm2)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def reset(self):
        self.set_params(GeometryParams(), ReliefParams())
        self.geometryChanged.emit()
