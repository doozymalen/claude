"""글자 깊이 편집 패널."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.depth_edit import DepthEdit
from .widgets import FloatSlider

SCALE_MIN, SCALE_MAX = -3.0, 5.0


def _scale_slider(value=1.0):
    return FloatSlider(SCALE_MIN, SCALE_MAX, 0.05, value, "×{:.2f}")


def describe(depth_mm: float, polarity: str) -> str:
    """깊이 숫자를 '음각 1.2 mm'처럼 읽기 쉽게."""
    if abs(depth_mm) < 0.02:
        return "평평함"
    kind = polarity if depth_mm > 0 else ("raised" if polarity == "engraved" else "engraved")
    name = "음각(파임)" if kind == "engraved" else "양각(솟음)"
    return f"{name} {abs(depth_mm):.2f} mm"


class DepthPanel(QWidget):
    editChanged = Signal()  # 배율/배경/경계 등 설정이 바뀜
    pickModeToggled = Signal(bool)
    selectedScaleChanged = Signal(float)
    resetSelected = Signal()
    clearSelection = Signal()
    exportEdited = Signal()
    view3dChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)

        intro = QLabel("찾아낸 글자의 깊이를 늘리거나 줄이고, 음각↔양각을 뒤집습니다. "
                       "바뀐 깊이는 2D 이미지, 3D 화면, 모든 내보내기에 적용됩니다.")
        intro.setWordWrap(True)
        root.addWidget(intro)
        self.enabled = QCheckBox("깊이 편집 켜기")
        root.addWidget(self.enabled)

        # 전체
        self.g_all = QGroupBox("모든 글자")
        f = QFormLayout(self.g_all)
        self.global_scale = _scale_slider()
        f.addRow("깊이 배율", self.global_scale)
        quick = QHBoxLayout()
        for text, v in (("뒤집기 ×-1", -1.0), ("평평 ×0", 0.0), ("×0.5", 0.5), ("원래 ×1", 1.0), ("×2", 2.0), ("×3", 3.0)):
            b = QPushButton(text)
            b.setMinimumWidth(10)
            b.clicked.connect(lambda _=False, v=v: self.global_scale.setValue(v))
            quick.addWidget(b)
        f.addRow(quick)
        self.global_info = QLabel()
        self.global_info.setWordWrap(True)
        f.addRow(self.global_info)
        hint = QLabel("1보다 크면 깊게, 0이면 평평하게, 음수면 음각과 양각이 뒤집힙니다.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        f.addRow(hint)
        root.addWidget(self.g_all)

        # 글자별
        self.g_sel = QGroupBox("글자별")
        f = QFormLayout(self.g_sel)
        self.pick_btn = QPushButton("2D 화면에서 글자 고르기")
        self.pick_btn.setCheckable(True)
        f.addRow(self.pick_btn)
        how = QLabel("클릭 = 글자 고르기, Ctrl+클릭 = 더하기/빼기, 끌기 = 사각형 안 글자 모두")
        how.setWordWrap(True)
        how.setStyleSheet("color: #666;")
        f.addRow(how)
        self.sel_info = QLabel("고른 글자 없음")
        self.sel_info.setWordWrap(True)
        f.addRow(self.sel_info)
        self.sel_scale = _scale_slider()
        f.addRow("고른 글자 배율", self.sel_scale)
        row = QHBoxLayout()
        self.sel_reset = QPushButton("전체 배율 따르기")
        self.sel_clear = QPushButton("고르기 해제")
        row.addWidget(self.sel_reset)
        row.addWidget(self.sel_clear)
        f.addRow(row)
        root.addWidget(self.g_sel)

        # 세부
        self.g_opt = QGroupBox("세부 설정")
        f = QFormLayout(self.g_opt)
        merge_row = QHBoxLayout()
        self.auto_merge = QCheckBox("자동")
        self.auto_merge.setChecked(True)
        self.merge = FloatSlider(0.5, 20, 0.5, 3.0, "{:.1f}", " mm")
        self.merge.setEnabled(False)
        self.merge.setToolTip("떨어진 획을 한 글자로 묶는 거리. 한 글자가 여러 조각으로 나뉘면 키우고, "
                              "여러 글자가 하나로 묶이면 줄이세요.")
        merge_row.addWidget(self.auto_merge)
        merge_row.addWidget(self.merge, 1)
        f.addRow("글자 묶음 거리", merge_row)
        self.feather = FloatSlider(0, 3, 0.05, 0.3, "{:.2f}", " mm")
        self.feather.setToolTip("글자 둘레를 얼마나 넓게, 부드럽게 이어 붙일지")
        f.addRow("경계 부드럽게", self.feather)
        self.bg_mode = QComboBox()
        self.bg_mode.addItem("그대로 두기 (원본 표면)", "keep")
        self.bg_mode.addItem("매끈하게 정리", "smooth")
        f.addRow("배경(돌 표면)", self.bg_mode)
        self.bg_smooth = FloatSlider(0, 100, 5, 70, "{:.0f}", " %")
        self.bg_smooth.setToolTip("배경의 자잘한 거칠기를 얼마나 줄일지 (큰 굴곡은 그대로)")
        f.addRow("매끈한 정도", self.bg_smooth)
        root.addWidget(self.g_opt)

        # 3D
        self.g_3d = QGroupBox("3D 보기")
        f = QFormLayout(self.g_3d)
        self.live3d = QCheckBox("3D 화면에 바로 반영")
        self.live3d.setChecked(True)
        self.no_color = QCheckBox("색 없이 음영으로 보기 (깊이가 잘 보임)")
        self.no_color.setChecked(True)
        f.addRow(self.live3d)
        f.addRow(self.no_color)
        root.addWidget(self.g_3d)

        self.export_btn = QPushButton("깊이 바꾼 원본 3D 저장 (PLY/OBJ)…")
        self.export_btn.setMinimumHeight(30)
        root.addWidget(self.export_btn)
        root.addStretch(1)

        # 신호
        self.enabled.toggled.connect(self._on_enabled)
        for w in (self.global_scale, self.feather, self.bg_smooth, self.merge):
            w.valueChanged.connect(lambda _: self.editChanged.emit())
        self.auto_merge.toggled.connect(self._on_auto_merge)
        self.bg_mode.currentIndexChanged.connect(self._on_bg_mode)
        self.pick_btn.toggled.connect(self.pickModeToggled.emit)
        self.sel_scale.valueChanged.connect(self.selectedScaleChanged.emit)
        self.sel_reset.clicked.connect(self.resetSelected.emit)
        self.sel_clear.clicked.connect(self.clearSelection.emit)
        self.export_btn.clicked.connect(self.exportEdited.emit)
        self.live3d.toggled.connect(lambda _: self.view3dChanged.emit())
        self.no_color.toggled.connect(lambda _: self.view3dChanged.emit())
        self._on_bg_mode()
        self._on_enabled(False)
        self.set_selection(0, None)

    # ------------------------------------------------------------------
    def _on_enabled(self, on):
        for g in (self.g_all, self.g_sel, self.g_opt, self.g_3d):
            g.setEnabled(on)
        self.export_btn.setEnabled(on)
        if not on and self.pick_btn.isChecked():
            self.pick_btn.setChecked(False)
        self.editChanged.emit()

    def _on_auto_merge(self, on):
        self.merge.setEnabled(not on)
        self.editChanged.emit()

    def _on_bg_mode(self, *_):
        self.bg_smooth.setEnabled(self.bg_mode.currentData() == "smooth")
        self.editChanged.emit()

    def set_selection(self, count: int, scale: float | None, info: str = ""):
        """고른 글자 수와 그 배율을 보여 준다 (신호 없이)."""
        self.sel_scale.blockSignals(True)
        if scale is not None:
            self.sel_scale.setValue(scale)
        self.sel_scale.blockSignals(False)
        has = count > 0
        self.sel_scale.setEnabled(has)
        self.sel_reset.setEnabled(has)
        self.sel_clear.setEnabled(has)
        self.sel_info.setText(f"고른 글자 {count}개" + (f" — {info}" if info else "") if has else "고른 글자 없음")

    def set_global_info(self, text: str):
        self.global_info.setText(text)

    def edit(self, overrides: list | None = None) -> DepthEdit:
        return DepthEdit(
            enabled=self.enabled.isChecked(),
            global_scale=self.global_scale.value(),
            overrides=list(overrides or []),
            merge_mm=0.0 if self.auto_merge.isChecked() else self.merge.value(),
            feather_mm=self.feather.value(),
            bg_mode=self.bg_mode.currentData(),
            bg_smooth=self.bg_smooth.value() / 100.0,
        )

    def set_edit(self, e: DepthEdit):
        widgets = [self.enabled, self.global_scale, self.auto_merge, self.merge, self.feather, self.bg_mode,
                   self.bg_smooth]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.enabled.setChecked(e.enabled)
            self.global_scale.setValue(e.global_scale)
            self.auto_merge.setChecked(not e.merge_mm)
            if e.merge_mm:
                self.merge.setValue(e.merge_mm)
            self.feather.setValue(e.feather_mm)
            i = self.bg_mode.findData(e.bg_mode)
            self.bg_mode.setCurrentIndex(max(i, 0))
            self.bg_smooth.setValue(e.bg_smooth * 100)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self.merge.setEnabled(not self.auto_merge.isChecked())
        self.bg_smooth.setEnabled(self.bg_mode.currentData() == "smooth")
        for g in (self.g_all, self.g_sel, self.g_opt, self.g_3d):
            g.setEnabled(e.enabled)
        self.export_btn.setEnabled(e.enabled)

    def show_merge_value(self, mm: float):
        if self.auto_merge.isChecked():
            self.merge.blockSignals(True)
            self.merge.setValue(mm)
            self.merge.blockSignals(False)
