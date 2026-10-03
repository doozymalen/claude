"""공통으로 쓰는 작은 위젯들."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class FloatSlider(QWidget):
    """실수 값을 다루는 슬라이더 + 값 표시."""

    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, step: float, value: float,
                 fmt: str = "{:.2f}", suffix: str = "", parent=None):
        super().__init__(parent)
        self._min, self._step, self._fmt, self._suffix = minimum, step, fmt, suffix
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round((maximum - minimum) / step)))
        self.label = QLabel()
        self.label.setMinimumWidth(70)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label)
        self.slider.valueChanged.connect(self._on_change)
        self.setValue(value)
        self._update_label()

    def value(self) -> float:
        return self._min + self.slider.value() * self._step

    def setValue(self, v: float):
        self.slider.setValue(int(round((v - self._min) / self._step)))
        self._update_label()

    def _update_label(self):
        self.label.setText(self._fmt.format(self.value()) + self._suffix)

    def _on_change(self, _):
        self._update_label()
        self.valueChanged.emit(self.value())
