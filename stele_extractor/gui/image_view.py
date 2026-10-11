"""2D 결과 이미지 보기: 확대/축소, 이동, 영역 선택."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView


def numpy_to_qimage(img: np.ndarray) -> QImage:
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        h, w = img.shape
        q = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, _ = img.shape
        q = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
    return q.copy()


class ImageView(QGraphicsView):
    """마우스 휠 = 확대/축소, 드래그 = 이동, 선택 모드에서 드래그 = 영역 선택."""

    regionSelected = Signal(int, int, int, int)  # r0, r1, c0, c1 (픽셀) - 영역 선택 모드
    rectPicked = Signal(int, int, int, int, bool)  # 글자 고르기 모드에서 끌기 (r0, r1, c0, c1, Ctrl)
    clicked = Signal(int, int, bool)  # 글자 고르기 모드에서 클릭 (row, col, Ctrl)
    mouseMoved = Signal(int, int)  # row, col (이미지 밖이면 -1)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self._pixmap_item)
        self._sel_item = QGraphicsRectItem()
        pen = QPen(QColor(255, 60, 60), 0, Qt.DashLine)
        pen.setCosmetic(True)
        pen.setWidth(2)
        self._sel_item.setPen(pen)
        self._sel_item.setBrush(QColor(255, 60, 60, 40))
        self._sel_item.setZValue(10)
        self._sel_item.hide()
        self.scene().addItem(self._sel_item)

        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor(90, 90, 90))
        self.setMouseTracking(True)
        self._mode = None  # None(이동) / "crop"(영역 선택) / "pick"(글자 고르기)
        self._sel_start: QPointF | None = None
        self._press_pos = None
        self._shape: tuple[int, int] | None = None

    # -- 이미지 -----------------------------------------------------------
    def set_image(self, img: np.ndarray | None, fit: bool = False):
        if img is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._shape = None
            return
        first = self._shape is None or self._shape != img.shape[:2]
        self._pixmap_item.setPixmap(QPixmap.fromImage(numpy_to_qimage(img)))
        self._shape = img.shape[:2]
        self.scene().setSceneRect(QRectF(0, 0, img.shape[1], img.shape[0]))
        if fit or first:
            self.fit()

    def fit(self):
        if self._shape:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def zoom(self, factor: float):
        self.scale(factor, factor)

    def wheelEvent(self, event):
        self.zoom(1.25 if event.angleDelta().y() > 0 else 0.8)

    # -- 영역 선택 / 글자 고르기 ---------------------------------------------
    def set_select_mode(self, on: bool):
        self.set_mode("crop" if on else None)

    def set_pick_mode(self, on: bool):
        self.set_mode("pick" if on else None)

    def set_mode(self, mode):
        self._mode = mode
        self.setDragMode(QGraphicsView.NoDrag if mode else QGraphicsView.ScrollHandDrag)
        self.viewport().setCursor(
            Qt.CrossCursor if mode == "crop" else Qt.PointingHandCursor if mode == "pick" else Qt.OpenHandCursor
        )
        self._sel_start = None
        self._sel_item.hide()

    def mousePressEvent(self, event):
        if self._mode and event.button() == Qt.LeftButton and self._shape:
            self._press_pos = event.position().toPoint()
            self._sel_start = self.mapToScene(self._press_pos)
            self._sel_item.setRect(QRectF(self._sel_start, self._sel_start))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        p = self.mapToScene(event.position().toPoint())
        if self._shape and 0 <= p.x() < self._shape[1] and 0 <= p.y() < self._shape[0]:
            self.mouseMoved.emit(int(p.y()), int(p.x()))
        else:
            self.mouseMoved.emit(-1, -1)
        if self._mode and self._sel_start is not None:
            self._sel_item.setRect(QRectF(self._sel_start, p).normalized())
            self._sel_item.show()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._mode and self._sel_start is not None and event.button() == Qt.LeftButton:
            ctrl = bool(event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
            moved = (event.position().toPoint() - self._press_pos).manhattanLength() if self._press_pos else 0
            rect = self._sel_item.rect().intersected(self.scene().sceneRect())
            start = self._sel_start
            self._sel_start = None
            self._sel_item.hide()
            if self._mode == "pick" and moved < 5:
                self.clicked.emit(int(start.y()), int(start.x()), ctrl)
            elif rect.width() >= 3 and rect.height() >= 3:
                box = (int(rect.top()), int(rect.bottom()), int(rect.left()), int(rect.right()))
                if self._mode == "pick":
                    self.rectPicked.emit(*box, ctrl)
                else:
                    self.regionSelected.emit(*box)
            return
        super().mouseReleaseEvent(event)
