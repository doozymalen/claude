"""글자 판독(OCR) 패널."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..core import ocr


class OcrPanel(QWidget):
    runOffline = Signal(dict)
    runAi = Signal(dict)
    downloadKorean = Signal()
    textEdited = Signal()
    showBoxes = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)

        g = QGroupBox("판독 방법")
        f = QFormLayout(g)
        self.r_offline = QRadioButton("오프라인 OCR (무료, 인터넷 불필요)")
        self.r_ai = QRadioButton("AI 판독 (Claude, 인터넷·API 키 필요)")
        self.r_offline.setChecked(True)
        f.addRow(self.r_offline)
        f.addRow(self.r_ai)
        root.addWidget(g)

        # 오프라인 설정
        self.g_off = QGroupBox("오프라인 OCR 설정")
        f = QFormLayout(self.g_off)
        self.lang = QComboBox()
        for k, v in ocr.OFFLINE_LANGS.items():
            self.lang.addItem(v, k)
        kor_row = QHBoxLayout()
        self.kor_status = QLabel()
        self.kor_btn = QPushButton("한국어 모델 내려받기")
        self.kor_btn.clicked.connect(self.downloadKorean.emit)
        kor_row.addWidget(self.kor_status, 1)
        kor_row.addWidget(self.kor_btn)
        self.direction = QComboBox()
        self.direction.addItem("세로쓰기 (오른쪽 열부터)", "vertical")
        self.direction.addItem("가로쓰기 (위 줄부터)", "horizontal")
        self.auto_rot = QCheckBox("글자 방향 자동 찾기 (90° 단위로 돌려 시험)")
        self.auto_rot.setChecked(True)
        self.source = QComboBox()
        self.source.addItem("글자 추출(흑백)", "mask")
        self.source.addItem("글자 강조", "enhanced")
        f.addRow("문자", self.lang)
        f.addRow(kor_row)
        f.addRow("쓰기 방향", self.direction)
        f.addRow(self.auto_rot)
        f.addRow("읽을 이미지", self.source)
        root.addWidget(self.g_off)

        # AI 설정
        self.g_ai = QGroupBox("AI 판독 설정")
        f = QFormLayout(self.g_ai)
        self.ai_images = QComboBox()
        self.ai_images.addItem("글자 강조 + 음영 기복 + 탁본 (권장)", "multi")
        self.ai_images.addItem("글자 강조만", "enhanced")
        self.ai_images.addItem("탁본만", "rubbing")
        self.hint = QLineEdit()
        self.hint.setPlaceholderText("예: 조선 후기 묘비, 한문 세로쓰기")
        f.addRow("보낼 이미지", self.ai_images)
        f.addRow("참고 정보", self.hint)
        cost = QLabel("API 사용 요금이 부과됩니다 (보통 1회 수십 원~수백 원).")
        cost.setWordWrap(True)
        f.addRow(cost)
        root.addWidget(self.g_ai)

        self.run_btn = QPushButton("판독 실행")
        self.run_btn.setMinimumHeight(32)
        root.addWidget(self.run_btn)

        root.addWidget(QLabel("판독 결과 (직접 고칠 수 있습니다)"))
        self.result = QPlainTextEdit()
        self.result.setPlaceholderText("판독 결과가 여기에 표시됩니다.")
        root.addWidget(self.result, 2)
        row = QHBoxLayout()
        self.boxes_btn = QPushButton("읽은 위치 보기")
        self.boxes_btn.setEnabled(False)
        self.boxes_btn.clicked.connect(self.showBoxes.emit)
        row.addWidget(self.boxes_btn)
        row.addStretch(1)
        root.addLayout(row)

        root.addWidget(QLabel("판독 기록 (프로젝트에 저장됨)"))
        self.history = QListWidget()
        self.history.setMaximumHeight(120)
        root.addWidget(self.history, 1)

        self.r_offline.toggled.connect(self._update)
        self.lang.currentIndexChanged.connect(self._update)
        self.run_btn.clicked.connect(self._run)
        self.history.currentItemChanged.connect(self._on_history)
        self.result.textChanged.connect(self.textEdited.emit)
        self._results: list[dict] = []
        self._update()

    def _update(self):
        off = self.r_offline.isChecked()
        self.g_off.setVisible(off)
        self.g_ai.setVisible(not off)
        ready = ocr.korean_model_ready()
        is_kor = self.lang.currentData() == "korean"
        self.kor_status.setText("한국어 모델: 준비됨" if ready else "한국어 모델: 없음")
        self.kor_status.setVisible(is_kor)
        self.kor_btn.setVisible(is_kor and not ready)

    def refresh(self):
        self._update()

    def _run(self):
        if self.r_offline.isChecked():
            self.runOffline.emit({
                "lang": self.lang.currentData(),
                "direction": self.direction.currentData(),
                "auto_rotate": self.auto_rot.isChecked(),
                "source": self.source.currentData(),
            })
        else:
            self.runAi.emit({"images": self.ai_images.currentData(), "hint": self.hint.text()})

    def set_busy(self, busy: bool):
        self.run_btn.setEnabled(not busy)
        self.run_btn.setText("판독 중…" if busy else "판독 실행")

    # -- 결과/기록 ----------------------------------------------------------
    def set_results(self, results: list[dict]):
        self._results = list(results)
        self.history.blockSignals(True)
        self.history.clear()
        for r in self._results:
            self.history.addItem(QListWidgetItem(f"{r.get('created', '')[:16].replace('T', ' ')}  {r.get('engine', '')}"))
        self.history.blockSignals(False)
        if self._results:
            self.history.setCurrentRow(len(self._results) - 1)
        else:
            self.result.blockSignals(True)
            self.result.clear()
            self.result.blockSignals(False)

    def add_result(self, r: dict):
        self._results.append(r)
        self.set_results(self._results)

    def results(self) -> list[dict]:
        row = self.history.currentRow()
        if 0 <= row < len(self._results):
            self._results[row]["text"] = self.result.toPlainText()
        return self._results

    def current_result(self) -> dict | None:
        row = self.history.currentRow()
        return self._results[row] if 0 <= row < len(self._results) else None

    def _on_history(self, cur, prev):
        prev_row = self.history.row(prev) if prev is not None else -1
        if 0 <= prev_row < len(self._results):
            self._results[prev_row]["text"] = self.result.toPlainText()
        r = self.current_result()
        self.result.blockSignals(True)
        self.result.setPlainText(r.get("text", "") if r else "")
        self.result.blockSignals(False)
        self.boxes_btn.setEnabled(bool(r and r.get("lines")))
