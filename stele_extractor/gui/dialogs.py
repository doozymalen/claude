"""대화상자 모음."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..core.model3d import PrintOptions
from ..core.ocr import DEFAULT_AI_MODEL
from ..core.pipeline import IMAGE_NAMES
from ..core.project import ProjectNotes
from ..core.synthetic import SyntheticOptions


def _spin(lo, hi, val, step=0.1, decimals=2, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


def _buttons(dialog, ok_text="확인"):
    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    bb.button(QDialogButtonBox.Ok).setText(ok_text)
    bb.button(QDialogButtonBox.Cancel).setText("취소")
    bb.accepted.connect(dialog.accept)
    bb.rejected.connect(dialog.reject)
    return bb


class FolderPicker(QWidget):
    def __init__(self, path: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(path)
        btn = QPushButton("찾아보기…")
        btn.clicked.connect(self._browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택", self.edit.text())
        if d:
            self.edit.setText(d)

    def path(self) -> str:
        return self.edit.text().strip()


# ---------------------------------------------------------------------------


class SyntheticDialog(QDialog):
    """가상 비석 스캔 만들기."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("가상 비석 데이터 만들기")
        f = QFormLayout()
        info = QLabel("실제 스캔 없이 프로그램을 시험할 수 있도록 글자가 새겨진 돌 표면을 만듭니다.\n"
                      "만든 파일은 Revo Scan에서 내보낸 PLY와 같은 형식으로 저장됩니다.")
        info.setWordWrap(True)
        self.text = QPlainTextEdit("大韓\n民國")
        self.text.setFixedHeight(70)
        self.letter = QComboBox()
        self.letter.addItem("음각 (파인 글자)", "engraved")
        self.letter.addItem("양각 (솟은 글자)", "raised")
        self.surface = QComboBox()
        self.surface.addItem("평평한 면", "flat")
        self.surface.addItem("둥근 면 (원통형)", "curved")
        self.width = _spin(20, 500, 80, 5, 0, " mm")
        self.height = _spin(20, 500, 80, 5, 0, " mm")
        self.depth = _spin(0.1, 10, 1.2, 0.1, 1, " mm")
        self.wear = _spin(0, 1, 0.3, 0.1, 1)
        self.wear.setToolTip("0 = 새 비석, 1 = 많이 닳은 비석")
        self.painted = QCheckBox("글자에 흰색을 칠함 (색상 보조 시험용)")
        f.addRow(info)
        f.addRow("새길 글자 (줄바꿈 가능)", self.text)
        f.addRow("글자 형태", self.letter)
        f.addRow("표면", self.surface)
        f.addRow("가로", self.width)
        f.addRow("세로", self.height)
        f.addRow("글자 깊이", self.depth)
        f.addRow("마모 정도", self.wear)
        f.addRow(self.painted)
        lay = QVBoxLayout(self)
        lay.addLayout(f)
        lay.addWidget(_buttons(self, "만들기"))

    def options(self) -> SyntheticOptions:
        w, h = self.width.value(), self.height.value()
        # 큰 면은 정점 간격을 넓혀 파일이 너무 커지지 않게 한다
        spacing = max(0.2, round(((w * h) / 400_000) ** 0.5, 2))
        return SyntheticOptions(
            text=self.text.toPlainText().strip() or "大韓",
            letter_type=self.letter.currentData(),
            surface=self.surface.currentData(),
            width_mm=w,
            height_mm=h,
            spacing_mm=spacing,
            depth_mm=self.depth.value(),
            weathering=self.wear.value(),
            painted=self.painted.isChecked(),
        )


class ExportImagesDialog(QDialog):
    def __init__(self, folder: str, base_name: str, available: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("이미지 내보내기")
        lay = QVBoxLayout(self)
        g = QGroupBox("저장할 이미지")
        gl = QVBoxLayout(g)
        self.checks = {}
        for key in available:
            cb = QCheckBox(IMAGE_NAMES.get(key, key))
            cb.setChecked(key in ("rubbing", "hillshade", "enhanced", "mask"))
            self.checks[key] = cb
            gl.addWidget(cb)
        lay.addWidget(g)
        f = QFormLayout()
        self.fmt = QComboBox()
        self.fmt.addItem("PNG", "png")
        self.fmt.addItem("TIFF (무손실, 인쇄용)", "tif")
        self.scale_bar = QCheckBox("아래에 mm 눈금자 붙이기")
        self.scale_bar.setChecked(True)
        self.depth = QCheckBox("깊이 값(mm) TIFF도 저장 (분석용)")
        self.depth.setChecked(True)
        self.folder = FolderPicker(folder)
        self.name = QLineEdit(base_name)
        f.addRow("파일 형식", self.fmt)
        f.addRow(self.scale_bar)
        f.addRow(self.depth)
        f.addRow("저장 폴더", self.folder)
        f.addRow("파일 이름 앞부분", self.name)
        lay.addLayout(f)
        note = QLabel("이미지에 DPI 정보가 들어가 100% 크기로 인쇄하면 실제 크기와 같습니다.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addWidget(_buttons(self, "저장"))

    def selected(self) -> list[str]:
        return [k for k, cb in self.checks.items() if cb.isChecked()]


class VectorExportDialog(QDialog):
    def __init__(self, folder: str, base_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("벡터 외곽선 내보내기")
        f = QFormLayout(self)
        self.svg = QCheckBox("SVG (일러스트레이터, 잉크스케이프)")
        self.svg.setChecked(True)
        self.dxf = QCheckBox("DXF (오토캐드 등 CAD)")
        self.dxf.setChecked(True)
        self.smooth = _spin(0, 2, 0.1, 0.05, 2, " mm")
        self.smooth.setToolTip("외곽선의 계단 모양을 부드럽게 합니다.")
        self.simplify = _spin(0, 1, 0.05, 0.01, 2, " mm")
        self.simplify.setToolTip("점 개수를 줄입니다. 값이 크면 모양이 단순해집니다.")
        self.fill = QCheckBox("SVG 글자를 검게 채우기 (끄면 선만)")
        self.fill.setChecked(True)
        self.folder = FolderPicker(folder)
        self.name = QLineEdit(base_name)
        f.addRow(self.svg)
        f.addRow(self.dxf)
        f.addRow("부드럽게", self.smooth)
        f.addRow("단순화", self.simplify)
        f.addRow(self.fill)
        f.addRow("저장 폴더", self.folder)
        f.addRow("파일 이름 앞부분", self.name)
        note = QLabel("단위는 mm이며 실제 크기(1:1)입니다.")
        f.addRow(note)
        f.addRow(_buttons(self, "저장"))


class Model3DDialog(QDialog):
    """글자 3D 모델 내보내기."""

    def __init__(self, folder: str, base_name: str, has_faces: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("글자 3D 모델 내보내기")
        lay = QVBoxLayout(self)

        g1 = QGroupBox("기록·열람용 (실제 표면 그대로)")
        l1 = QFormLayout(g1)
        self.r_cut = QRadioButton("원본 스캔에서 글자 부분만 잘라내기")
        self.r_cut.setEnabled(has_faces)
        if not has_faces:
            self.r_cut.setToolTip("점군 스캔은 잘라낼 수 없습니다. 메쉬로 내보낸 파일을 쓰세요.")
        self.margin = _spin(0, 10, 0.5, 0.1, 1, " mm")
        self.margin.setToolTip("글자 둘레를 얼마나 더 남길지")
        self.cut_fmt = QComboBox()
        self.cut_fmt.addItem("PLY (색상 포함)", "ply")
        self.cut_fmt.addItem("OBJ", "obj")
        l1.addRow(self.r_cut)
        l1.addRow("글자 둘레 여백", self.margin)
        l1.addRow("파일 형식", self.cut_fmt)
        lay.addWidget(g1)

        g2 = QGroupBox("3D 프린팅용 (두께가 있는 닫힌 STL)")
        l2 = QFormLayout(g2)
        self.r_plate = QRadioButton("글자판: 바닥판 위에 글자")
        self.r_letters = QRadioButton("글자만: 입체 글자")
        self.r_relief = QRadioButton("표면 복제: 비석 표면 굴곡 그대로")
        self.engraved = QCheckBox("글자판을 음각으로 (글자를 파낸 판)")
        self.base = _spin(0.4, 50, 3.0, 0.5, 1, " mm")
        self.letter_h = _spin(0.2, 50, 2.0, 0.5, 1, " mm")
        self.relief_scale = _spin(0.5, 20, 1.0, 0.5, 1, " 배")
        self.border = _spin(0, 50, 3.0, 1, 1, " mm")
        self.mirror = QCheckBox("좌우 반전 (도장처럼 찍어낼 때)")
        for w in (self.r_plate, self.r_letters, self.r_relief):
            l2.addRow(w)
        l2.addRow(self.engraved)
        l2.addRow("바닥판 두께", self.base)
        l2.addRow("글자 높이/깊이", self.letter_h)
        l2.addRow("표면 높이 배율", self.relief_scale)
        l2.addRow("판 둘레 여백", self.border)
        l2.addRow(self.mirror)
        lay.addWidget(g2)

        f = QFormLayout()
        self.folder = FolderPicker(folder)
        self.name = QLineEdit(base_name)
        f.addRow("저장 폴더", self.folder)
        f.addRow("파일 이름 앞부분", self.name)
        lay.addLayout(f)
        lay.addWidget(_buttons(self, "만들기"))

        # 서로 다른 상자에 있는 선택지를 하나의 묶음으로
        self.group = QButtonGroup(self)
        for r in (self.r_cut, self.r_plate, self.r_letters, self.r_relief):
            self.group.addButton(r)
        (self.r_cut if has_faces else self.r_plate).setChecked(True)
        for r in (self.r_cut, self.r_plate, self.r_letters, self.r_relief):
            r.toggled.connect(self._update)
        self._update()

    def _update(self):
        plate = self.r_plate.isChecked()
        relief = self.r_relief.isChecked()
        cut = self.r_cut.isChecked()
        self.margin.setEnabled(cut)
        self.cut_fmt.setEnabled(cut)
        self.engraved.setEnabled(plate)
        self.base.setEnabled(plate or relief)
        self.letter_h.setEnabled(plate or self.r_letters.isChecked())
        self.relief_scale.setEnabled(relief)
        self.border.setEnabled(plate or relief)
        self.mirror.setEnabled(not cut)

    def mode(self) -> str:
        if self.r_cut.isChecked():
            return "cut"
        if self.r_letters.isChecked():
            return "letters"
        if self.r_relief.isChecked():
            return "relief"
        return "plate"

    def print_options(self) -> PrintOptions:
        return PrintOptions(
            kind=self.mode(),
            base_mm=self.base.value(),
            letter_mm=self.letter_h.value(),
            engraved_plate=self.engraved.isChecked(),
            relief_scale=self.relief_scale.value(),
            border_mm=self.border.value(),
            mirror=self.mirror.isChecked(),
        )


class NotesDialog(QDialog):
    """비석 정보(메모)."""

    def __init__(self, notes: ProjectNotes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("비석 정보")
        f = QFormLayout(self)
        self.name = QLineEdit(notes.name)
        self.location = QLineEdit(notes.location)
        self.date = QLineEdit(notes.date or date.today().isoformat())
        self.surveyor = QLineEdit(notes.surveyor)
        self.memo = QPlainTextEdit(notes.memo)
        f.addRow("비석 이름", self.name)
        f.addRow("위치", self.location)
        f.addRow("조사 날짜", self.date)
        f.addRow("조사자", self.surveyor)
        f.addRow("메모", self.memo)
        f.addRow(_buttons(self))

    def notes(self) -> ProjectNotes:
        return ProjectNotes(
            name=self.name.text().strip(),
            location=self.location.text().strip(),
            date=self.date.text().strip(),
            surveyor=self.surveyor.text().strip(),
            memo=self.memo.toPlainText().strip(),
        )


class NewProjectDialog(QDialog):
    def __init__(self, default_name: str, default_parent: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("새 프로젝트로 저장")
        f = QFormLayout(self)
        self.name = QLineEdit(default_name)
        self.parent_dir = FolderPicker(default_parent)
        self.copy_scan = QCheckBox("스캔 파일을 프로젝트 폴더에 복사 (폴더만 옮겨도 열 수 있음)")
        self.copy_scan.setChecked(True)
        f.addRow("프로젝트 이름", self.name)
        f.addRow("만들 위치", self.parent_dir)
        f.addRow(self.copy_scan)
        f.addRow(QLabel("선택한 위치 안에 프로젝트 이름으로 폴더가 만들어집니다."))
        f.addRow(_buttons(self, "저장"))

    def folder(self) -> Path:
        name = self.name.text().strip() or "비석"
        for ch in '\\/:*?"<>|':
            name = name.replace(ch, "_")
        return Path(self.parent_dir.path()) / name


class SettingsDialog(QDialog):
    def __init__(self, api_key: str, model: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        f = QFormLayout(self)
        g = QLabel(
            "AI 판독(Claude)을 쓰려면 Anthropic API 키가 필요합니다.\n"
            "console.anthropic.com 에서 키를 만들 수 있으며, 사용한 만큼 요금이 부과됩니다.\n"
            "키는 이 컴퓨터의 사용자 설정에만 저장됩니다."
        )
        g.setWordWrap(True)
        self.key = QLineEdit(api_key)
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("sk-ant-…")
        show = QCheckBox("키 보이기")
        show.toggled.connect(lambda on: self.key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        self.model = QLineEdit(model or DEFAULT_AI_MODEL)
        f.addRow(g)
        f.addRow("API 키", self.key)
        f.addRow("", show)
        f.addRow("AI 모델", self.model)
        f.addRow(_buttons(self, "저장"))


class TextDialog(QDialog):
    """긴 안내문/오류 내용 보기."""

    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 480)
        lay = QVBoxLayout(self)
        t = QPlainTextEdit(text)
        t.setReadOnly(True)
        lay.addWidget(t)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.button(QDialogButtonBox.Close).setText("닫기")
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
