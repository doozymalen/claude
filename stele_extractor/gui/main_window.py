"""메인 창."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..core import export as exporter
from ..core import depth_edit as de
from ..core import model3d, ocr, vector
from ..core.pipeline import IMAGE_NAMES, ReliefResult, build_geometry, params_to_dict, process_relief
from ..core.project import Project, crop_to_pixels, load_project, pixels_to_crop, save_project
from ..core.scan import SUPPORTED_EXTENSIONS, UNIT_SCALES, ScanData, guess_unit, load_scan
from ..core.synthetic import make_synthetic_scan, save_scan_ply
from .depth_panel import DepthPanel, describe
from .dialogs import (
    ExportImagesDialog,
    Model3DDialog,
    NewProjectDialog,
    NotesDialog,
    SettingsDialog,
    SyntheticDialog,
    TextDialog,
    VectorExportDialog,
)
from .image_view import ImageView
from .ocr_panel import OcrPanel
from .params_panel import ParamsPanel
from .viewer3d import Viewer3D
from .worker import run_task

APP_TITLE = "비석 글자 추출기"
OCR_VIEW_KEY = "__ocr__"

HELP_TEXT = """\
<h2>비석 글자 추출기 사용법</h2>
<h3>1. 스캔 불러오기</h3>
<p>Revo Scan에서 스캔을 <b>PLY</b>(권장), OBJ, STL로 내보낸 뒤 [파일 → 스캔 열기]로 엽니다.
파일을 창으로 끌어다 놓아도 됩니다. 실제 스캔이 없으면 [파일 → 가상 비석 만들기]로 시험해 보세요.</p>
<h3>2. 결과 보기</h3>
<p>[2D 결과] 탭 위쪽 목록에서 <b>탁본, 음영 기복, 글자 강조, 글자 추출(흑백), 깊이 색상</b>을 바꿔 볼 수 있습니다.
마우스 휠로 확대/축소, 끌어서 이동합니다.</p>
<h3>3. 보정</h3>
<ul>
<li>글자가 뒤집혀 보이면 오른쪽 [회전] 버튼, 음각/양각이 반대면 [앞뒤 뒤집기]</li>
<li>둥근 비석은 [표면 형태 → 원통 펼치기]</li>
<li>얕은 글자가 안 잡히면 [감도]를 높이고, 잡티가 많으면 [최소 크기]·[잡음 제거]를 높이세요.</li>
<li>[영역 선택]을 누르고 2D 화면에서 끌어서 글자 부분만 잘라 처리할 수 있습니다.</li>
<li>마모된 글자는 [음영 기복]에서 [빛 방향]을 돌려 보면 잘 보입니다.</li>
</ul>
<h3>4. 내보내기</h3>
<ul>
<li>이미지: PNG/TIFF, 실제 크기 DPI와 mm 눈금자 포함</li>
<li>벡터 외곽선: SVG/DXF, mm 단위 1:1</li>
<li>3D 모델: 글자 부분만 잘라낸 원본(PLY/OBJ), 3D 프린팅용 STL(글자판/글자만/표면 복제)</li>
</ul>
<h3>5. 글자 판독</h3>
<p>오른쪽 [글자 판독] 패널에서 오프라인 OCR(한자·영문 / 한글) 또는 AI 판독(Claude)을 실행합니다.
AI 판독은 [설정]에서 API 키를 넣어야 합니다. 결과는 직접 고칠 수 있고 프로젝트에 저장됩니다.</p>
<h3>6. 글자 깊이 편집</h3>
<p>오른쪽 [깊이 편집] 패널에서 [깊이 편집 켜기]를 체크하고 [깊이 배율]을 움직이면 모든 글자의 깊이가 바뀝니다
(×2 = 두 배 깊게, ×0 = 평평하게, 음수 = 음각↔양각 뒤집기).
[2D 화면에서 글자 고르기]를 누르고 글자를 클릭(Ctrl+클릭으로 여러 개, 끌기로 범위)하면 고른 글자만 따로 조절합니다.
바뀐 깊이는 3D 화면, 이미지, 3D 내보내기에 모두 적용되고 [깊이 바꾼 원본 3D 저장]으로 스캔 전체를 저장할 수 있습니다.</p>
<h3>7. 프로젝트</h3>
<p>[파일 → 프로젝트 저장]을 하면 비석별 폴더에 설정·자른 영역·비석 정보·판독 결과가 저장되어
나중에 그대로 다시 열 수 있습니다.</p>
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("SteleExtractor", "SteleExtractor")
        self.resize(1400, 880)
        self.setAcceptDrops(True)

        # 상태
        self.raw_scan: ScanData | None = None
        self.scan: ScanData | None = None
        self.proj = None
        self.hm_full = None
        self.hm = None
        self.result: ReliefResult | None = None  # 화면·내보내기에 쓰는 결과 (깊이 편집 적용)
        self.base_result: ReliefResult | None = None  # 깊이 편집 전 결과
        self.labels = None  # 글자 번호 지도
        self.n_letters = 0
        self._labels_merge = None
        self.depth_edit = de.DepthEdit()
        self.selected: list[int] = []  # 고른 글자 번호
        self._selected_uv: list = []  # 다시 처리해도 고른 글자를 기억하기 위한 위치(mm)
        self._edit_running = self._edit_pending = False
        self._shown_edited_3d = False
        self.crop_mm: list | None = None
        self.project = Project()
        self.dirty = False
        self._geo_key = None
        self._busy = 0
        self._geo_running = self._geo_pending = False
        self._rel_running = self._rel_pending = False
        self._ocr_view = None  # (이미지, 결과 dict)

        self._build_ui()
        self._build_actions()
        self._update_title()
        self._update_enabled()

    # ------------------------------------------------------------------
    # 화면 구성
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.tabs = QTabWidget()
        self.viewer3d = Viewer3D()
        self.tabs.addTab(self.viewer3d, "3D 보기")

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("보기:"))
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(160)
        bar.addWidget(self.image_combo)
        self.select_btn = QPushButton("영역 선택")
        self.select_btn.setCheckable(True)
        self.select_btn.setToolTip("2D 화면에서 끌어서 처리할 영역을 고릅니다.")
        self.uncrop_btn = QPushButton("전체 영역")
        fit_btn = QPushButton("화면 맞춤")
        bar.addWidget(self.select_btn)
        bar.addWidget(self.uncrop_btn)
        bar.addWidget(fit_btn)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #444;")
        lay.addWidget(self.stats_label)
        self.image_view = ImageView()
        lay.addWidget(self.image_view, 1)
        self.tabs.addTab(page, "2D 결과")

        self.welcome = QTextBrowser()
        self.welcome.setOpenExternalLinks(True)
        self.welcome.setHtml(
            f"<h1>{APP_TITLE}</h1><p>POP 4 등 3D 스캐너로 스캔한 비석에서 글자를 추출합니다.</p>"
            "<p><b>시작하기:</b> [파일 → 스캔 열기]로 Revo Scan에서 내보낸 PLY/OBJ/STL 파일을 열거나, "
            "[파일 → 가상 비석 만들기]로 시험용 데이터를 만들어 보세요.</p>" + HELP_TEXT
        )
        self.tabs.addTab(self.welcome, "시작")
        self.tabs.setCurrentWidget(self.welcome)
        self.setCentralWidget(self.tabs)

        # 오른쪽: 설정 / 글자 판독
        self.params = ParamsPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.params)
        dock = QDockWidget("설정", self)
        dock.setObjectName("dock_params")
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.ocr_panel = OcrPanel()
        ocr_scroll = QScrollArea()
        ocr_scroll.setWidgetResizable(True)
        ocr_scroll.setWidget(self.ocr_panel)
        dock2 = QDockWidget("글자 판독", self)
        dock2.setObjectName("dock_ocr")
        dock2.setWidget(ocr_scroll)
        dock2.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock2)
        self.tabifyDockWidget(dock, dock2)

        self.depth_panel = DepthPanel()
        depth_scroll = QScrollArea()
        depth_scroll.setWidgetResizable(True)
        depth_scroll.setWidget(self.depth_panel)
        dock3 = QDockWidget("깊이 편집", self)
        dock3.setObjectName("dock_depth")
        dock3.setWidget(depth_scroll)
        dock3.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock3)
        self.tabifyDockWidget(dock2, dock3)
        dock.raise_()

        # 상태 표시줄
        self.pos_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(160)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.pos_label)
        self.statusBar().addPermanentWidget(self.progress)

        # 신호
        self.params.geometryChanged.connect(self._schedule_geometry)
        self.params.reliefChanged.connect(self._schedule_relief)
        self.image_combo.currentIndexChanged.connect(self._show_current_image)
        self.select_btn.toggled.connect(self.image_view.set_select_mode)
        self.uncrop_btn.clicked.connect(self.clear_crop)
        fit_btn.clicked.connect(self.image_view.fit)
        self.image_view.regionSelected.connect(self._on_region)
        self.image_view.mouseMoved.connect(self._on_mouse)
        self.ocr_panel.runOffline.connect(self.run_offline_ocr)
        self.ocr_panel.runAi.connect(self.run_ai_ocr)
        self.ocr_panel.downloadKorean.connect(self.download_korean)
        self.ocr_panel.textEdited.connect(self._mark_dirty)
        self.ocr_panel.showBoxes.connect(self.show_ocr_boxes)
        self.depth_panel.editChanged.connect(self._schedule_edit)
        self.depth_panel.pickModeToggled.connect(self._on_pick_mode)
        self.depth_panel.selectedScaleChanged.connect(self._on_selected_scale)
        self.depth_panel.resetSelected.connect(lambda: self._on_selected_scale(None))
        self.depth_panel.clearSelection.connect(lambda: self._set_selection([]))
        self.depth_panel.exportEdited.connect(self.export_edited_scan)
        self.depth_panel.view3dChanged.connect(self._schedule_edit)
        self.image_view.clicked.connect(self._on_pick_click)
        self.image_view.rectPicked.connect(self._on_pick_rect)

        self._geo_timer = QTimer(self, singleShot=True, interval=400, timeout=self.request_geometry)
        self._rel_timer = QTimer(self, singleShot=True, interval=150, timeout=self.request_relief)
        self._edit_timer = QTimer(self, singleShot=True, interval=120, timeout=self.request_edit)

    def _build_actions(self):
        mb = self.menuBar()
        m_file = mb.addMenu("파일(&F)")
        self.a_open = QAction("스캔 열기…", self, shortcut=QKeySequence.Open, triggered=self.open_scan_dialog)
        self.a_synth = QAction("가상 비석 만들기…", self, triggered=self.make_synthetic)
        self.a_open_proj = QAction("프로젝트 열기…", self, shortcut="Ctrl+Shift+O", triggered=self.open_project_dialog)
        self.a_save = QAction("프로젝트 저장", self, shortcut=QKeySequence.Save, triggered=self.save_project)
        self.a_save_as = QAction("다른 이름으로 프로젝트 저장…", self, shortcut="Ctrl+Shift+S",
                                 triggered=lambda: self.save_project(save_as=True))
        self.a_notes = QAction("비석 정보…", self, triggered=self.edit_notes)
        self.a_exp_img = QAction("이미지 내보내기…", self, shortcut="Ctrl+E", triggered=self.export_images)
        self.a_exp_vec = QAction("벡터 외곽선 내보내기 (SVG/DXF)…", self, triggered=self.export_vector)
        self.a_exp_3d = QAction("글자 3D 모델 내보내기…", self, triggered=self.export_3d)
        self.a_settings = QAction("설정 (AI API 키)…", self, triggered=self.edit_settings)
        a_quit = QAction("끝내기", self, shortcut=QKeySequence.Quit, triggered=self.close)
        for a in (self.a_open, self.a_synth, None, self.a_open_proj, self.a_save, self.a_save_as, self.a_notes,
                  None, self.a_exp_img, self.a_exp_vec, self.a_exp_3d, None, self.a_settings, None, a_quit):
            if a is None:
                m_file.addSeparator()
            else:
                m_file.addAction(a)
        m_help = mb.addMenu("도움말(&H)")
        m_help.addAction(QAction("사용법", self, shortcut=QKeySequence.HelpContents,
                                 triggered=lambda: self.tabs.setCurrentWidget(self.welcome)))
        m_help.addAction(QAction("프로그램 정보", self, triggered=self.about))

        tb = QToolBar("도구")
        tb.setObjectName("toolbar")
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        for a in (self.a_open, self.a_synth, None, self.a_open_proj, self.a_save, self.a_notes, None,
                  self.a_exp_img, self.a_exp_vec, self.a_exp_3d):
            if a is None:
                tb.addSeparator()
            else:
                tb.addAction(a)
        self.addToolBar(tb)

    # ------------------------------------------------------------------
    # 공통
    # ------------------------------------------------------------------
    def _update_title(self):
        name = self.project.notes.name or (Path(self.raw_scan.source_path).stem
                                           if self.raw_scan and self.raw_scan.source_path else "")
        t = APP_TITLE
        if name:
            t = f"{name} - {t}"
        if self.dirty:
            t = "* " + t
        self.setWindowTitle(t)

    def _mark_dirty(self, *_):
        if self.raw_scan is not None and not self.dirty:
            self.dirty = True
            self._update_title()

    def _update_enabled(self):
        has = self.result is not None
        for a in (self.a_exp_img, self.a_exp_vec, self.a_exp_3d, self.a_save, self.a_save_as, self.a_notes):
            a.setEnabled(has if a in (self.a_exp_img, self.a_exp_vec, self.a_exp_3d) else self.raw_scan is not None)
        self.ocr_panel.run_btn.setEnabled(has)
        self.select_btn.setEnabled(has)
        self.uncrop_btn.setEnabled(self.crop_mm is not None)

    def _busy_on(self, text: str):
        self._busy += 1
        self.progress.show()
        self.statusBar().showMessage(text)

    def _busy_off(self, text: str = ""):
        self._busy = max(0, self._busy - 1)
        if self._busy == 0:
            self.progress.hide()
        if text:
            self.statusBar().showMessage(text, 8000)

    def _error(self, title: str, message: str, details: str = ""):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(title)
        box.setText(message)
        if details:
            box.setDetailedText(details)
        box.exec()

    def _default_out_dir(self) -> str:
        if self.project.results_dir is not None:
            return str(self.project.results_dir)
        if self.raw_scan is not None and self.raw_scan.source_path:
            p = Path(self.raw_scan.source_path)
            return str(p.parent / f"{p.stem}_결과")
        return str(Path.home())

    def _base_name(self) -> str:
        if self.project.notes.name:
            return self.project.notes.name
        if self.raw_scan is not None and self.raw_scan.source_path:
            return Path(self.raw_scan.source_path).stem
        return "비석"

    # ------------------------------------------------------------------
    # 스캔 불러오기
    # ------------------------------------------------------------------
    def open_scan_dialog(self):
        if not self._confirm_discard():
            return
        start = self.settings.value("last_scan_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self, "스캔 파일 열기", start, "3D 스캔 (*.ply *.obj *.stl);;모든 파일 (*)"
        )
        if path:
            self.settings.setValue("last_scan_dir", str(Path(path).parent))
            self.project = Project()
            self.crop_mm = None
            self.load_scan_file(path)

    def load_scan_file(self, path: str, unit: str | None = None, after=None):
        self._busy_on(f"스캔을 불러오는 중… {Path(path).name}")

        def done(scan: ScanData):
            self._busy_off(f"불러옴: {Path(path).name} (정점 {scan.n_points:,}개)")
            self.raw_scan = scan
            self.params.set_unit(unit or guess_unit(scan))
            self.params.set_color_available(scan.colors is not None)
            self.viewer3d.show_scan(scan.scaled(UNIT_SCALES[self.params.unit_key()]))
            self.tabs.setCurrentIndex(1)
            self.result = None
            self.base_result = None
            self.labels = None
            self.selected, self._selected_uv = [], []
            self._shown_edited_3d = False
            self.depth_edit = de.DepthEdit.from_dict(self.project.depth_edit)
            self.depth_panel.set_edit(self.depth_edit)
            self._ocr_view = None
            self.ocr_panel.set_results(self.project.ocr_results)
            self.dirty = False
            self._update_title()
            if after:
                after()
            self._geo_key = None
            self.request_geometry()

        def failed(msg, details):
            self._busy_off("불러오기 실패")
            self._error("스캔 열기 실패", f"파일을 열지 못했습니다.\n{msg}", details)

        run_task(load_scan, path, "mm", on_done=done, on_failed=failed)

    def make_synthetic(self):
        if not self._confirm_discard():
            return
        dlg = SyntheticDialog(self)
        if dlg.exec() != SyntheticDialog.Accepted:
            return
        opts = dlg.options()
        opts.seed = zlib.crc32(opts.text.encode("utf-8")) % 10_000
        default = str(Path(self.settings.value("last_scan_dir", str(Path.home()))) / "가상비석.ply")
        path, _ = QFileDialog.getSaveFileName(self, "가상 스캔 저장", default, "PLY (*.ply)")
        if not path:
            return
        if not path.lower().endswith(".ply"):
            path += ".ply"
        self._busy_on("가상 비석을 만드는 중…")

        def work():
            scan = make_synthetic_scan(opts)
            save_scan_ply(scan, path)
            return path

        def done(p):
            self._busy_off("가상 비석을 만들었습니다.")
            self.project = Project()
            self.crop_mm = None
            self.load_scan_file(p, unit="mm")

        def failed(msg, details):
            self._busy_off()
            self._error("가상 데이터 만들기 실패", msg, details)

        run_task(work, on_done=done, on_failed=failed)

    # ------------------------------------------------------------------
    # 처리 (3D→2D, 글자 찾기)
    # ------------------------------------------------------------------
    def _schedule_geometry(self):
        self._mark_dirty()
        self._geo_timer.start()

    def _schedule_relief(self):
        self._mark_dirty()
        self._rel_timer.start()

    def request_geometry(self):
        if self.raw_scan is None:
            return
        if self._geo_running:
            self._geo_pending = True
            return
        self._geo_running = True
        unit = self.params.unit_key()
        gp = self.params.geometry_params()
        key = (unit, gp.surface_mode, gp.flip, round(gp.rotation_deg, 3), gp.auto_align)
        if self._geo_key is not None and key != self._geo_key and self.crop_mm is not None:
            self.crop_mm = None
            self.statusBar().showMessage("방향/형태가 바뀌어 선택 영역을 해제했습니다.", 6000)
        self._geo_key = key
        raw = self.raw_scan
        self._busy_on("3D 스캔을 2D로 펼치는 중…")

        def work():
            scan = raw.scaled(UNIT_SCALES[unit])
            proj, hm = build_geometry(scan, gp)
            return scan, proj, hm

        def done(res):
            self._geo_running = False
            self._busy_off()
            if self._geo_pending:
                self._geo_pending = False
                self.request_geometry()
                return
            self.scan, self.proj, self.hm_full = res
            self.params.set_pixel_size(self.hm_full.pixel_mm)
            info = self.hm_full.info
            if info.get("mode") == "cylinder":
                self.statusBar().showMessage(f"원통으로 펼쳤습니다 (반지름 약 {info.get('radius_mm', 0):.0f} mm)", 8000)
            self._apply_crop()
            self.request_relief()

        def failed(msg, details):
            self._geo_running = False
            self._busy_off("처리 실패")
            self._geo_pending = False
            self._error("스캔 펼치기 실패", msg, details)

        run_task(work, on_done=done, on_failed=failed)

    def _apply_crop(self):
        hm = self.hm_full
        self.hm = hm
        if hm is None or self.crop_mm is None:
            self._update_enabled()
            return
        r0, r1, c0, c1 = crop_to_pixels(self.crop_mm, hm.u0, hm.v0, hm.pixel_mm)
        try:
            self.hm = hm.crop(r0, r1, c0, c1)
        except ValueError:
            self.crop_mm = None
            self.hm = hm
        self._update_enabled()

    def request_relief(self):
        if self.hm is None:
            return
        if self._rel_running:
            self._rel_pending = True
            return
        self._rel_running = True
        hm, rp = self.hm, self.params.relief_params()
        merge = self.depth_panel.edit().merge_mm
        self._busy_on("글자를 찾는 중…")

        def work():
            result = process_relief(hm, rp)
            labels, n = de.label_letters(result.mask, hm.pixel_mm, merge)
            return result, labels, n

        def done(res):
            self._rel_running = False
            self._busy_off("처리 완료")
            if self._rel_pending:
                self._rel_pending = False
                self.request_relief()
                return
            result, self.labels, self.n_letters = res
            self._labels_merge = merge
            self.base_result = result
            self.params.set_background_value(result.background_mm)
            self._restore_selection()
            self.depth_panel.show_merge_value(de.auto_merge_mm(result.mask, result.heightmap.pixel_mm)
                                              if result.mask.any() else 3.0)
            self.request_edit(immediate=True)

        def failed(msg, details):
            self._rel_running = False
            self._rel_pending = False
            self._busy_off("처리 실패")
            self._error("글자 찾기 실패", msg, details)

        run_task(work, on_done=done, on_failed=failed)

    # ------------------------------------------------------------------
    # 깊이 편집
    # ------------------------------------------------------------------
    def _schedule_edit(self):
        if self.base_result is not None:
            self._mark_dirty()
        self._edit_timer.start()

    def _set_result(self, result: ReliefResult):
        self.result = result
        self._refresh_image_list()
        st = result.stats()
        self.stats_label.setText("   ".join(f"{k}: {v}" for k, v in st.items()))
        self._update_depth_info()
        self._update_enabled()

    def request_edit(self, immediate: bool = False):
        """깊이 편집을 (다시) 적용한다. 편집이 꺼져 있으면 원래 결과를 그대로 쓴다."""
        base = self.base_result
        if base is None:
            return
        edit = self.depth_panel.edit(self.depth_edit.overrides)
        self.depth_edit = edit
        show3d = self.viewer3d.available and self.scan is not None and self.depth_panel.live3d.isChecked()
        use_color = not self.depth_panel.no_color.isChecked()
        if not edit.enabled or edit.is_identity():
            self._set_result(base)
            if self._shown_edited_3d and self.scan is not None:
                self.viewer3d.show_scan(self.scan, reset_camera=False)
                self._shown_edited_3d = False
            if not edit.enabled and self.depth_panel.pick_btn.isChecked():
                self.depth_panel.pick_btn.setChecked(False)
            return
        if self._edit_running:
            self._edit_pending = True
            return
        self._edit_running = True
        labels, n = self.labels, self.n_letters
        merge_changed = self._labels_merge != edit.merge_mm
        rp = self.params.relief_params()
        scan, proj = self.scan, self.proj

        def work():
            lab, cnt = labels, n
            if merge_changed:
                lab, cnt = de.label_letters(base.mask, base.heightmap.pixel_mm, edit.merge_mm)
            edited = de.apply_edit(base, lab, cnt, edit, rp)
            moved = de.displace_scan(scan, proj, edited) if show3d else None
            return lab, cnt, edited, moved

        def done(res):
            self._edit_running = False
            lab, cnt, edited, moved = res
            if merge_changed:
                self.labels, self.n_letters, self._labels_merge = lab, cnt, edit.merge_mm
                self._restore_selection()
            if self._edit_pending:
                self._edit_pending = False
                self.request_edit()
                return
            self._set_result(edited)
            if moved is not None:
                self.viewer3d.show_scan(moved, reset_camera=False, use_color=use_color)
                self._shown_edited_3d = True

        def failed(msg, details):
            self._edit_running = False
            self._edit_pending = False
            self._error("깊이 편집 실패", msg, details)

        run_task(work, on_done=done, on_failed=failed)

    def _update_depth_info(self):
        base = self.base_result
        if base is None or self.labels is None or not self.n_letters:
            self.depth_panel.set_global_info("찾은 글자가 없습니다.")
            self.depth_panel.set_selection(0, None)
            return
        before = de.letter_depth_mm(base, self.labels)
        after = de.letter_depth_mm(self.result, self.labels) if self.result is not None else before
        self.depth_panel.set_global_info(
            f"글자 {self.n_letters}개 · 원래 {describe(before, base.polarity)} → 지금 {describe(after, base.polarity)}"
        )
        if self.selected:
            table = de.letter_scales(self.depth_edit, self.labels, self.n_letters, base.heightmap)
            d0 = de.letter_depth_mm(base, self.labels, self.selected)
            d1 = de.letter_depth_mm(self.result, self.labels, self.selected)
            self.depth_panel.set_selection(len(self.selected), float(table[self.selected[0]]),
                                           f"{describe(d0, base.polarity)} → {describe(d1, base.polarity)}")
        else:
            self.depth_panel.set_selection(0, None)

    def _on_pick_mode(self, on: bool):
        if on:
            self.select_btn.setChecked(False)
            self.tabs.setCurrentIndex(1)
        self.image_view.set_pick_mode(on)

    def _set_selection(self, ids: list[int]):
        self.selected = sorted(set(int(i) for i in ids if i))
        hm = self.base_result.heightmap if self.base_result else None
        self._selected_uv = []
        if hm is not None and self.labels is not None:
            for lab in self.selected:
                a = de.letter_anchor(self.labels, lab)
                if a is not None:
                    self._selected_uv.append(de.pixel_to_uv(hm, *a))
        self._update_depth_info()
        self._show_current_image()

    def _restore_selection(self):
        """다시 처리해서 글자 번호가 바뀌어도 고른 글자를 유지한다."""
        if self.base_result is None or self.labels is None:
            self.selected = []
            return
        hm = self.base_result.heightmap
        ids = []
        for u, v in self._selected_uv:
            r, c = de.uv_to_pixel(hm, u, v)
            lab = de.label_at(self.labels, r, c, max(1, int(1.0 / hm.pixel_mm)))
            if lab:
                ids.append(lab)
        self.selected = sorted(set(ids))

    def _on_pick_click(self, row, col, add):
        if self.labels is None:
            return
        lab = de.label_at(self.labels, row, col, max(2, int(round(0.5 / self.base_result.heightmap.pixel_mm))))
        if add:
            ids = set(self.selected)
            if lab:
                ids ^= {lab}
            self._set_selection(list(ids))
        else:
            self._set_selection([lab] if lab else [])

    def _on_pick_rect(self, r0, r1, c0, c1, add):
        if self.labels is None:
            return
        ids = de.labels_in_rect(self.labels, r0, r1, c0, c1)
        self._set_selection(list(set(self.selected) | set(ids)) if add else ids)

    def _on_selected_scale(self, scale):
        if not self.selected or self.base_result is None:
            return
        edit = self.depth_panel.edit(self.depth_edit.overrides)
        de.set_letter_scale(edit, self.labels, self.base_result.heightmap, self.selected, scale)
        self.depth_edit = edit
        self._schedule_edit()

    def _selection_overlay(self, img: np.ndarray) -> np.ndarray:
        """고른 글자를 주황색 테두리와 옅은 색으로 표시한다 (바뀐 모양이 가려지지 않게)."""
        if not self.selected or self.labels is None or self.labels.shape != img.shape[:2]:
            return img
        import cv2

        rgb = img if img.ndim == 3 else np.repeat(img[..., None], 3, axis=2)
        rgb = rgb.astype(np.float32)
        m = np.isin(self.labels, self.selected).astype(np.uint8)
        t = max(1, min(img.shape[:2]) // 300)
        k = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
        edge = (cv2.dilate(m, k) - cv2.erode(m, k)).astype(bool)
        orange = np.array([255, 140, 0], np.float32)
        fill = m.astype(bool) & ~edge
        rgb[fill] = rgb[fill] * 0.8 + orange * 0.2
        rgb[edge] = orange
        return rgb.astype(np.uint8)

    def export_edited_scan(self):
        if self.result is None or self.scan is None or self.result.delta is None:
            QMessageBox.information(self, "깊이 편집", "아직 바뀐 깊이가 없습니다. 배율을 조정해 보세요.")
            return
        default = str(Path(self._default_out_dir()) / f"{self._base_name()}_깊이수정.ply")
        path, flt = QFileDialog.getSaveFileName(self, "깊이 바꾼 원본 3D 저장", default,
                                                "PLY (색상 포함) (*.ply);;OBJ (*.obj);;STL (*.stl)")
        if not path:
            return
        if Path(path).suffix.lower() not in (".ply", ".obj", ".stl"):
            path += ".ply"
        scan, proj, result = self.scan, self.proj, self.result

        def work():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            moved = de.displace_scan(scan, proj, result)
            model3d.save_scan_mesh(moved, path)
            return [path], f"정점 {moved.n_points:,}개"

        self._busy_on("깊이 바꾼 3D를 저장하는 중…")
        run_task(work, on_done=lambda r: self._saved(r[0], str(Path(path).parent), r[1]),
                 on_failed=self._export_failed)

    def _refresh_image_list(self):
        current = self.image_combo.currentData() or "rubbing"
        if current == OCR_VIEW_KEY:
            current = "rubbing"
        keys = [k for k in IMAGE_NAMES if self.result and k in self.result.images]
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for k in keys:
            self.image_combo.addItem(IMAGE_NAMES[k], k)
        if self._ocr_view is not None:
            self.image_combo.addItem("판독 위치", OCR_VIEW_KEY)
        i = self.image_combo.findData(current)
        self.image_combo.setCurrentIndex(max(i, 0))
        self.image_combo.blockSignals(False)
        self._show_current_image()

    def _show_current_image(self, *_):
        key = self.image_combo.currentData()
        if key == OCR_VIEW_KEY and self._ocr_view is not None:
            self.select_btn.setChecked(False)
            self.select_btn.setEnabled(False)
            self.image_view.set_image(self._ocr_view[0])
            return
        self.select_btn.setEnabled(self.result is not None)
        if self.result is None or key not in self.result.images:
            self.image_view.set_image(None)
            return
        img = self.result.images[key]
        if key != "color":
            img = self._selection_overlay(img)
        self.image_view.set_image(img)

    def _on_region(self, r0, r1, c0, c1):
        if self.hm is None:
            return
        self.crop_mm = pixels_to_crop(r0, r1, c0, c1, self.hm.u0, self.hm.v0, self.hm.pixel_mm)
        self.select_btn.setChecked(False)
        self._mark_dirty()
        self._apply_crop()
        self.image_view.set_image(None)
        self.request_relief()

    def clear_crop(self):
        if self.crop_mm is None:
            return
        self.crop_mm = None
        self._mark_dirty()
        self._apply_crop()
        self.image_view.set_image(None)
        self.request_relief()

    def _on_mouse(self, row, col):
        if row < 0 or self.result is None or self.image_combo.currentData() == OCR_VIEW_KEY:
            self.pos_label.setText("")
            return
        hm = self.result.heightmap
        if row >= hm.shape[0] or col >= hm.shape[1]:
            return
        px = hm.pixel_mm
        txt = f"위치 x {col * px:.1f} mm, y {row * px:.1f} mm"
        if hm.valid[row, col]:
            txt += f"   높이 {self.result.residual[row, col]:+.2f} mm"
        self.pos_label.setText(txt)

    # ------------------------------------------------------------------
    # 내보내기
    # ------------------------------------------------------------------
    def export_images(self):
        if self.result is None:
            return
        dlg = ExportImagesDialog(self._default_out_dir(), self._base_name(), list(self.result.images), self)
        if dlg.exec() != ExportImagesDialog.Accepted:
            return
        which = dlg.selected()
        folder, name = dlg.folder.path(), dlg.name.text().strip() or "비석"
        result = self.result
        params = params_to_dict(self.params.geometry_params(), self.params.relief_params())
        self._busy_on("이미지를 저장하는 중…")
        run_task(
            exporter.export_result, result, folder, name, which, dlg.fmt.currentData(),
            dlg.scale_bar.isChecked(), dlg.depth.isChecked(), params,
            on_done=lambda files: self._saved(files, folder),
            on_failed=self._export_failed,
        )

    def export_vector(self):
        if self.result is None:
            return
        dlg = VectorExportDialog(self._default_out_dir(), self._base_name(), self)
        if dlg.exec() != VectorExportDialog.Accepted:
            return
        result = self.result
        folder = Path(dlg.folder.path())
        name = dlg.name.text().strip() or "비석"
        smooth, simplify = dlg.smooth.value(), dlg.simplify.value()
        do_svg, do_dxf, fill = dlg.svg.isChecked(), dlg.dxf.isChecked(), dlg.fill.isChecked()

        def work():
            hm = result.heightmap
            outlines = vector.mask_to_outlines(result.mask, hm.pixel_mm, simplify, smooth)
            if not outlines:
                raise ValueError("추출된 글자가 없습니다. 감도를 조정해 보세요.")
            folder.mkdir(parents=True, exist_ok=True)
            w, h = hm.size_mm
            files = []
            if do_svg:
                files.append(vector.save_svg(outlines, folder / f"{name}_외곽선.svg", w, h, fill))
            if do_dxf:
                files.append(vector.save_dxf(outlines, folder / f"{name}_외곽선.dxf", w, h))
            return files

        self._busy_on("외곽선을 만드는 중…")
        run_task(work, on_done=lambda files: self._saved(files, str(folder)), on_failed=self._export_failed)

    def export_3d(self):
        if self.result is None or self.scan is None:
            return
        dlg = Model3DDialog(self._default_out_dir(), self._base_name(), self.scan.has_faces, self)
        if dlg.exec() != Model3DDialog.Accepted:
            return
        result, scan, proj = self.result, self.scan, self.proj
        folder = Path(dlg.folder.path())
        name = dlg.name.text().strip() or "비석"
        mode = dlg.mode()
        opts = dlg.print_options()
        margin, cut_fmt = dlg.margin.value(), dlg.cut_fmt.currentData()

        def work():
            hm = result.heightmap
            folder.mkdir(parents=True, exist_ok=True)
            if mode == "cut":
                # 깊이를 바꿨으면 바뀐 표면에서 잘라낸다
                part = model3d.cut_letters_from_scan(de.displace_scan(scan, proj, result), proj, hm,
                                                     result.mask, margin)
                path = folder / f"{name}_글자_원본표면.{cut_fmt}"
                model3d.save_scan_mesh(part, path)
                return [path], f"삼각형 {len(part.faces):,}개"
            polygons = None
            if mode == "letters":
                outlines = vector.mask_to_outlines(result.mask, hm.pixel_mm)
                polygons = vector.outlines_to_polygons(outlines, hm.size_mm[1])
            mesh = model3d.make_print_model(result.mask, hm.valid, hm.pixel_mm, opts, result.residual, polygons)
            label = {"plate": "글자판", "letters": "입체글자", "relief": "표면복제"}[mode]
            path = folder / f"{name}_{label}.stl"
            model3d.save_mesh(mesh, path)
            rep = model3d.mesh_report(mesh)
            return [path], ", ".join(f"{k} {v}" for k, v in rep.items())

        self._busy_on("3D 모델을 만드는 중…")
        run_task(work, on_done=lambda r: self._saved(r[0], str(folder), r[1]), on_failed=self._export_failed)

    def _saved(self, files, folder: str, extra: str = ""):
        self._busy_off("저장했습니다.")
        names = "\n".join(Path(f).name for f in files)
        msg = f"다음 파일을 저장했습니다.\n\n{names}\n\n폴더: {folder}"
        if extra:
            msg += f"\n\n{extra}"
        box = QMessageBox(self)
        box.setWindowTitle("저장 완료")
        box.setText(msg)
        open_btn = box.addButton("폴더 열기", QMessageBox.ActionRole)
        box.addButton("확인", QMessageBox.AcceptRole)
        box.exec()
        if box.clickedButton() is open_btn:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _export_failed(self, msg, details):
        self._busy_off("저장 실패")
        self._error("저장 실패", msg, details)

    # ------------------------------------------------------------------
    # 글자 판독
    # ------------------------------------------------------------------
    def run_offline_ocr(self, opts: dict):
        if self.result is None:
            return
        lang = opts["lang"]
        if lang == "korean" and not ocr.korean_model_ready():
            if QMessageBox.question(self, "한국어 모델", "한국어 모델을 지금 내려받을까요? (약 4MB, 인터넷 필요)") \
                    == QMessageBox.Yes:
                self.download_korean()
            return
        img = self.result.images[opts["source"]]
        mask = self.result.mask
        self.ocr_panel.set_busy(True)
        self._busy_on("글자를 판독하는 중…")

        def done(res: ocr.OcrResult):
            self.ocr_panel.set_busy(False)
            self._busy_off("판독 완료")
            d = res.to_dict()
            d["options"]["source"] = opts["source"]
            self.ocr_panel.add_result(d)
            self._mark_dirty()
            if not res.text.strip():
                QMessageBox.information(self, "판독 결과", "읽은 글자가 없습니다. 감도나 영역을 조정해 보세요.")
            elif res.options.get("rotation_ccw"):
                deg = int(res.options["rotation_ccw"])
                if QMessageBox.question(
                        self, "글자 방향",
                        f"글자가 {deg}° 돌린 방향에서 가장 잘 읽혔습니다.\n"
                        "이미지를 그 방향으로 돌려 둘까요? (선택한 영역은 해제됩니다)") == QMessageBox.Yes:
                    v = self.params.rotation.value() + deg
                    self.params.rotation.setValue(((v + 180) % 360) - 180)

        def failed(msg, details):
            self.ocr_panel.set_busy(False)
            self._busy_off("판독 실패")
            self._error("판독 실패", msg, details)

        run_task(ocr.run_offline, img, mask, lang, opts["direction"], opts["auto_rotate"],
                 on_done=done, on_failed=failed)

    def run_ai_ocr(self, opts: dict):
        if self.result is None:
            return
        key = self.settings.value("api_key", "")
        if not key:
            QMessageBox.information(self, "API 키 필요", "AI 판독을 쓰려면 먼저 [설정]에서 Anthropic API 키를 입력하세요.")
            self.edit_settings()
            key = self.settings.value("api_key", "")
            if not key:
                return
        model = self.settings.value("ai_model", ocr.DEFAULT_AI_MODEL)
        sel = opts["images"]
        keys = ["enhanced", "hillshade", "rubbing"] if sel == "multi" else [sel]
        images = [self.result.images[k] for k in keys if k in self.result.images]
        hint = opts.get("hint", "")
        notes = self.project.notes
        if notes.name or notes.location:
            hint = f"{hint} (비석: {notes.name} {notes.location})".strip()
        self.ocr_panel.set_busy(True)
        self._busy_on("AI가 판독하는 중… (1분 이상 걸릴 수 있습니다)")

        def done(res: ocr.OcrResult):
            self.ocr_panel.set_busy(False)
            self._busy_off("AI 판독 완료")
            self.ocr_panel.add_result(res.to_dict())
            self._mark_dirty()

        def failed(msg, details):
            self.ocr_panel.set_busy(False)
            self._busy_off("AI 판독 실패")
            self._error("AI 판독 실패", msg, details)

        run_task(ocr.run_ai, images, key, model, hint, on_done=done, on_failed=failed)

    def download_korean(self):
        self._busy_on("한국어 OCR 모델을 내려받는 중…")

        def done(_):
            self._busy_off("한국어 모델 준비 완료")
            self.ocr_panel.refresh()

        def failed(msg, details):
            self._busy_off("내려받기 실패")
            self._error("내려받기 실패", msg, details)

        run_task(ocr.download_korean_model, on_done=done, on_failed=failed)

    def show_ocr_boxes(self):
        r = self.ocr_panel.current_result()
        if self.result is None or not r or not r.get("lines"):
            return
        res = ocr.OcrResult.from_dict(r)
        src = r.get("options", {}).get("source", "mask")
        img = self.result.images.get(src, self.result.images["mask"])
        k = int(r.get("options", {}).get("rotation_ccw", 0)) // 90
        img = np.ascontiguousarray(np.rot90(img, k))
        self._ocr_view = (ocr.draw_ocr_boxes(img, res), r)
        self._refresh_image_list()
        self.image_combo.setCurrentIndex(self.image_combo.findData(OCR_VIEW_KEY))
        self.tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # 프로젝트
    # ------------------------------------------------------------------
    def _collect_project(self):
        p = self.project
        p.scan_path = self.raw_scan.source_path or p.scan_path
        p.unit = self.params.unit_key()
        p.geometry = self.params.geometry_params()
        p.relief = self.params.relief_params()
        p.crop_mm = self.crop_mm
        p.ocr_results = self.ocr_panel.results()
        p.depth_edit = self.depth_panel.edit(self.depth_edit.overrides).to_dict()

    def save_project(self, save_as: bool = False) -> bool:
        if self.raw_scan is None:
            return False
        copy_scan = False
        if self.project.folder is None or save_as:
            default_parent = self.settings.value("last_project_parent", str(Path.home() / "Documents"))
            dlg = NewProjectDialog(self._base_name(), default_parent, self)
            if dlg.exec() != NewProjectDialog.Accepted:
                return False
            folder = dlg.folder()
            if (folder / "project.json").exists() and QMessageBox.question(
                    self, "덮어쓰기", f"이미 프로젝트가 있습니다. 덮어쓸까요?\n{folder}") != QMessageBox.Yes:
                return False
            self.settings.setValue("last_project_parent", str(folder.parent))
            self.project.folder = folder
            copy_scan = dlg.copy_scan.isChecked()
        self._collect_project()
        preview = self.result.images.get("rubbing") if self.result else None
        try:
            pf = save_project(self.project, preview, copy_scan)
        except Exception as e:  # noqa: BLE001
            self._error("프로젝트 저장 실패", str(e))
            return False
        if copy_scan:
            self.raw_scan.source_path = self.project.scan_path
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"프로젝트를 저장했습니다: {pf.parent}", 8000)
        return True

    def open_project_dialog(self):
        if not self._confirm_discard():
            return
        start = self.settings.value("last_project_parent", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", start, "프로젝트 (project.json)")
        if path:
            self.open_project(path)

    def open_project(self, path: str):
        try:
            proj = load_project(path)
        except Exception as e:  # noqa: BLE001
            self._error("프로젝트 열기 실패", str(e))
            return
        scan_path = proj.resolve_scan_path()
        if scan_path is None:
            QMessageBox.information(self, "스캔 파일 찾기",
                                    f"스캔 파일을 찾을 수 없습니다.\n{proj.scan_path}\n\n파일 위치를 직접 골라 주세요.")
            p, _ = QFileDialog.getOpenFileName(self, "스캔 파일 찾기", str(proj.folder),
                                               "3D 스캔 (*.ply *.obj *.stl)")
            if not p:
                return
            scan_path = Path(p)
            proj.scan_path = str(scan_path)
        self.settings.setValue("last_project_parent", str(proj.folder.parent))
        self.project = proj
        self.crop_mm = proj.crop_mm

        def after():
            self.params.set_params(proj.geometry, proj.relief, proj.unit)
            self.params.set_color_available(self.raw_scan.colors is not None)
            self.ocr_panel.set_results(proj.ocr_results)
            self.dirty = False
            self._update_title()

        self.load_scan_file(str(scan_path), unit=proj.unit, after=after)

    def edit_notes(self):
        dlg = NotesDialog(self.project.notes, self)
        if dlg.exec() == NotesDialog.Accepted:
            self.project.notes = dlg.notes()
            self.dirty = True
            self._update_title()

    def edit_settings(self):
        dlg = SettingsDialog(self.settings.value("api_key", ""),
                             self.settings.value("ai_model", ocr.DEFAULT_AI_MODEL), self)
        if dlg.exec() == SettingsDialog.Accepted:
            self.settings.setValue("api_key", dlg.key.text().strip())
            self.settings.setValue("ai_model", dlg.model.text().strip() or ocr.DEFAULT_AI_MODEL)

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(
            self, "저장하지 않은 변경",
            "저장하지 않은 변경 내용이 있습니다. 프로젝트를 저장할까요?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if r == QMessageBox.Save:
            return self.save_project()
        return r == QMessageBox.Discard

    def about(self):
        TextDialog(
            "프로그램 정보",
            f"{APP_TITLE} {__version__}\n\n"
            "POP 4 등 3D 스캐너로 스캔한 비석의 음각·양각 글자를 추출합니다.\n\n"
            "사용한 오픈소스: Python, PySide6(Qt), NumPy, SciPy, OpenCV, trimesh, PyVista(VTK), "
            "shapely, ezdxf, RapidOCR(PaddleOCR 모델), Anthropic SDK",
            self,
        ).exec()

    # ------------------------------------------------------------------
    # 끌어다 놓기 / 닫기
    # ------------------------------------------------------------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).name == "project.json":
                if self._confirm_discard():
                    self.open_project(p)
                return
            if Path(p).suffix.lower() in SUPPORTED_EXTENSIONS:
                if self._confirm_discard():
                    self.project = Project()
                    self.crop_mm = None
                    self.load_scan_file(p)
                return

    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        self.viewer3d.close_viewer()
        e.accept()
