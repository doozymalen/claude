"""비석별 프로젝트 폴더 저장/열기.

폴더 구조:
    <프로젝트 이름>/
        project.json   설정, 자른 영역, 메모, 판독 결과
        scan/          (선택) 원본 스캔 파일 복사본
        결과/          내보낸 이미지·벡터·3D 파일
        미리보기.png   목록에서 알아보기 쉽도록 탁본 미리보기
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .pipeline import GeometryParams, ReliefParams

PROJECT_FILE = "project.json"
FORMAT_VERSION = 1


@dataclass
class ProjectNotes:
    name: str = ""  # 비석 이름
    location: str = ""  # 위치
    date: str = ""  # 조사 날짜
    surveyor: str = ""  # 조사자
    memo: str = ""


@dataclass
class Project:
    folder: Path | None = None
    scan_path: str = ""  # 원본 스캔 경로 (절대 경로)
    scan_relpath: str = ""  # 프로젝트 폴더 기준 상대 경로 (폴더를 옮겨도 찾도록)
    unit: str = "mm"
    geometry: GeometryParams = field(default_factory=GeometryParams)
    relief: ReliefParams = field(default_factory=ReliefParams)
    crop_mm: list | None = None  # [u_min, u_max, v_min, v_max] 펼친 면 기준 mm
    notes: ProjectNotes = field(default_factory=ProjectNotes)
    ocr_results: list = field(default_factory=list)  # OcrResult.to_dict() 목록
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    modified: str = ""

    @property
    def project_file(self) -> Path | None:
        return None if self.folder is None else self.folder / PROJECT_FILE

    @property
    def results_dir(self) -> Path | None:
        return None if self.folder is None else self.folder / "결과"

    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "program": "비석 글자 추출기",
            "scan_path": self.scan_path,
            "scan_relpath": self.scan_relpath,
            "unit": self.unit,
            "geometry": asdict(self.geometry),
            "relief": asdict(self.relief),
            "crop_mm": self.crop_mm,
            "notes": asdict(self.notes),
            "ocr_results": self.ocr_results,
            "created": self.created,
            "modified": self.modified,
        }

    def resolve_scan_path(self) -> Path | None:
        """스캔 파일 위치를 찾는다 (상대 경로 우선)."""
        candidates = []
        if self.folder is not None and self.scan_relpath:
            candidates.append(self.folder / self.scan_relpath)
        if self.scan_path:
            candidates.append(Path(self.scan_path))
        for c in candidates:
            if c.exists():
                return c
        return None


def _filter_fields(cls, data: dict) -> dict:
    names = set(cls.__dataclass_fields__)
    return {k: v for k, v in (data or {}).items() if k in names}


def load_project(path: str | Path) -> Project:
    path = Path(path)
    if path.is_dir():
        path = path / PROJECT_FILE
    if not path.exists():
        raise FileNotFoundError(f"프로젝트 파일이 없습니다: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Project(
        folder=path.parent,
        scan_path=data.get("scan_path", ""),
        scan_relpath=data.get("scan_relpath", ""),
        unit=data.get("unit", "mm"),
        geometry=GeometryParams(**_filter_fields(GeometryParams, data.get("geometry"))),
        relief=ReliefParams(**_filter_fields(ReliefParams, data.get("relief"))),
        crop_mm=data.get("crop_mm"),
        notes=ProjectNotes(**_filter_fields(ProjectNotes, data.get("notes"))),
        ocr_results=data.get("ocr_results", []),
        created=data.get("created", ""),
        modified=data.get("modified", ""),
    )


def save_project(project: Project, preview: np.ndarray | None = None, copy_scan: bool = False) -> Path:
    if project.folder is None:
        raise ValueError("프로젝트 폴더가 정해지지 않았습니다.")
    folder = Path(project.folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "결과").mkdir(exist_ok=True)

    if copy_scan and project.scan_path:
        src = Path(project.scan_path)
        dest_dir = folder / "scan"
        dest = dest_dir / src.name
        if src.exists() and src.resolve() != dest.resolve():
            dest_dir.mkdir(exist_ok=True)
            shutil.copy2(src, dest)
            # OBJ의 재질/텍스처 파일도 함께 복사
            if src.suffix.lower() == ".obj":
                for extra in src.parent.glob(src.stem + "*"):
                    if extra.suffix.lower() in (".mtl", ".jpg", ".jpeg", ".png") and extra != src:
                        shutil.copy2(extra, dest_dir / extra.name)
            project.scan_path = str(dest)

    if project.scan_path:
        try:
            project.scan_relpath = os.path.relpath(project.scan_path, folder)
        except ValueError:  # 드라이브가 다른 경우 (Windows)
            project.scan_relpath = ""

    project.modified = datetime.now().isoformat(timespec="seconds")
    pf = folder / PROJECT_FILE
    tmp = pf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(pf)

    if preview is not None:
        img = preview
        h, w = img.shape[:2]
        s = 800 / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", img)
        if ok:
            (folder / "미리보기.png").write_bytes(buf.tobytes())
    return pf


def crop_to_pixels(crop_mm: list, u0: float, v0: float, pixel_mm: float) -> tuple[int, int, int, int]:
    """mm 영역 → 높이 지도 픽셀 (r0, r1, c0, c1)."""
    u_min, u_max, v_min, v_max = crop_mm
    c0 = int(round((u_min - u0) / pixel_mm))
    c1 = int(round((u_max - u0) / pixel_mm))
    r0 = int(round((v0 - v_max) / pixel_mm))
    r1 = int(round((v0 - v_min) / pixel_mm))
    return r0, r1, c0, c1


def pixels_to_crop(r0: int, r1: int, c0: int, c1: int, u0: float, v0: float, pixel_mm: float) -> list:
    return [u0 + c0 * pixel_mm, u0 + c1 * pixel_mm, v0 - r1 * pixel_mm, v0 - r0 * pixel_mm]
