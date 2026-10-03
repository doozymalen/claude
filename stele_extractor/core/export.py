"""결과 이미지를 실제 크기(1:1) 정보와 함께 저장한다."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .pipeline import IMAGE_NAMES, ReliefResult


def _nice_length(total_mm: float) -> float:
    """눈금자 길이: 이미지 폭의 1/4 이하인 보기 좋은 값."""
    target = total_mm / 4
    best = 1.0
    for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if v <= target:
            best = float(v)
    return best


def add_scale_bar(img: np.ndarray, pixel_mm: float) -> np.ndarray:
    """이미지 아래에 mm 눈금자를 붙인다."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    total = w * pixel_mm
    length_mm = _nice_length(total)
    bar_px = int(round(length_mm / pixel_mm))
    pad = max(40, h // 12)
    canvas = np.full((h + pad, w, 3), 255, np.uint8)
    canvas[:h] = img
    x0 = max(10, w // 40)
    y = h + pad // 2
    thick = max(2, pad // 12)
    cv2.rectangle(canvas, (x0, y - thick), (x0 + bar_px, y + thick), (0, 0, 0), -1)
    ticks = 10 if length_mm >= 10 else int(length_mm)
    for i in range(ticks + 1):
        x = x0 + int(round(i * bar_px / max(ticks, 1)))
        th = pad // 3 if i in (0, ticks) else pad // 6
        cv2.line(canvas, (x, y - th), (x, y), (0, 0, 0), max(1, thick // 2))
    label = f"{length_mm:g} mm"
    scale = pad / 50
    cv2.putText(canvas, label, (x0 + bar_px + 10, y + int(8 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * scale, (0, 0, 0), max(1, int(2 * scale)), cv2.LINE_AA)
    return canvas


def save_image(img: np.ndarray, path: str | Path, pixel_mm: float, scale_bar: bool = False) -> Path:
    """PNG/TIFF로 저장. DPI를 넣어 인쇄하면 실제 크기와 같다."""
    path = Path(path)
    if scale_bar:
        img = add_scale_bar(img, pixel_mm)
    dpi = 25.4 / pixel_mm
    pil = Image.fromarray(img)
    if path.suffix.lower() in (".tif", ".tiff"):
        pil.save(path, dpi=(dpi, dpi), compression="tiff_lzw")
    else:
        pil.save(path, dpi=(dpi, dpi))
    return path


def save_depth_tiff(residual: np.ndarray, valid: np.ndarray, path: str | Path, pixel_mm: float) -> Path:
    """깊이(mm) 값을 그대로 담은 32비트 TIFF. 빈 곳은 NaN."""
    data = residual.astype(np.float32).copy()
    data[~valid] = np.nan
    dpi = 25.4 / pixel_mm
    Image.fromarray(data, mode="F").save(path, dpi=(dpi, dpi))
    return Path(path)


def export_result(
    result: ReliefResult,
    folder: str | Path,
    base_name: str,
    which: list[str] | None = None,
    fmt: str = "png",
    scale_bar: bool = True,
    depth_tiff: bool = True,
    params: dict | None = None,
) -> list[Path]:
    """선택한 결과 이미지를 폴더에 저장하고 저장한 파일 목록을 돌려준다."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    hm = result.heightmap
    which = which or list(result.images.keys())
    saved = []
    for key in which:
        img = result.images.get(key)
        if img is None:
            continue
        name = IMAGE_NAMES.get(key, key)
        p = folder / f"{base_name}_{name}.{fmt}"
        saved.append(save_image(img, p, hm.pixel_mm, scale_bar))
    if depth_tiff:
        saved.append(save_depth_tiff(result.residual, hm.valid, folder / f"{base_name}_깊이값_mm.tif", hm.pixel_mm))

    meta = {
        "프로그램": "비석 글자 추출기",
        "저장 시각": datetime.now().isoformat(timespec="seconds"),
        "해상도_mm_per_pixel": hm.pixel_mm,
        "DPI": hm.dpi,
        "이미지_크기_px": [int(hm.shape[1]), int(hm.shape[0])],
        "실제_크기_mm": [round(hm.size_mm[0], 3), round(hm.size_mm[1], 3)],
        "글자_형태": "음각" if result.polarity == "engraved" else "양각",
        "배경_제거_크기_mm": result.background_mm,
        "통계": result.stats(),
        "설정": params or {},
        "표면_정보": {k: v for k, v in hm.info.items() if isinstance(v, (int, float, str, bool, list))},
        "안내": "이미지에 DPI 정보가 들어 있어 100% 크기로 인쇄하면 실제 크기와 같습니다.",
    }
    mp = folder / f"{base_name}_정보.json"
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    saved.append(mp)
    return saved
