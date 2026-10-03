"""글자 판독(OCR).

- 오프라인: RapidOCR(PaddleOCR 모델, PC에서 무료 실행).
    * 한자·영문: 프로그램에 들어 있는 기본 모델
    * 한글: 처음 쓸 때 한 번 내려받는 한국어 모델
- AI 판독: Claude API (인터넷 + API 키 필요). 마모된 글자·옛 글자에 강하다.
"""

from __future__ import annotations

import base64
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

OFFLINE_LANGS = {"hanja": "한자·영문 (기본 모델)", "korean": "한글 (한국어 모델)"}

KOREAN_MODEL_URL = (
    "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv1/korean_mobile_v2.0_rec_infer.onnx"
)
KOREAN_DICT_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/dict/korean_dict.txt"
)

DEFAULT_AI_MODEL = "claude-opus-5-5"

AI_PROMPT = """이 이미지는 비석(석비)의 3D 스캔에서 만든 글자 이미지입니다. \
흰 바탕에 어둡게 보이는 부분이 새겨진 글자입니다.{extra}

다음 형식으로 한국어로 답해 주세요.

[판독문]
비석에 새겨진 글자를 원래 배열(줄/열) 그대로 옮겨 적으세요.
- 세로쓰기라면 오른쪽 열부터 한 열을 한 줄로 적으세요.
- 읽을 수 없는 글자는 □, 확실하지 않은 글자는 글자 뒤에 (?)를 붙이세요.
- 추측으로 글자를 채우지 마세요.

[문자 종류]
한자/한글/옛한글/기타 중 무엇인지, 서체 특징이 있으면 짧게.

[풀이]
뜻을 알 수 있으면 현대 한국어로 짧게 풀이하세요. 모르면 '알 수 없음'.

[메모]
판독이 어려운 부분과 그 이유(마모, 깨짐 등)."""


@dataclass
class OcrLine:
    text: str
    score: float
    box: list  # [[x, y] * 4] 이미지 픽셀 좌표


@dataclass
class OcrResult:
    engine: str
    text: str
    lines: list[OcrLine] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    options: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "OcrResult":
        lines = [OcrLine(**ln) for ln in d.get("lines", [])]
        return OcrResult(
            engine=d.get("engine", ""),
            text=d.get("text", ""),
            lines=lines,
            created=d.get("created", ""),
            options=d.get("options", {}),
        )


class OcrError(Exception):
    pass


# ---------------------------------------------------------------------------
# 공통 준비
# ---------------------------------------------------------------------------


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(Path.home(), ".local", "share")
    d = Path(base) / "SteleExtractor"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 오프라인 OCR
# ---------------------------------------------------------------------------


def korean_model_paths() -> tuple[Path, Path]:
    d = app_data_dir() / "models"
    return d / "korean_mobile_v2.0_rec_infer.onnx", d / "korean_dict.txt"


def korean_model_ready() -> bool:
    m, k = korean_model_paths()
    return m.exists() and k.exists() and m.stat().st_size > 100_000


def download_korean_model(progress=None) -> None:
    m, k = korean_model_paths()
    m.parent.mkdir(parents=True, exist_ok=True)
    for url, dest in ((KOREAN_DICT_URL, k), (KOREAN_MODEL_URL, m)):
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        if progress:
            progress(f"내려받는 중: {dest.name}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise OcrError(f"한국어 모델을 내려받지 못했습니다. 인터넷 연결을 확인하세요.\n({e})") from e


_ENGINES: dict = {}


def _engine(lang: str):
    if lang in _ENGINES:
        return _ENGINES[lang]
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as e:
        raise OcrError(f"오프라인 OCR 구성요소를 불러오지 못했습니다: {e}") from e
    kwargs = {}
    if lang == "korean":
        if not korean_model_ready():
            raise OcrError("한국어 모델이 아직 없습니다. 먼저 내려받아 주세요.")
        m, k = korean_model_paths()
        kwargs = {"rec_model_path": str(m), "rec_keys_path": str(k), "rec_img_shape": [3, 32, 320]}
    eng = RapidOCR(**kwargs)
    _ENGINES[lang] = eng
    return eng


def _segments(profile: np.ndarray, min_gap: int = 1) -> list[tuple[int, int]]:
    """1차원 투영값에서 내용이 있는 구간들."""
    on = profile > max(profile.max() * 0.01, 0)
    segs, start = [], None
    gap = 0
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            gap = 0
            end = i + 1
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                segs.append((start, end))
                start = None
    if start is not None:
        segs.append((start, end))
    return segs


def _drop_thin(segs, ratio=0.25):
    if not segs:
        return segs
    med = np.median([b - a for a, b in segs])
    return [s for s in segs if (s[1] - s[0]) >= ratio * med]


def segment_text(mask: np.ndarray, direction: str = "horizontal") -> list[dict]:
    """글자 마스크를 읽는 순서대로 줄(가로) 또는 열(세로)로 나눈다.

    돌려주는 항목: {"box": (r0, r1, c0, c1), "chars": [(r0, r1, c0, c1), ...]}
    """
    m = mask.astype(bool)
    if not m.any():
        return []
    gap = max(2, int(min(m.shape) * 0.01))
    out = []
    if direction == "vertical":
        cols = _drop_thin(_segments(m.sum(axis=0), gap))
        for c0, c1 in sorted(cols, key=lambda s: -s[0]):  # 오른쪽 열부터
            col = m[:, c0:c1]
            rows = _segments(col.sum(axis=1), 1)
            if not rows:
                continue
            width = c1 - c0
            # 한 글자 안의 틈(예: 三)으로 잘게 나뉜 조각을 글자 크기만큼 합친다
            chars, cur = [], list(rows[0])
            for a, b in rows[1:]:
                if b - cur[0] <= 1.25 * width:
                    cur[1] = b
                else:
                    chars.append(tuple(cur))
                    cur = [a, b]
            chars.append(tuple(cur))
            chars = [(a, b, c0, c1) for a, b in chars]
            out.append({"box": (chars[0][0], chars[-1][1], c0, c1), "chars": chars})
    else:
        rows = _drop_thin(_segments(m.sum(axis=1), gap))
        for r0, r1 in rows:
            line = m[r0:r1]
            cs = _segments(line.sum(axis=0), 1)
            if not cs:
                continue
            out.append({"box": (r0, r1, cs[0][0], cs[-1][1]), "chars": []})
    return out


def _gap_fraction(profile: np.ndarray) -> float:
    """글자 영역 안에서 비어 있는(여백) 칸의 비율."""
    nz = np.nonzero(profile)[0]
    if len(nz) < 2:
        return 0.0
    inner = profile[nz[0]:nz[-1] + 1]
    return float((inner <= inner.max() * 0.01).mean())


def _layout_agreement(mask: np.ndarray, direction: str) -> float:
    """쓰기 방향과 여백 모양이 맞으면 +, 반대면 - (-1~1).

    세로쓰기는 열 사이 세로 여백이, 가로쓰기는 줄 사이 가로 여백이 뚜렷하다.
    """
    col_gap = _gap_fraction(mask.sum(axis=0))
    row_gap = _gap_fraction(mask.sum(axis=1))
    diff = col_gap - row_gap if direction == "vertical" else row_gap - col_gap
    return float(np.clip(diff * 5, -1, 1))


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _pad(img: np.ndarray, p: int) -> np.ndarray:
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _line_strip(img: np.ndarray, item: dict, direction: str) -> np.ndarray:
    """인식기에 넣을 가로 한 줄 이미지 (세로 열은 글자를 옆으로 늘어놓는다)."""
    r0, r1, c0, c1 = item["box"]
    if direction != "vertical":
        crop = img[r0:r1, c0:c1]
        return _pad(crop, max(4, (r1 - r0) // 6))
    size = c1 - c0
    tiles = []
    for a, b, x0, x1 in item["chars"]:
        crop = img[a:b, x0:x1]
        side = max(size, b - a)
        sq = np.full((side, side, 3), 255, np.uint8)
        oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
        sq[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
        tiles.append(sq)
        tiles.append(np.full((size, max(2, size // 4), 3), 255, np.uint8))
    strip = np.hstack(tiles[:-1]) if tiles else np.full((size, size, 3), 255, np.uint8)
    return _pad(strip, max(4, size // 6))


def run_offline(
    img: np.ndarray,
    mask: np.ndarray,
    lang: str = "hanja",
    direction: str = "horizontal",
    auto_rotate: bool = False,
) -> OcrResult:
    """img: 인식할 이미지(흰 바탕 검은 글자), mask: 글자 위치(줄/열 나누기용).

    direction: horizontal(가로쓰기, 위 줄부터) / vertical(세로쓰기, 오른쪽 열부터).
    auto_rotate: 90도씩 돌려 가며 가장 잘 읽히는 방향을 고른다.
        고른 방향은 options["rotation_ccw"] (반시계 방향 각도)에 기록된다.
    """
    if not auto_rotate:
        return _run_offline_once(img, mask, lang, direction)
    best, best_score = None, -1.0
    for k in range(4):
        m = np.ascontiguousarray(np.rot90(mask, k))
        res = _run_offline_once(np.ascontiguousarray(np.rot90(img, k)), m, lang, direction)
        score = sum(ln.score * len(ln.text) for ln in res.lines)
        # 인식기는 옆으로 누운 글자도 꽤 읽어 내므로, 줄/열 사이 여백 모양으로 한 번 더 가린다.
        score *= 1.0 + 0.3 * _layout_agreement(m, direction)
        res.options["rotation_ccw"] = 90 * k
        if score > best_score:
            best, best_score = res, score
    return best


def _run_offline_once(img: np.ndarray, mask: np.ndarray, lang: str, direction: str) -> OcrResult:
    eng = _engine(lang)
    bgr = _to_bgr(img)
    items = segment_text(mask, direction)
    if not items:
        return OcrResult(engine=f"오프라인 OCR ({OFFLINE_LANGS.get(lang, lang)})", text="",
                         options={"lang": lang, "direction": direction})
    strips = []
    for it in items:
        strip = _line_strip(bgr, it, direction)
        # 인식 모델이 읽기 좋은 크기 (글자 높이 약 48~64픽셀)
        h = strip.shape[0]
        target = 64
        s = target / max(h, 1)
        strips.append(cv2.resize(strip, (max(8, int(strip.shape[1] * s)), target),
                                 interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
    rec, _ = eng.text_rec(strips)
    lines = []
    for it, r in zip(items, rec):
        r0, r1, c0, c1 = it["box"]
        box = [[c0, r0], [c1, r0], [c1, r1], [c0, r1]]
        text = (r[0] or "").replace(" ", "") if direction == "vertical" else (r[0] or "")
        lines.append(OcrLine(text=text, score=float(r[1]), box=box))
    text = "\n".join(ln.text for ln in lines)
    return OcrResult(
        engine=f"오프라인 OCR ({OFFLINE_LANGS.get(lang, lang)})",
        text=text,
        lines=lines,
        options={"lang": lang, "direction": direction, "rotation_ccw": 0},
    )


def draw_ocr_boxes(img: np.ndarray, result: OcrResult) -> np.ndarray:
    """판독한 줄/열 위치를 이미지 위에 빨간 상자로 표시 (확인용, RGB)."""
    out = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    out = out.copy()
    t = max(2, min(out.shape[:2]) // 300)
    for i, ln in enumerate(result.lines):
        b = np.asarray(ln.box, dtype=np.int32)
        cv2.polylines(out, [b], True, (230, 30, 30), t)
        cv2.putText(out, str(i + 1), (int(b[0][0]) + 3, int(b[0][1]) + 12 * t), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * t, (230, 30, 30), max(1, t // 2), cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# AI 판독 (Claude API)
# ---------------------------------------------------------------------------


def encode_png(img: np.ndarray, max_side: int = 1568) -> str:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise OcrError("이미지를 PNG로 변환하지 못했습니다.")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def run_ai(
    images: list[np.ndarray],
    api_key: str,
    model: str = DEFAULT_AI_MODEL,
    hint: str = "",
) -> OcrResult:
    """글자 이미지(여러 장 가능)를 Claude에게 보내 판독문을 받는다."""
    try:
        import anthropic
    except Exception as e:
        raise OcrError(f"AI 판독 구성요소를 불러오지 못했습니다: {e}") from e
    if not api_key:
        raise OcrError("API 키가 없습니다. [설정]에서 Anthropic API 키를 입력하세요.")

    extra = ""
    if len(images) > 1:
        extra = "\n같은 부분을 서로 다른 방식(글자 강조, 음영 기복 등)으로 만든 이미지가 여러 장 있습니다. 모두 참고하세요."
    if hint.strip():
        extra += f"\n참고 정보: {hint.strip()}"
    content = []
    for img in images:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": encode_png(img)},
        })
    content.append({"type": "text", "text": AI_PROMPT.format(extra=extra)})

    client = anthropic.Anthropic(api_key=api_key, timeout=600.0)
    try:
        with client.beta.messages.stream(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        raise OcrError("API 키가 올바르지 않습니다. [설정]에서 키를 확인하세요.") from e
    except anthropic.PermissionDeniedError as e:
        raise OcrError(f"이 API 키로는 사용할 수 없습니다: {e.message}") from e
    except anthropic.NotFoundError as e:
        raise OcrError(f"모델 이름을 확인하세요 ({model}): {e.message}") from e
    except anthropic.RateLimitError as e:
        raise OcrError("요청이 너무 많습니다. 잠시 후 다시 시도하세요.") from e
    except anthropic.APIStatusError as e:
        raise OcrError(f"AI 서버 오류 ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise OcrError("AI 서버에 연결하지 못했습니다. 인터넷 연결을 확인하세요.") from e

    if response.stop_reason == "refusal":
        raise OcrError("AI가 이 요청을 처리하지 않았습니다. 이미지나 참고 정보를 바꿔 다시 시도하세요.")
    text = "\n".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
    if response.stop_reason == "max_tokens":
        text += "\n\n(응답이 길어 중간에 잘렸습니다.)"
    return OcrResult(
        engine=f"AI 판독 ({response.model})",
        text=text,
        options={"model": model, "images": len(images), "hint": hint},
    )
