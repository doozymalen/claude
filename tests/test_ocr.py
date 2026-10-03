"""글자 판독 시험."""

import os
import sys
import types

import numpy as np
import pytest

from stele_extractor.core import ocr
from stele_extractor.core.pipeline import GeometryParams, ReliefParams, build_geometry, process_relief
from stele_extractor.core.synthetic import SyntheticOptions, _font_has, find_font, make_synthetic_scan


def _result(text, font=None, **kw):
    scan = make_synthetic_scan(SyntheticOptions(text=text, width_mm=100, height_mm=90, font_path=font, **kw))
    _, hm = build_geometry(scan, GeometryParams())
    return process_relief(hm, ReliefParams())


def test_segment_vertical_columns_right_to_left():
    m = np.zeros((100, 100), bool)
    m[10:30, 70:90] = True  # 오른쪽 열 첫 글자
    m[50:70, 70:90] = True  # 오른쪽 열 둘째 글자
    m[10:30, 10:30] = True  # 왼쪽 열
    items = ocr.segment_text(m, "vertical")
    assert len(items) == 2
    assert items[0]["box"][2] == 70  # 오른쪽 열이 먼저
    assert len(items[0]["chars"]) == 2


def test_segment_horizontal_lines():
    m = np.zeros((100, 100), bool)
    m[10:30, 10:90] = True
    m[60:80, 10:50] = True
    items = ocr.segment_text(m, "horizontal")
    assert [it["box"][0] for it in items] == [10, 60]


HANJA = "忠孝仁義大民韓國"
hanja_font = find_font(text=HANJA)
has_hanja_font = hanja_font is not None and _font_has(hanja_font, HANJA)


@pytest.mark.skipif(not has_hanja_font, reason="한자 글꼴 없음")
@pytest.mark.parametrize("direction,text,expected", [
    ("horizontal", "忠孝\n仁義", "忠孝\n仁義"),
    ("vertical", "大  民\n韓  國", "民國\n大韓"),  # 세로쓰기: 열 사이가 넓고 오른쪽 열(民國)부터
])
def test_offline_hanja(direction, text, expected):
    r = _result(text, seed=1)
    res = ocr.run_offline(r.images["mask"], r.mask, "hanja", direction, auto_rotate=True)
    assert res.text == expected


@pytest.mark.skipif(not os.environ.get("TEST_KOREAN_OCR"), reason="한국어 모델 내려받기가 필요한 시험")
def test_offline_korean():
    ocr.download_korean_model()
    assert ocr.korean_model_ready()
    r = _result("대한\n민국", seed=2)
    res = ocr.run_offline(r.images["mask"], r.mask, "korean", "horizontal", auto_rotate=True)
    assert "대한" in res.text.replace("\n", "") or "민국" in res.text.replace("\n", "")


def test_ai_request_shape(monkeypatch):
    """Claude API 호출 형식 확인 (실제 요청은 보내지 않음)."""
    captured = {}

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            block = types.SimpleNamespace(type="text", text="[판독문]\n大韓")
            return types.SimpleNamespace(content=[block], stop_reason="end_turn", model="claude-opus-5-5")

    class FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.beta = types.SimpleNamespace(messages=FakeMessages())

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    img = np.full((50, 80), 255, np.uint8)
    res = ocr.run_ai([img, img], api_key="sk-test", hint="조선 묘비")
    assert res.text.startswith("[판독문]")
    assert captured["model"] == ocr.DEFAULT_AI_MODEL
    assert captured["fallbacks"] == "default"
    assert captured["thinking"] == {"type": "adaptive"}
    content = captured["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert "조선 묘비" in content[-1]["text"]

    # 실제 SDK가 이 인자들을 받는지 확인
    import inspect

    from anthropic.resources.beta.messages import Messages

    params = inspect.signature(Messages.stream).parameters
    for name in ("model", "max_tokens", "thinking", "output_config", "betas", "fallbacks", "messages"):
        assert name in params, name


def test_ai_requires_key():
    with pytest.raises(ocr.OcrError):
        ocr.run_ai([np.zeros((5, 5), np.uint8)], api_key="")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
