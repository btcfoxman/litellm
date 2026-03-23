import os
import sys
import json
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.flow2api.image_generation.transformation import (
    Flow2APIImageGenerationConfig,
)
from litellm.llms.flow2api.videos.veo_transformation import Flow2APIVideoConfig
from litellm.types.utils import ImageResponse
from litellm.utils import get_optional_params_image_gen


def _make_response(body: str) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "http://127.0.0.1:4020/v1/chat/completions"),
        content=body.encode("utf-8"),
    )


def test_flow2api_image_request_uses_stream_mode():
    cfg = Flow2APIImageGenerationConfig()
    payload = cfg.transform_image_generation_request(
        model="gemini-3.1-flash-image-landscape",
        prompt="画一只猫",
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert payload["stream"] is True


def test_flow2api_image_request_includes_image_url_reference():
    cfg = Flow2APIImageGenerationConfig()
    payload = cfg.transform_image_generation_request(
        model="gemini-3.1-flash-image-landscape",
        prompt="transform this image",
        optional_params={"image_url": "https://example.com/input.jpg"},
        litellm_params={},
        headers={},
    )
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "https://example.com/input.jpg"


def test_flow2api_optional_params_keeps_input_reference():
    cfg = Flow2APIImageGenerationConfig()
    optional_params = get_optional_params_image_gen(
        model="gemini-3.1-flash-image-landscape",
        custom_llm_provider="flow2api",
        provider_config=cfg,
        input_reference=["data:image/jpeg;base64,ZmFrZQ=="],
    )
    assert optional_params["input_reference"] == ["data:image/jpeg;base64,ZmFrZQ=="]


def test_flow2api_image_sse_response_parsed_to_url():
    cfg = Flow2APIImageGenerationConfig()
    sse_body = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"处理中..."}}]}',
            'data: {"choices":[{"delta":{"content":"![Generated Image](https://example.com/a.png)"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
    )
    raw_response = _make_response(sse_body)
    result = cfg.transform_image_generation_response(
        model="gemini-3.1-flash-image-landscape",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"prompt": "画一只猫"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert result.data is not None
    assert result.data[0].url == "https://example.com/a.png"


def test_flow2api_image_plain_text_response_parsed_to_url():
    cfg = Flow2APIImageGenerationConfig()
    raw_response = _make_response(
        "![Generated Image](https://example.com/plain.png)"
    )
    result = cfg.transform_image_generation_response(
        model="gemini-3.1-flash-image-landscape",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"prompt": "draw an image"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert result.data is not None
    assert result.data[0].url == "https://example.com/plain.png"


def test_flow2api_image_portrait_2k_request_uses_target_model():
    cfg = Flow2APIImageGenerationConfig()
    payload = cfg.transform_image_generation_request(
        model="gemini-3.1-flash-image-portrait-2k",
        prompt="portrait 2k image",
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert payload["stream"] is True
    assert payload["model"] == "gemini-3.1-flash-image-portrait-2k"


def test_flow2api_image_portrait_4k_request_uses_target_model():
    cfg = Flow2APIImageGenerationConfig()
    payload = cfg.transform_image_generation_request(
        model="gemini-3.1-flash-image-portrait-4k",
        prompt="portrait 4k image",
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert payload["stream"] is True
    assert payload["model"] == "gemini-3.1-flash-image-portrait-4k"


def test_flow2api_video_request_uses_stream_mode():
    cfg = Flow2APIVideoConfig()
    request_payload, _, _ = cfg.transform_video_create_request(
        model="veo_3_1_t2v_fast_landscape",
        prompt="一只猫在草地上跑",
        api_base="http://127.0.0.1:4020/v1",
        video_create_optional_request_params={},
        litellm_params=MagicMock(),
        headers={},
    )
    assert request_payload["stream"] is True


def test_flow2api_video_i2v_portrait_fl_request_has_two_images():
    cfg = Flow2APIVideoConfig()
    request_payload, _, _ = cfg.transform_video_create_request(
        model="veo_3_1_i2v_s_fast_portrait_fl",
        prompt="transition between first and last frame",
        api_base="http://127.0.0.1:4020/v1",
        video_create_optional_request_params={
            "input_reference": [
                {"url": "https://example.com/frame-start.jpg"},
                {"url": "https://example.com/frame-end.jpg"},
            ]
        },
        litellm_params=MagicMock(),
        headers={},
    )
    assert request_payload["stream"] is True
    assert request_payload["model"] == "veo_3_1_i2v_s_fast_portrait_fl"
    content = request_payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 3
    assert content[1]["image_url"]["url"] == "https://example.com/frame-start.jpg"
    assert content[2]["image_url"]["url"] == "https://example.com/frame-end.jpg"


def test_flow2api_video_accepts_openai_aspect_ratio_param():
    cfg = Flow2APIVideoConfig()
    mapped = cfg.map_openai_params(
        video_create_optional_params={"aspect_ratio": "9:16"},
        model="veo_3_1_i2v_s_fast_portrait_fl",
        drop_params=False,
    )
    assert mapped["aspect_ratio"] == "9:16"


def test_flow2api_video_accepts_openai_duration_param():
    cfg = Flow2APIVideoConfig()
    mapped = cfg.map_openai_params(
        video_create_optional_params={"duration": "8"},
        model="veo_3_1_i2v_s_fast_portrait_fl",
        drop_params=False,
    )
    assert mapped["duration"] == "8"


def test_flow2api_video_accepts_end_image_aliases():
    cfg = Flow2APIVideoConfig()
    mapped = cfg.map_openai_params(
        video_create_optional_params={
            "image_url": "https://example.com/frame-start.jpg",
            "end_image_url": "https://example.com/frame-end.jpg",
            "last_image_url": "https://example.com/frame-end-last.jpg",
        },
        model="veo_3_1_i2v_s_fast_portrait_fl",
        drop_params=False,
    )
    assert mapped["image_url"] == "https://example.com/frame-start.jpg"
    assert mapped["end_image_url"] == "https://example.com/frame-end.jpg"
    assert mapped["last_image_url"] == "https://example.com/frame-end-last.jpg"


def test_flow2api_video_request_uses_image_and_end_image():
    cfg = Flow2APIVideoConfig()
    request_payload, _, _ = cfg.transform_video_create_request(
        model="veo_3_1_i2v_s_fast_portrait_fl",
        prompt="transition between first and last frame",
        api_base="http://127.0.0.1:4020/v1",
        video_create_optional_request_params={
            "image_url": "https://example.com/frame-start.jpg",
            "end_image_url": "https://example.com/frame-end.jpg",
        },
        litellm_params=MagicMock(),
        headers={},
    )
    content = request_payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 3
    assert content[1]["image_url"]["url"] == "https://example.com/frame-start.jpg"
    assert content[2]["image_url"]["url"] == "https://example.com/frame-end.jpg"


def test_flow2api_video_request_keeps_tail_frame_with_input_reference():
    cfg = Flow2APIVideoConfig()
    request_payload, _, _ = cfg.transform_video_create_request(
        model="veo_3_1_i2v_s_fast_fl",
        prompt="from first frame to last frame",
        api_base="http://127.0.0.1:4020/v1",
        video_create_optional_request_params={
            "image_url": "https://example.com/frame-start.jpg",
            "input_reference": "https://example.com/frame-start.jpg",
            "end_image_url": "https://example.com/frame-end.jpg",
        },
        litellm_params=MagicMock(),
        headers={},
    )
    content = request_payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 3
    assert content[1]["image_url"]["url"] == "https://example.com/frame-start.jpg"
    assert content[2]["image_url"]["url"] == "https://example.com/frame-end.jpg"


def test_flow2api_video_duration_maps_to_seconds_in_request():
    cfg = Flow2APIVideoConfig()
    request_payload, _, _ = cfg.transform_video_create_request(
        model="veo_3_1_t2v_fast_landscape",
        prompt="a running cat",
        api_base="http://127.0.0.1:4020/v1",
        video_create_optional_request_params={"duration": 10},
        litellm_params=MagicMock(),
        headers={},
    )
    assert request_payload["seconds"] == "10"


def test_flow2api_video_ignores_non_standard_params():
    cfg = Flow2APIVideoConfig()
    mapped = cfg.map_openai_params(
        video_create_optional_params={
            "duration_band": "medium",
            "video_provider": "sora",
            "generation_plan": {"strategy": "single"},
        },
        model="veo_3_1_i2v_s_fast_portrait_fl",
        drop_params=False,
    )
    assert mapped == {}


def test_flow2api_video_sse_response_parsed_to_video_url():
    cfg = Flow2APIVideoConfig()
    sse_body = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"处理中..."}}]}',
            'data: {"choices":[{"delta":{"content":"```html\\n<video src=\'https://example.com/v.mp4\' controls></video>\\n```"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
    )
    raw_response = _make_response(sse_body)
    result = cfg.transform_video_create_response(
        model="veo_3_1_t2v_fast_landscape",
        raw_response=raw_response,
        logging_obj=MagicMock(),
        request_data={},
    )
    assert result.status == "completed"
    assert result.video_url == "https://example.com/v.mp4"


def test_flow2api_video_plain_text_response_parsed_to_video_url():
    cfg = Flow2APIVideoConfig()
    raw_response = _make_response(
        "```html\n<video src='https://example.com/raw.mp4' controls></video>\n```"
    )
    result = cfg.transform_video_create_response(
        model="veo_3_1_t2v_fast_landscape",
        raw_response=raw_response,
        logging_obj=MagicMock(),
        request_data={},
    )
    assert result.status == "completed"
    assert result.video_url == "https://example.com/raw.mp4"


def test_flow2api_video_error_payload_returns_upstream_message():
    cfg = Flow2APIVideoConfig()
    raw_response = _make_response(
        json.dumps(
            {
                "error": "生成失败: Failed to obtain reCAPTCHA token",
                "performance": {
                    "status": "failed",
                    "error": "生成失败: Failed to obtain reCAPTCHA token",
                },
            },
            ensure_ascii=False,
        )
    )
    with pytest.raises(Exception) as exc:
        cfg.transform_video_create_response(
            model="veo_3_1_i2v_s_fast_portrait_fl",
            raw_response=raw_response,
            logging_obj=MagicMock(),
            request_data={},
        )
    assert "Failed to obtain reCAPTCHA token" in str(exc.value)
