import os
import sys
from unittest.mock import MagicMock

import httpx

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.aiapi.videos.transformation import AIAPIVideoConfig


def _make_response(body: dict) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "http://orchestration-service:8000/api/v1/sora/video/tasks"),
        json=body,
    )


def test_aiapi_create_request_infers_duration_and_reference_image_url():
    cfg = AIAPIVideoConfig()
    request_data, _, url = cfg.transform_video_create_request(
        model="sora2-portrait-10s",
        prompt="make a portrait sora video",
        api_base="http://orchestration-service:8000",
        video_create_optional_request_params={
            "input_reference": [{"url": "https://example.com/p.jpg"}],
            "user": "tester",
        },
        litellm_params=MagicMock(),
        headers={},
    )

    assert url.endswith("/api/v1/sora/video/tasks")
    assert request_data["duration"] == 10
    assert request_data["reference_image_url"] == "https://example.com/p.jpg"
    assert request_data["user_id"] == "tester"
    assert request_data["business_system"] == "litellm"


def test_aiapi_create_response_parses_running_task():
    cfg = AIAPIVideoConfig()
    response = _make_response(
        {
            "code": 200,
            "msg": "ok",
            "data": {
                "task_id": "task_sora_abc",
                "status": "RUNNING",
                "progress": "10%",
                "result_url": None,
            },
        }
    )
    result = cfg.transform_video_create_response(
        model="sora2-landscape-10s",
        raw_response=response,
        logging_obj=MagicMock(),
        request_data={"duration": 10},
    )

    assert result.id == "task_sora_abc"
    assert result.status == "in_progress"
    assert result.progress == 10
    assert result.seconds == "10"


def test_aiapi_status_response_parses_completed_video_url():
    cfg = AIAPIVideoConfig()
    response = _make_response(
        {
            "code": 200,
            "msg": "ok",
            "data": {
                "task_id": "task_sora_done",
                "status": "COMPLETED",
                "progress": 100,
                "result_url": "https://videos.openai.com/az/files/1.mp4",
            },
        }
    )
    result = cfg.transform_video_status_retrieve_response(
        raw_response=response,
        logging_obj=MagicMock(),
    )

    assert result.id == "task_sora_done"
    assert result.status == "completed"
    assert result.video_url == "https://videos.openai.com/az/files/1.mp4"
