import base64
import json
import re
import time
from io import BufferedReader
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, quote, urlparse

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.openai.image_edit.transformation import ImageEditRequestUtils
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any


class Flow2APIVideoConfig(BaseVideoConfig):
    """
    Map OpenAI /v1/videos requests to flow2api /v1/chat/completions.

    flow2api video provider in LiteLLM only supports veo model families.
    """

    _INLINE_VIDEO_ID_PREFIX = "flow2api_inline_"
    _VIDEO_HTML_PATTERN = re.compile(
        r"<video[^>]*src=['\"](?P<url>[^'\"]+)['\"]", re.I
    )
    _URL_PATTERN = re.compile(r"(?P<url>https?://[^\s'\"`<>]+)")
    _MARKDOWN_LINK_PATTERN = re.compile(r"\((?P<url>https?://[^)\s]+)\)")

    def get_supported_openai_params(self, model: str) -> list:
        self._validate_veo_model(model)
        return [
            "model",
            "prompt",
            "input_reference",
            "image_url",
            "end_image_url",
            "last_image_url",
            "seconds",
            "duration",
            "size",
            "aspect_ratio",
            "user",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        self._validate_veo_model(model)
        supported = set(self.get_supported_openai_params(model))
        mapped: Dict[str, Any] = {}
        for key, value in video_create_optional_params.items():
            if key in supported:
                mapped[key] = value
            else:
                # Keep flow2api provider permissive: ignore non-standard orchestration params.
                continue
        return mapped

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        resolved_api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("FLOW2API_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError(
                "FLOW2API API key is required. Set FLOW2API_API_KEY env var or pass api_key."
            )
        headers["Authorization"] = f"Bearer {resolved_api_key}"
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        resolved_base = (api_base or "http://127.0.0.1:4020/v1").rstrip("/")
        if not resolved_base.endswith("/v1"):
            resolved_base = f"{resolved_base}/v1"
        return resolved_base

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        self._validate_veo_model(model)
        provider_model = video_create_optional_request_params.get("model") or model
        self._validate_veo_model(str(provider_model))
        message_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        source_images = self._collect_source_images(video_create_optional_request_params)
        normalized_images: List[str] = []
        seen_images: set[str] = set()
        for image_item in source_images:
            normalized = self._to_image_url(image_item)
            if not normalized or normalized in seen_images:
                continue
            seen_images.add(normalized)
            normalized_images.append(normalized)

        for image_url in normalized_images:
            message_content.append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

        request_payload: Dict[str, Any] = {
            "model": provider_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        prompt if len(message_content) == 1 else message_content
                    ),
                }
            ],
            # flow2api returns real generation results only in stream mode.
            "stream": True,
        }
        size_value = video_create_optional_request_params.get("size")
        if size_value is not None:
            normalized_size = str(size_value).strip()
            lowered_size = normalized_size.lower()
            if lowered_size in {"sd", "hd", "fhd", "2k", "4k"}:
                normalized_size = lowered_size
            request_payload["size"] = normalized_size
        seconds_value = video_create_optional_request_params.get("seconds")
        if seconds_value is None:
            seconds_value = video_create_optional_request_params.get("duration")
        if seconds_value is not None:
            request_payload["seconds"] = str(seconds_value)
        if video_create_optional_request_params.get("aspect_ratio") is not None:
            request_payload["aspect_ratio"] = str(
                video_create_optional_request_params["aspect_ratio"]
            )

        return request_payload, [], f"{api_base}/chat/completions"

    def _collect_source_images(
        self, video_create_optional_request_params: Dict[str, Any]
    ) -> List[Any]:
        images: List[Any] = []

        def _extend(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, list):
                for item in value:
                    _extend(item)
                return
            images.append(value)

        _extend(video_create_optional_request_params.get("input_reference"))
        _extend(video_create_optional_request_params.get("image_url"))

        end_frame = video_create_optional_request_params.get("end_image_url")
        if end_frame is None:
            end_frame = video_create_optional_request_params.get("last_image_url")
        _extend(end_frame)

        return images

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        response_json, content = self._extract_response_payload_and_content(raw_response)
        self._raise_for_error_payload(
            response_json=response_json, headers=raw_response.headers
        )
        video_url = self._extract_video_url(content)
        if not video_url:
            raise self.get_error_class(
                error_message="flow2api video response did not include a valid video URL.",
                status_code=502,
                headers=raw_response.headers,
            )

        raw_video_id = self._encode_inline_video_id(video_url)
        video_data: Dict[str, Any] = {
            "id": raw_video_id,
            "object": "video",
            "status": "completed",
            "created_at": int(time.time()),
            "completed_at": int(time.time()),
            "progress": 100,
            "model": model,
            "video_url": video_url,
            "seconds": (
                self._safe_to_str(request_data, "seconds")
                or self._safe_to_str(request_data, "duration")
            ),
            "size": self._safe_to_str(request_data, "size"),
        }
        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]
        video_obj._hidden_params["video_url"] = video_url

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )
        return video_obj

    def _raise_for_error_payload(
        self, response_json: Dict[str, Any], headers: Union[dict, httpx.Headers]
    ) -> None:
        if not isinstance(response_json, dict):
            return

        error_value: Optional[Any] = response_json.get("error")
        performance = response_json.get("performance")
        if error_value is None and isinstance(performance, dict):
            if str(performance.get("status", "")).lower() == "failed":
                error_value = performance.get("error")

        if error_value is None:
            return

        if isinstance(error_value, dict):
            error_message = (
                error_value.get("message")
                or error_value.get("error")
                or json.dumps(error_value, ensure_ascii=False)
            )
        else:
            error_message = str(error_value)

        raise self.get_error_class(
            error_message=error_message or "flow2api request failed.",
            status_code=502,
            headers=headers,
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        video_url = self._decode_inline_video_id(original_video_id)
        if not video_url:
            raise ValueError(
                "flow2api video id does not contain an inline video URL for content retrieval."
            )
        return video_url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        return raw_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video remix is not supported by flow2api.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by flow2api.")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video list is not supported by flow2api.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video list is not supported by flow2api.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video delete is not supported by flow2api.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by flow2api.")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        # flow2api does not expose an OpenAI-style video status endpoint;
        # call a lightweight endpoint and reconstruct status from encoded id.
        original_video_id = extract_original_video_id(video_id)
        encoded_video_id = quote(original_video_id, safe="")
        return f"{api_base}/models?video_id={encoded_video_id}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        request_url = str(raw_response.request.url)
        query = parse_qs(urlparse(request_url).query)
        original_video_id = query.get("video_id", [""])[0]
        if not original_video_id:
            raise self.get_error_class(
                error_message="Missing flow2api video_id query parameter.",
                status_code=500,
                headers=raw_response.headers,
            )

        video_url = self._decode_inline_video_id(original_video_id)
        if not video_url:
            raise self.get_error_class(
                error_message="Invalid flow2api inline video id.",
                status_code=500,
                headers=raw_response.headers,
            )

        video_obj = VideoObject(
            id=original_video_id,
            object="video",
            status="completed",
            created_at=int(time.time()),
            completed_at=int(time.time()),
            progress=100,
            video_url=video_url,
        )
        video_obj._hidden_params["video_url"] = video_url
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, None
            )
        return video_obj

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _extract_choice_content(self, response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            return ""
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return ""

    def _extract_video_url(self, content: str) -> Optional[str]:
        if not content:
            return None
        for pattern in (
            self._VIDEO_HTML_PATTERN,
            self._MARKDOWN_LINK_PATTERN,
            self._URL_PATTERN,
        ):
            match = pattern.search(content)
            if match:
                url = match.group("url").strip()
                if url:
                    return url
        return None

    def _extract_response_payload_and_content(
        self, raw_response: httpx.Response
    ) -> Tuple[Dict[str, Any], str]:
        raw_text = raw_response.text or ""
        sse_content = self._extract_content_from_sse(raw_text)
        if sse_content:
            return {"sse": True}, sse_content

        stripped_text = raw_text.strip()
        if not stripped_text:
            raise self.get_error_class(
                error_message="flow2api returned an empty response body.",
                status_code=502,
                headers=raw_response.headers,
            )

        # Some flow2api deployments return plain text/markdown instead of JSON.
        # Keep the raw content path so URL extraction still works.
        try:
            response_json = raw_response.json()
            return response_json, self._extract_choice_content(response_json)
        except Exception:
            return {"raw_text": True}, stripped_text

    def _extract_content_from_sse(self, raw_text: str) -> str:
        if "data:" not in raw_text:
            return ""

        content_parts: List[str] = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            data_segment = stripped[5:].strip()
            if not data_segment or data_segment == "[DONE]":
                continue

            try:
                chunk_json = json.loads(data_segment)
            except Exception:
                continue

            choices = chunk_json.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta")
            if isinstance(delta, dict):
                delta_content = delta.get("content")
                if isinstance(delta_content, str):
                    content_parts.append(delta_content)

            message = choice.get("message")
            if isinstance(message, dict):
                message_content = message.get("content")
                if isinstance(message_content, str):
                    content_parts.append(message_content)
                elif isinstance(message_content, list):
                    for part in message_content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str):
                                content_parts.append(text)

        return "".join(content_parts).strip()

    def _to_image_url(self, image: Any) -> Optional[str]:
        if image is None:
            return None

        if isinstance(image, dict):
            dict_url = image.get("url")
            if not isinstance(dict_url, str):
                nested_image = image.get("image_url")
                if isinstance(nested_image, dict):
                    dict_url = nested_image.get("url")
            if isinstance(dict_url, str):
                return self._to_image_url(dict_url)

        if isinstance(image, str):
            if image.startswith(("http://", "https://", "data:image")):
                return image
            path = Path(image)
            if path.exists() and path.is_file():
                image = path.read_bytes()
            else:
                return None

        if isinstance(image, Path):
            image = image.read_bytes()

        if isinstance(image, BufferedReader):
            image = image.read()

        content_type = ImageEditRequestUtils.get_image_content_type(image)
        encoded = base64.b64encode(image).decode("utf-8")
        return f"data:{content_type};base64,{encoded}"

    def _encode_inline_video_id(self, video_url: str) -> str:
        encoded = base64.urlsafe_b64encode(video_url.encode("utf-8")).decode("ascii")
        return f"{self._INLINE_VIDEO_ID_PREFIX}{encoded.rstrip('=')}"

    def _decode_inline_video_id(self, video_id: str) -> Optional[str]:
        if not video_id.startswith(self._INLINE_VIDEO_ID_PREFIX):
            return None
        raw = video_id[len(self._INLINE_VIDEO_ID_PREFIX) :]
        if not raw:
            return None
        padding = "=" * (-len(raw) % 4)
        try:
            decoded = base64.urlsafe_b64decode(raw + padding).decode("utf-8")
        except Exception:
            return None
        return decoded or None

    def _safe_to_str(self, request_data: Optional[Dict[str, Any]], key: str) -> Optional[str]:
        if not request_data:
            return None
        value = request_data.get(key)
        if value is None:
            return None
        return str(value)

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if raw_response.status_code < 400:
            return

        error_message = raw_response.text
        try:
            payload = raw_response.json()
            if isinstance(payload, dict):
                if isinstance(payload.get("error"), dict):
                    error_message = payload["error"].get("message", error_message)
                else:
                    error_message = (
                        payload.get("detail")
                        or payload.get("message")
                        or payload.get("error")
                        or error_message
                    )
        except Exception:
            pass
        raise self.get_error_class(
            error_message=str(error_message),
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

    def _validate_veo_model(self, model: str) -> None:
        model_token = str(model or "").strip().lower()
        if not model_token.startswith("veo_"):
            raise ValueError(
                "flow2api videos provider only supports veo_* video models."
            )
