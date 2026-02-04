from io import BufferedReader, BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import base64
import httpx

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider, extract_original_video_id

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any


class WTAPIVideoConfig(BaseVideoConfig):
    """
    Configuration class for WTAPI Sora2 video generation.

    API Base: https://api.whatai.cc
    Endpoints:
      - POST /v2/videos/generations
      - GET  /v2/videos/generations/{task_id}
    """

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "image_url",
            "images",
            "aspect_ratio",
            "duration",
            "hd",
            "notify_hook",
            "watermark",
            "private",
            "user",
            "extra_headers",
            "extra_body",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped_params: Dict[str, Any] = {}
        for key in (
            "input_reference",
            "image_url",
            "images",
            "aspect_ratio",
            "duration",
            "hd",
            "notify_hook",
            "watermark",
            "private",
        ):
            if key in video_create_optional_params:
                mapped_params[key] = video_create_optional_params[key]

        if not drop_params:
            supported_openai_params = set(self.get_supported_openai_params(model))
            for key, value in video_create_optional_params.items():
                if key in mapped_params or key in supported_openai_params:
                    continue
                mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("WTAPI_API_KEY")
            or get_secret_str("WHATI_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "WTAPI API key is required. Set WTAPI_API_KEY env var or pass api_key."
            )

        headers.update({"Authorization": f"Bearer {api_key}"})
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or "https://api.whatai.cc"
        return api_base.rstrip("/")

    def _infer_aspect_ratio_from_model(self, model: str) -> str:
        if "portrait" in model:
            return "9:16"
        return "16:9"

    def _infer_duration_from_model(self, model: str) -> str:
        if "25s" in model:
            return "25"
        if "15s" in model:
            return "15"
        return "10"

    def _infer_model_variant(self, model: str) -> str:
        if "pro" in model:
            return "sora-2-pro"
        return "sora-2"

    def _infer_hd(self, model: str) -> bool:
        return "hd" in model

    def _image_to_base64(self, image: Any) -> str:
        if isinstance(image, BufferedReader):
            data = image.read()
        elif isinstance(image, BytesIO):
            data = image.getvalue()
        elif isinstance(image, bytes):
            data = image
        elif isinstance(image, Path):
            data = image.read_bytes()
        elif isinstance(image, str):
            return image
        else:
            if hasattr(image, "read"):
                data = image.read()
            else:
                raise ValueError("Unsupported image type for WTAPI")

        return base64.b64encode(data).decode("ascii")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, list, str]:
        request_data: Dict[str, Any] = {
            "prompt": prompt,
            "model": self._infer_model_variant(model),
            "aspect_ratio": self._infer_aspect_ratio_from_model(model),
            "duration": self._infer_duration_from_model(model),
            "hd": self._infer_hd(model),
        }

        # Allow overrides
        for key in ("model", "aspect_ratio", "duration", "hd", "notify_hook", "watermark", "private"):
            value = video_create_optional_request_params.get(key)
            if value is not None:
                request_data[key] = value

        # images (required)
        images: List[str] = []
        images_value = video_create_optional_request_params.get("images")
        if isinstance(images_value, list):
            images.extend([str(v) for v in images_value if v])
        elif isinstance(images_value, str) and images_value:
            images.append(images_value)

        image_url = video_create_optional_request_params.get("image_url")
        if isinstance(image_url, str) and image_url:
            images.append(image_url)

        input_reference = video_create_optional_request_params.get("input_reference")
        if input_reference is not None:
            if isinstance(input_reference, list):
                for img in input_reference:
                    images.append(self._image_to_base64(img))
            else:
                images.append(self._image_to_base64(input_reference))

        if not images:
            raise ValueError("WTAPI requires at least one image in `images`.")

        request_data["images"] = images

        full_api_base = f"{api_base}/v2/videos/generations"
        return request_data, [], full_api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        task_id = response_data.get("task_id", "")

        video_data: Dict[str, Any] = {
            "id": task_id,
            "object": "video",
            "status": "queued",
            "progress": 0,
            "model": model,
        }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, None
            )
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/v2/videos/generations/{original_video_id}"
        return url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        video_url = (response_data.get("data") or {}).get("output")
        if not video_url:
            raise ValueError("output not found in response. Video may not be ready.")

        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        video_url = (response_data.get("data") or {}).get("output")
        if not video_url:
            raise ValueError("output not found in response. Video may not be ready.")

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.WTAPI,
        )
        video_response = await async_httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video remix is not supported by WTAPI.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by WTAPI.")

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
        raise NotImplementedError("Video list is not supported by WTAPI.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video list is not supported by WTAPI.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video delete is not supported by WTAPI.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by WTAPI.")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/v2/videos/generations/{original_video_id}"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        status_value = response_data.get("status")

        status = "queued"
        if status_value == "NOT_START":
            status = "queued"
        elif status_value == "IN_PROGRESS":
            status = "in_progress"
        elif status_value == "SUCCESS":
            status = "completed"
        elif status_value == "FAILURE":
            status = "failed"

        video_url = (response_data.get("data") or {}).get("output")
        progress = response_data.get("progress")

        video_data: Dict[str, Any] = {
            "id": response_data.get("task_id", ""),
            "object": "video",
            "status": status,
            "progress": progress,
            "video_url": video_url,
        }

        if status == "failed":
            video_data["error"] = {
                "code": "FAILURE",
                "message": response_data.get("fail_reason"),
            }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]
        if video_url:
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
