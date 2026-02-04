from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode

import httpx

import litellm
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


class S2APIVideoConfig(BaseVideoConfig):
    """
    Configuration class for Sora2API (sora2api.ai) video generation.
    Supports text-to-video and image-to-video (via imageUrls).
    """

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "user",
            "extra_headers",
            "extra_body",
            "image_url",
            "imageUrls",
            "aspectRatio",
            "quality",
            "watermark",
            "callBackUrl",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped_params: Dict[str, Any] = {}

        for key in (
            "image_url",
            "imageUrls",
            "aspectRatio",
            "quality",
            "watermark",
            "callBackUrl",
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
            or get_secret_str("S2API_API_KEY")
            or get_secret_str("SORA2API_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "S2API API key is required. Set S2API_API_KEY env var or pass api_key."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        if api_base is None:
            api_base = "https://api.sora2api.ai"
        api_base = api_base.rstrip("/")
        if api_base.endswith("/v1"):
            api_base = api_base[: -len("/v1")]
        return api_base

    def _infer_aspect_ratio_from_model(self, model: str) -> Optional[str]:
        if "landscape" in model:
            return "landscape"
        if "portrait" in model:
            return "portrait"
        return None

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
        }

        # aspectRatio + quality are required by S2API
        aspect_ratio = video_create_optional_request_params.get("aspectRatio")
        if aspect_ratio is None:
            aspect_ratio = self._infer_aspect_ratio_from_model(model)
        if aspect_ratio is None:
            aspect_ratio = "landscape"

        # quality = video_create_optional_request_params.get("quality") or "standard"
        quality = "hd"

        request_data["aspectRatio"] = aspect_ratio
        request_data["quality"] = quality

        # image inputs
        image_urls: List[str] = []
        image_urls_value = video_create_optional_request_params.get("imageUrls")
        if isinstance(image_urls_value, list):
            image_urls.extend([str(u) for u in image_urls_value if u])
        elif isinstance(image_urls_value, str) and image_urls_value:
            image_urls.append(image_urls_value)

        image_url = video_create_optional_request_params.get("image_url")
        if isinstance(image_url, str) and image_url:
            image_urls.append(image_url)

        input_reference = video_create_optional_request_params.get("input_reference")
        if isinstance(input_reference, str) and input_reference:
            image_urls.append(input_reference)
        elif input_reference is not None:
            raise ValueError("S2API 仅支持图片URL，不支持文件上传。")

        if image_urls:
            request_data["imageUrls"] = image_urls

        # optional params
        for key in ("watermark", "callBackUrl"):
            value = video_create_optional_request_params.get(key)
            if value is not None:
                request_data[key] = value

        full_api_base = f"{api_base}/api/v1/sora2api/generate"

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
        task_id = response_data.get("data", {}).get("taskId", "")

        video_data: Dict[str, Any] = {
            "id": task_id,
            "object": "video",
            "status": "queued",
            "progress": 0,
            "model": model,
        }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            # Only persist model_id when it looks like a provider model (e.g. "s2api/...")
            model_id_for_encode = model if isinstance(model, str) and "/" in model else None
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model_id_for_encode
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
        base_url = f"{api_base}/api/v1/sora2api/record-info"
        query = urlencode({"taskId": original_video_id})
        url = f"{base_url}?{query}"
        return url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        image_url = (
            response_data.get("data", {})
            .get("response", {})
            .get("imageUrl")
        )
        if not image_url:
            raise ValueError("imageUrl not found in response. Video may not be ready.")

        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(image_url)
        video_response.raise_for_status()

        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        image_url = (
            response_data.get("data", {})
            .get("response", {})
            .get("imageUrl")
        )
        if not image_url:
            raise ValueError("imageUrl not found in response. Video may not be ready.")

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.S2API,
        )
        video_response = await async_httpx_client.get(image_url)
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
        raise NotImplementedError("Video remix is not supported by S2API.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by S2API.")

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
        raise NotImplementedError("Video list is not supported by S2API.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video list is not supported by S2API.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video delete is not supported by S2API.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by S2API.")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        base_url = f"{api_base}/api/v1/sora2api/record-info"
        query = urlencode({"taskId": original_video_id})
        url = f"{base_url}?{query}"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        data = response_data.get("data", {}) or {}
        success_flag = data.get("successFlag")

        status = "queued"
        if success_flag == 0:
            status = "in_progress"
        elif success_flag == 1:
            status = "completed"
        elif success_flag in (2, 3):
            status = "failed"

        image_url = (data.get("response") or {}).get("imageUrl")
        video_data: Dict[str, Any] = {
            "id": data.get("taskId", ""),
            "object": "video",
            "status": status,
            "progress": 100 if success_flag == 1 else None,
            "video_url": image_url,
        }

        if success_flag in (2, 3):
            video_data["error"] = {
                "code": data.get("errorCode"),
                "message": data.get("errorMessage") or response_data.get("msg"),
            }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if image_url:
            video_obj._hidden_params["video_url"] = image_url

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
