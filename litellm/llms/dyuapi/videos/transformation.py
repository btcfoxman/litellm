from io import BufferedReader
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
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


class DyuapiVideoConfig(BaseVideoConfig):
    """
    Configuration class for Dyuapi Sora2 video generation.

    Supports:
    - JSON body (text-to-video or image_url)
    - multipart/form-data (input_reference file upload)
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
            "style",
            "storyboard",
            "trim",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped_params: Dict[str, Any] = {}

        size_mapping = {
            "portrait": "720x1280",
            "landscape": "1280x720",
            "square": "1024x1024"
        }

        raw_size = video_create_optional_params.get("size")
        raw_aspect = video_create_optional_params.get("aspect_ratio")
        final_size = None
        # 如果 size 是 portrait/landscape，先查表
        if isinstance(raw_size, str) and raw_size.lower() in size_mapping:
            final_size = size_mapping[raw_size.lower()]
        # 如果没有 size 但有 aspect_ratio，也可以做转换
        elif raw_aspect == "9:16":
            final_size = "720x1280"
        elif raw_aspect == "16:9":
            final_size = "1280x720"
        # 否则尝试直接透传原始 size
        elif raw_size:
            final_size = str(raw_size)
        if final_size:
            mapped_params["size"] = final_size

        # Supported dyuapi params
        for key in ("input_reference", "image_url", "style", "storyboard", "trim"):
            if key in video_create_optional_params:
                value = video_create_optional_params[key]
                if key == "seconds":
                    mapped_params["seconds"] = str(value)
                # 特别处理 trim 字段，解决 seconds 必须为字符串的问题
                elif key == "trim" and isinstance(value, dict):
                    if "seconds" in value and value["seconds"] is not None:
                        value["seconds"] = str(value["seconds"])
                    mapped_params[key] = value
                elif key not in mapped_params:
                    mapped_params[key] = value

        # Pass-through for any additional provider-specific params when drop_params is False
        if not drop_params:
            supported_openai_params = set(self.get_supported_openai_params(model))
            for key, value in video_create_optional_params.items():
                if key in mapped_params or key in supported_openai_params or key in ("size", "aspect_ratio"):
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
            or get_secret_str("DYUAPI_API_KEY")
            or get_secret_str("DYUAPI_KEY")
        )

        if api_key is None:
            raise ValueError(
                "Dyuapi API key is required. Set DYUAPI_API_KEY env var or pass api_key."
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
            api_base = "https://api.dyuapi.com/v1"
        return api_base.rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        request_data: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }

        files_list: List[Tuple[str, Any]] = []
        input_reference = video_create_optional_request_params.pop(
            "input_reference", None
        )

        # If a file input is provided, use multipart/form-data
        if input_reference is not None:
            self._add_image_to_files(
                files_list=files_list,
                image=input_reference,
                field_name="input_reference",
            )
        else:
            # Support image_url in JSON body when not using file upload
            image_url = video_create_optional_request_params.get("image_url")
            if image_url:
                request_data["image_url"] = image_url

        # Merge remaining optional params (style/storyboard/trim, etc.)
        for key, value in video_create_optional_request_params.items():
            if key in {"image_url"} and input_reference is not None:
                continue
            if value is not None:
                request_data[key] = value

        full_api_base = f"{api_base}/videos"

        return request_data, files_list, full_api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        if response_data.get("status") == "failed":
            raise self.get_error_class(
                error_message=str(
                    response_data.get("error_message")
                    or response_data.get("error")
                    or "video generation failed"
                ),
                status_code=400,
                headers=raw_response.headers,
            )

        video_data: Dict[str, Any] = {
            "id": response_data.get("id", ""),
            "object": response_data.get("object", "video"),
            "status": response_data.get("status", "queued"),
            "created_at": response_data.get("created_at"),
            "progress": response_data.get("progress"),
            "size": response_data.get("size"),
            "model": response_data.get("model", model),
            "seconds": self._infer_seconds(response_data.get("seconds")),
        }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )

        usage_data = {}
        if video_obj and getattr(video_obj, "seconds", None):
            try:
                usage_data["duration_seconds"] = float(video_obj.seconds)
            except (ValueError, TypeError):
                pass
        video_obj.usage = usage_data

        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/videos/{original_video_id}"
        params: Dict[str, Any] = {}
        return url, params

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        video_url = response_data.get("video_url")
        if not video_url:
            raise ValueError("video_url not found in response. Video may not be ready.")

        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()

        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        video_url = response_data.get("video_url")
        if not video_url:
            raise ValueError("video_url not found in response. Video may not be ready.")

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.DYUAPI,
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
        raise NotImplementedError("Video remix is not supported by Dyuapi.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by Dyuapi.")

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
        raise NotImplementedError("Video list is not supported by Dyuapi.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video list is not supported by Dyuapi.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video delete is not supported by Dyuapi.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by Dyuapi.")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/videos/{original_video_id}"
        data: Dict[str, Any] = {}
        return url, data

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()

        video_data: Dict[str, Any] = {
            "id": response_data.get("id", ""),
            "object": response_data.get("object", "video"),
            "status": response_data.get("status", "queued"),
            "created_at": response_data.get("created_at"),
            "completed_at": response_data.get("completed_at"),
            "progress": response_data.get("progress"),
            "size": response_data.get("size"),
            "model": response_data.get("model"),
            "seconds": self._infer_seconds(response_data.get("seconds")),
            "video_url": response_data.get("video_url"),
        }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        video_url = response_data.get("video_url")
        if video_url:
            video_obj._hidden_params["video_url"] = video_url

        if response_data.get("status") == "failed":
            raise self.get_error_class(
                error_message=str(
                    response_data.get("error_message")
                    or response_data.get("error")
                    or "video generation failed"
                ),
                status_code=400,
                headers=raw_response.headers,
            )

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, None
            )

        usage_data = {}
        if video_obj and getattr(video_obj, "seconds", None):
            try:
                usage_data["duration_seconds"] = float(video_obj.seconds)
            except (ValueError, TypeError):
                pass
        video_obj.usage = usage_data

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

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if raw_response.status_code < 400:
            return
        error_message = raw_response.text
        try:
            response_json = raw_response.json()
            error_message = (
                response_json.get("detail")
                or response_json.get("msg")
                or response_json.get("message")
                or response_json.get("error")
                or response_json.get("error_message")
                or error_message
            )
        except Exception:
            pass
        raise self.get_error_class(
            error_message=str(error_message),
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

    def _add_image_to_files(
        self,
        files_list: List[Tuple[str, Any]],
        image: Any,
        field_name: str,
    ) -> None:
        image_content_type = ImageEditRequestUtils.get_image_content_type(image)

        if isinstance(image, BufferedReader):
            files_list.append((field_name, (image.name, image, image_content_type)))
        else:
            files_list.append((field_name, ("input_reference.png", image, image_content_type)))

    def _infer_seconds(self, model: str) -> str:
        if "25s" in model:
            return "25"
        if "15s" in model:
            return "15"
        return "10"