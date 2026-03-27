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

        if "seconds" in video_create_optional_params:
            mapped_params["seconds"] = str(video_create_optional_params["seconds"])

        # Supported dyuapi params
        for key in ("input_reference", "image_url", "style", "storyboard", "trim"):
            if key in video_create_optional_params:
                value = video_create_optional_params[key]
                if key == "trim" and isinstance(value, dict) and "seconds" in value:
                    value["seconds"] = str(value["seconds"])
                if key == "extra_body" and isinstance(value, dict):
                    if "seconds" in value:
                        value["seconds"] = str(value["seconds"])
                    if "size" in value and value["size"] == "portrait":
                        value["size"] = "720x1280"
                mapped_params[key] = value

        if not drop_params:
            supported_openai_params = set(self.get_supported_openai_params(model))
            for key, value in video_create_optional_params.items():
                if key in mapped_params or key in supported_openai_params or key in ("size", "aspect_ratio"):
                    continue
                if key == "seconds":
                    mapped_params[key] = str(value)
                else:
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
        status = self._normalize_status(response_data.get("status"))
        if status == "failed":
            raise self.get_error_class(
                error_message=self._extract_error_message(response_data),
                status_code=400,
                headers=raw_response.headers,
            )

        video_data: Dict[str, Any] = {
            "id": response_data.get("id", ""),
            "object": response_data.get("object", "video"),
            "status": status,
            "created_at": response_data.get("created_at"),
            "progress": self._parse_progress(response_data.get("progress"), status),
            "size": response_data.get("size"),
            "model": response_data.get("model", model),
            "seconds": self._infer_seconds(
                response_data.get("seconds"), response_data.get("model", model)
            ),
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
        status = self._normalize_status(response_data.get("status"))
        if status == "failed":
            raise self.get_error_class(
                error_message=self._extract_error_message(response_data),
                status_code=400,
                headers=raw_response.headers,
            )
        video_url = self._extract_video_url(response_data)
        if not video_url:
            raise self.get_error_class(
                error_message=f"Dyuapi video is not ready yet. status={status}",
                status_code=409,
                headers=raw_response.headers,
            )

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
        status = self._normalize_status(response_data.get("status"))
        if status == "failed":
            raise self.get_error_class(
                error_message=self._extract_error_message(response_data),
                status_code=400,
                headers=raw_response.headers,
            )
        video_url = self._extract_video_url(response_data)
        if not video_url:
            raise self.get_error_class(
                error_message=f"Dyuapi video is not ready yet. status={status}",
                status_code=409,
                headers=raw_response.headers,
            )

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
        status = self._normalize_status(response_data.get("status"))
        progress = self._parse_progress(response_data.get("progress"), status)
        video_url = self._extract_video_url(response_data)

        video_data: Dict[str, Any] = {
            "id": response_data.get("id", ""),
            "object": response_data.get("object", "video"),
            "status": status,
            "created_at": response_data.get("created_at"),
            "completed_at": response_data.get("completed_at"),
            "progress": progress,
            "size": response_data.get("size"),
            "model": response_data.get("model"),
            "seconds": self._infer_seconds(
                response_data.get("seconds"), response_data.get("model")
            ),
            "video_url": video_url,
        }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if video_url:
            video_obj._hidden_params["video_url"] = video_url

        if status == "failed":
            raise self.get_error_class(
                error_message=self._extract_error_message(response_data),
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
        response_json: Dict[str, Any] = {}
        error_message = raw_response.text
        try:
            parsed_json = raw_response.json()
            if isinstance(parsed_json, dict):
                response_json = parsed_json
        except Exception:
            response_json = {}

        if raw_response.status_code < 400:
            status = self._normalize_status(response_json.get("status"))
            if status == "failed":
                self.get_error_class(
                    error_message=self._extract_error_message(response_json),
                    status_code=400,
                    headers=raw_response.headers,
                )
            return

        if response_json:
            error_message = self._extract_error_message(
                response_json, fallback=str(error_message)
            )
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

    def _extract_error_message(
        self, response_data: Dict[str, Any], fallback: str = "video generation failed"
    ) -> str:
        error_obj = response_data.get("error")
        if isinstance(error_obj, dict):
            message = error_obj.get("message") or error_obj.get("msg") or error_obj.get(
                "detail"
            )
            if isinstance(message, str) and message.strip():
                return message.strip()
        if isinstance(error_obj, str) and error_obj.strip():
            return error_obj.strip()

        for key in ("error_message", "fail_reason", "message", "msg", "detail"):
            value = response_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return fallback

    def _extract_video_url(self, response_data: Dict[str, Any]) -> Optional[str]:
        for key in ("video_url", "url", "az_url"):
            value = response_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _normalize_status(self, status_value: Any) -> str:
        status = str(status_value or "").strip().lower()
        if status in {"failed", "error"}:
            return "failed"
        if status in {"completed", "succeeded", "success", "done"}:
            return "completed"
        if status in {"in_progress", "processing", "running"}:
            return "in_progress"
        return "queued"

    def _parse_progress(self, progress_value: Any, status: str) -> Optional[int]:
        if progress_value is None:
            if status == "completed":
                return 100
            if status == "queued":
                return 0
            return None
        if isinstance(progress_value, int):
            return progress_value
        if isinstance(progress_value, float):
            return int(progress_value)
        if isinstance(progress_value, str):
            value = progress_value.strip()
            if value.endswith("%"):
                value = value[:-1].strip()
            try:
                return int(float(value))
            except ValueError:
                return None
        return None

    def _infer_seconds(self, seconds_value: Any, model: Any) -> str:
        if seconds_value is not None:
            seconds_text = str(seconds_value).strip()
            if seconds_text:
                if seconds_text.endswith("s"):
                    seconds_text = seconds_text[:-1].strip()
                return seconds_text

        model_text = str(model or "")
        if "25s" in model_text:
            return "25"
        if "15s" in model_text:
            return "15"
        return "10"
