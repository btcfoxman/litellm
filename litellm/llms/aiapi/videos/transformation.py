from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

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


class AIAPIVideoConfig(BaseVideoConfig):
    """
    AI Orchestration Sora task API.

    API Base: http://orchestration-service:8000
    Endpoints:
      - POST /api/v1/sora/video/tasks
      - GET  /api/v1/sora/video/tasks/{task_id}
    """

    _SUPPORTED_DURATIONS = {10, 15}

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "image_url",
            "seconds",
            "size",
            "user",
            "asset_id",
            "user_id",
            "business_system",
            "callback_url",
            "callBackUrl",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped: Dict[str, Any] = {}
        for key in self.get_supported_openai_params(model):
            if key in video_create_optional_params:
                mapped[key] = video_create_optional_params[key]
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

        api_key = api_key or litellm.api_key or get_secret_str("AIAPI_API_KEY")
        if api_key is None:
            raise ValueError(
                "AIAPI API key is required. Set AIAPI_API_KEY env var or pass api_key."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "X-API-Key": str(api_key),
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        return (api_base or "http://orchestration-service:8000").rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, list, str]:
        duration = self._infer_duration(
            model=model,
            seconds=video_create_optional_request_params.get("seconds"),
        )

        request_data: Dict[str, Any] = {
            "prompt": prompt,
            "duration": duration,
            "user_id": str(
                video_create_optional_request_params.get("user_id")
                or video_create_optional_request_params.get("user")
                or "litellm"
            ),
            "business_system": str(
                video_create_optional_request_params.get("business_system") or "litellm"
            ),
        }

        reference_image_url = self._extract_reference_image_url(
            video_create_optional_request_params
        )
        if reference_image_url is not None:
            request_data["reference_image_url"] = reference_image_url

        asset_id = video_create_optional_request_params.get("asset_id")
        if asset_id:
            request_data["asset_id"] = str(asset_id)

        callback_url = video_create_optional_request_params.get(
            "callback_url"
        ) or video_create_optional_request_params.get("callBackUrl")
        if callback_url:
            request_data["callback_url"] = str(callback_url)

        return request_data, [], f"{api_base}/api/v1/sora/video/tasks"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        response_data = self._normalize_response_json(raw_response.json())
        status = self._normalize_status(response_data.get("status"))

        if status in {"failed", "cancelled"}:
            raise self.get_error_class(
                error_message=self._extract_error_message(response_data),
                status_code=400,
                headers=raw_response.headers,
            )

        task_id = str(response_data.get("task_id") or response_data.get("id") or "")
        video_data: Dict[str, Any] = {
            "id": task_id,
            "object": "video",
            "status": status,
            "created_at": response_data.get("created_at"),
            "progress": self._parse_progress(response_data.get("progress"), status),
            "model": model,
            "seconds": self._safe_seconds(request_data),
            "video_url": response_data.get("result_url"),
        }
        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if video_data.get("video_url"):
            video_obj._hidden_params["video_url"] = str(video_data["video_url"])

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
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        return f"{api_base}/api/v1/sora/video/tasks/{quote(original_video_id, safe='')}", {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        response_data = self._normalize_response_json(raw_response.json())
        video_url = response_data.get("result_url")
        if not video_url:
            raise ValueError("result_url not found in response. Video may not be ready.")

        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(str(video_url))
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        response_data = self._normalize_response_json(raw_response.json())
        video_url = response_data.get("result_url")
        if not video_url:
            raise ValueError("result_url not found in response. Video may not be ready.")

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.AIAPI,
        )
        video_response = await async_httpx_client.get(str(video_url))
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
        raise NotImplementedError("Video remix is not supported by AIAPI.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by AIAPI.")

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
        raise NotImplementedError("Video list is not supported by AIAPI.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video list is not supported by AIAPI.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video delete is not supported by AIAPI.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by AIAPI.")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        return f"{api_base}/api/v1/sora/video/tasks/{quote(original_video_id, safe='')}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        response_data = self._normalize_response_json(raw_response.json())
        status = self._normalize_status(response_data.get("status"))
        progress = self._parse_progress(response_data.get("progress"), status)
        video_url = response_data.get("result_url")

        if status in {"failed", "cancelled"}:
            raise self.get_error_class(
                error_message=self._extract_error_message(response_data),
                status_code=400,
                headers=raw_response.headers,
            )

        video_data: Dict[str, Any] = {
            "id": str(response_data.get("task_id") or response_data.get("id") or ""),
            "object": "video",
            "status": status,
            "created_at": response_data.get("created_at"),
            "completed_at": response_data.get("completed_at"),
            "progress": progress,
            "model": response_data.get("model"),
            "seconds": (
                str(response_data.get("duration"))
                if response_data.get("duration") is not None
                else None
            ),
            "video_url": str(video_url) if video_url else None,
        }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if video_data.get("video_url"):
            video_obj._hidden_params["video_url"] = str(video_data["video_url"])

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
            try:
                payload = raw_response.json()
            except Exception:
                return
            if self._is_business_error(payload):
                raise self.get_error_class(
                    error_message=self._extract_error_message(payload),
                    status_code=400,
                    headers=raw_response.headers,
                )
            return

        error_message = raw_response.text
        try:
            response_json = raw_response.json()
            if isinstance(response_json, dict):
                error_message = self._extract_error_message(response_json)
        except Exception:
            pass
        raise self.get_error_class(
            error_message=str(error_message),
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

    def _is_business_error(self, response_json: Any) -> bool:
        if not isinstance(response_json, dict):
            return False
        code = response_json.get("code")
        if code is not None:
            try:
                if int(code) != 200:
                    return True
            except (TypeError, ValueError):
                if str(code).strip().lower() not in {"200", "ok", "success"}:
                    return True
        return False

    def _extract_error_message(self, response_json: Dict[str, Any]) -> str:
        data = response_json.get("data")
        if isinstance(data, dict):
            nested_error = data.get("error_message") or data.get("error")
            if nested_error:
                return str(nested_error)
        return str(
            response_json.get("detail")
            or response_json.get("msg")
            or response_json.get("message")
            or response_json.get("error")
            or response_json
        )

    def _normalize_response_json(self, response_json: Any) -> Dict[str, Any]:
        if not isinstance(response_json, dict):
            return {}
        data = response_json.get("data")
        if isinstance(data, dict):
            return data
        return response_json

    def _safe_seconds(self, request_data: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(request_data, dict):
            return None
        value = request_data.get("duration")
        if value is None:
            return None
        return str(value)

    def _normalize_status(self, status: Any) -> str:
        value = str(status or "").strip().lower()
        if value in {"completed", "success", "succeeded", "done", "finished"}:
            return "completed"
        if value in {"failed", "error", "fail"}:
            return "failed"
        if value in {"cancelled", "canceled"}:
            return "cancelled"
        if value in {"running", "in_progress", "processing"}:
            return "in_progress"
        if value in {"pending", "queued", "submitted"}:
            return "queued"
        return "queued"

    def _parse_progress(self, progress: Any, status: str) -> Optional[int]:
        if isinstance(progress, int):
            return max(0, min(100, progress))
        if isinstance(progress, str):
            parsed = progress.strip()
            if parsed.endswith("%"):
                parsed = parsed[:-1].strip()
            try:
                return max(0, min(100, int(float(parsed))))
            except (TypeError, ValueError):
                pass
        if status == "completed":
            return 100
        if status in {"queued", "in_progress"}:
            return 0
        return None

    def _extract_reference_image_url(
        self, video_create_optional_request_params: Dict[str, Any]
    ) -> Optional[str]:
        candidate = video_create_optional_request_params.get("input_reference")
        if candidate is None:
            candidate = video_create_optional_request_params.get("image_url")

        if isinstance(candidate, list):
            for item in candidate:
                url = self._extract_url_from_item(item)
                if url is not None:
                    return url
            return None
        return self._extract_url_from_item(candidate)

    def _extract_url_from_item(self, item: Any) -> Optional[str]:
        if item is None:
            return None

        if isinstance(item, dict):
            dict_url = item.get("url")
            if not isinstance(dict_url, str):
                nested = item.get("image_url")
                if isinstance(nested, dict):
                    dict_url = nested.get("url")
            if isinstance(dict_url, str):
                return self._extract_url_from_item(dict_url)
            return None

        if isinstance(item, str):
            value = item.strip()
            if value.startswith("http://") or value.startswith("https://"):
                return value
            if value.startswith("data:image"):
                raise ValueError(
                    "AIAPI only supports HTTP(S) image URLs for reference_image_url."
                )
            path = Path(value)
            if path.exists() and path.is_file():
                raise ValueError(
                    "AIAPI only supports HTTP(S) image URLs, local files are not supported."
                )
        return None

    def _infer_duration(self, model: str, seconds: Any) -> int:
        if seconds is not None:
            try:
                requested = int(float(str(seconds)))
            except (TypeError, ValueError):
                requested = None
            if requested in self._SUPPORTED_DURATIONS:
                return requested

        model_token = model.lower()
        if "/" in model_token:
            model_token = model_token.split("/", 1)[1]
        if "-10s" in model_token:
            return 10
        if "-15s" in model_token:
            return 15
        return 15
