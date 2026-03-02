from io import BufferedReader
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

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
from litellm.llms.ywapi.videos.sora_transformation import (
    YWAPISoraVideoRequestTransformer,
)
from litellm.llms.ywapi.videos.veo_transformation import (
    YWAPIVeoVideoRequestTransformer,
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


class YWAPIVideoConfig(BaseVideoConfig):
    """
    Yunwu API (ywapi) in unified video mode.

    API Base: https://yunwu.ai
    Endpoints:
      - POST /v1/video/create
      - GET  /v1/video/query?id=<task_id>
    """

    def __init__(self) -> None:
        super().__init__()
        self._sora_request_transformer = YWAPISoraVideoRequestTransformer()
        self._veo_request_transformer = YWAPIVeoVideoRequestTransformer()

    def get_supported_openai_params(self, model: str) -> list:
        provider_model = self._resolve_provider_model(model)
        if self._is_veo_model(provider_model):
            return self._veo_request_transformer.get_supported_openai_params()
        return self._sora_request_transformer.get_supported_openai_params()

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        provider_model = self._resolve_provider_model(
            model, optional_model=video_create_optional_params.get("model")
        )
        if self._is_veo_model(provider_model):
            return self._veo_request_transformer.map_openai_params(
                video_create_optional_params, drop_params=drop_params
            )
        return self._sora_request_transformer.map_openai_params(
            video_create_optional_params, drop_params=drop_params
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = api_key or litellm.api_key or get_secret_str("YWAPI_API_KEY")

        if api_key is None:
            raise ValueError(
                "YWAPI API key is required. Set YWAPI_API_KEY env var or pass api_key."
            )

        headers.update({"Authorization": f"Bearer {api_key}"})
        headers.setdefault("Accept", "application/json")
        headers.setdefault("Content-Type", "application/json")
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        return (api_base or "https://yunwu.ai").rstrip("/")

    def _get_model_token(self, model: str) -> str:
        if not model:
            return ""
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    def _is_veo_model(self, model: str) -> bool:
        return self._veo_request_transformer.is_veo_provider_model(model)

    def _resolve_provider_model(
        self, model: str, optional_model: Optional[Any] = None
    ) -> str:
        if optional_model:
            return str(optional_model)
        model_token = self._get_model_token(model).strip().lower()
        if self._is_veo_model(model_token):
            return self._veo_request_transformer.infer_provider_model(model)
        return self._sora_request_transformer.infer_provider_model(model)

    def _image_to_string(self, image: Any) -> Optional[str]:
        if image is None:
            return None

        if isinstance(image, str):
            path = Path(image)
            if path.exists() and path.is_file():
                image = path.read_bytes()
            else:
                return image

        if isinstance(image, Path):
            if image.exists() and image.is_file():
                image = image.read_bytes()
            else:
                return None

        image_content_type = ImageEditRequestUtils.get_image_content_type(image)

        data: Optional[bytes] = None
        if isinstance(image, bytes):
            data = image
        elif isinstance(image, BufferedReader):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
        elif hasattr(image, "read"):
            current_pos = image.tell() if hasattr(image, "tell") else None
            if hasattr(image, "seek"):
                image.seek(0)
            data = image.read()
            if hasattr(image, "seek") and current_pos is not None:
                image.seek(current_pos)
            if isinstance(data, str):
                data = data.encode("utf-8")

        if not data:
            return None

        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{image_content_type};base64,{encoded}"

    def _collect_images(self, video_create_optional_request_params: Dict) -> List[str]:
        images: List[str] = []

        raw_images = video_create_optional_request_params.get("images")
        if isinstance(raw_images, list):
            for item in raw_images:
                image_str = self._image_to_string(item)
                if image_str:
                    images.append(image_str)
        elif raw_images is not None:
            image_str = self._image_to_string(raw_images)
            if image_str:
                images.append(image_str)

        image_url = video_create_optional_request_params.get("image_url")
        if image_url is not None:
            image_str = self._image_to_string(image_url)
            if image_str:
                images.append(image_str)

        input_reference = video_create_optional_request_params.get("input_reference")
        if input_reference is not None:
            if isinstance(input_reference, list):
                for item in input_reference:
                    image_str = self._image_to_string(item)
                    if image_str:
                        images.append(image_str)
            else:
                image_str = self._image_to_string(input_reference)
                if image_str:
                    images.append(image_str)

        deduped: List[str] = []
        seen = set()
        for img in images:
            if img in seen:
                continue
            seen.add(img)
            deduped.append(img)

        return deduped

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

    def _normalize_status(self, provider_status: Any) -> str:
        value = str(provider_status or "").strip().lower()
        if value in {"success", "succeeded", "completed", "done", "finished"}:
            return "completed"
        if value in {"failed", "fail", "error", "cancelled", "canceled"}:
            return "failed"
        if value in {
            "running",
            "processing",
            "in_progress",
            "pending",
            "working",
            "queued",
            "submitted",
        }:
            if value in {"queued", "pending", "submitted"}:
                return "queued"
            return "in_progress"
        return "queued"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, list, str]:
        images = self._collect_images(video_create_optional_request_params)
        provider_model = self._resolve_provider_model(
            model, optional_model=video_create_optional_request_params.get("model")
        )
        if self._is_veo_model(provider_model):
            request_data = self._veo_request_transformer.build_video_create_request(
                prompt=prompt,
                provider_model=provider_model,
                video_create_optional_request_params=video_create_optional_request_params,
                images=images,
            )
        else:
            request_data = self._sora_request_transformer.build_video_create_request(
                model=model,
                prompt=prompt,
                provider_model=provider_model,
                video_create_optional_request_params=video_create_optional_request_params,
                images=images,
            )

        full_api_base = f"{api_base}/v1/video/create"
        return request_data, [], full_api_base

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
        if status == "failed":
            raise self.get_error_class(
                error_message=self._extract_error_message(response_data),
                status_code=400,
                headers=raw_response.headers,
            )

        duration_seconds = None
        if request_data is not None:
            duration_seconds = request_data.get("duration")

        video_data: Dict[str, Any] = {
            "id": str(response_data.get("id", "")),
            "object": "video",
            "status": status,
            "created_at": response_data.get("status_update_time")
            or response_data.get("created_at"),
            "progress": self._parse_progress(response_data.get("progress"), status),
            "model": model,
            "seconds": str(duration_seconds) if duration_seconds is not None else None,
            "size": (request_data or {}).get("size") if request_data else None,
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
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/v1/video/query?id={quote(original_video_id, safe='')}"
        return url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        response_data = self._normalize_response_json(raw_response.json())
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
        response_data = self._normalize_response_json(raw_response.json())
        video_url = response_data.get("video_url")
        if not video_url:
            raise ValueError("video_url not found in response. Video may not be ready.")

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.YWAPI,
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
        raise NotImplementedError("Video remix is not supported by YWAPI.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by YWAPI.")

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
        raise NotImplementedError("Video list is not supported by YWAPI.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video list is not supported by YWAPI.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video delete is not supported by YWAPI.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by YWAPI.")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/v1/video/query?id={quote(original_video_id, safe='')}"
        data: Dict[str, Any] = {}
        return url, data

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

        video_data: Dict[str, Any] = {
            "id": str(response_data.get("id", "")),
            "object": "video",
            "status": status,
            "created_at": response_data.get("status_update_time")
            or response_data.get("created_at"),
            "completed_at": response_data.get("completed_at"),
            "progress": progress,
            "model": response_data.get("model"),
            "seconds": (
                str(response_data.get("duration"))
                if response_data.get("duration") is not None
                else None
            ),
            "video_url": response_data.get("video_url"),
        }

        if status == "failed":
            raise self.get_error_class(
                error_message=self._extract_error_message(response_data),
                status_code=400,
                headers=raw_response.headers,
            )

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if video_data.get("video_url"):
            video_obj._hidden_params["video_url"] = video_data["video_url"]

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

            normalized = self._normalize_response_json(payload)
            if not isinstance(normalized, dict):
                return

            if self._normalize_status(normalized.get("status")) == "failed":
                raise self.get_error_class(
                    error_message=self._extract_error_message(normalized),
                    status_code=400,
                    headers=raw_response.headers,
                )

            if normalized.get("success") is False:
                raise self.get_error_class(
                    error_message=self._extract_error_message(normalized),
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

    def _extract_error_message(self, response_json: Dict[str, Any]) -> str:
        return str(
            response_json.get("detail")
            or response_json.get("msg")
            or response_json.get("message")
            or response_json.get("error")
            or response_json.get("error_message")
            or response_json
        )

    def _normalize_response_json(self, response_json: Any) -> Dict[str, Any]:
        if not isinstance(response_json, dict):
            return {}
        data = response_json.get("data")
        if isinstance(data, dict):
            return data
        return response_json
