import base64
from io import BufferedReader
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

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


class BDWAPIVideoConfig(BaseVideoConfig):
    """
    Configuration class for BDWAPI Sora2 video generation.

    API Base: https://api.hellobabygo.com
    Endpoints:
      - POST /v1/videos
      - GET  /v1/videos/{id}
      - GET  /v1/videos/{id}/content
    """

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "image_url",
            "seconds",
            "duration",
            "size",
            "resolution",
            "aspect_ratio",
            "quality",
            "n",
            "user",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped_params: Dict[str, Any] = {}
        for key in self.get_supported_openai_params(model):
            if key in video_create_optional_params:
                mapped_params[key] = video_create_optional_params[key]
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
            or get_secret_str("BDWAPI_API_KEY")
            or get_secret_str("HELLOBABYGO_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "BDWAPI API key is required. Set BDWAPI_API_KEY env var or pass api_key."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        resolved_api_base = (api_base or "https://api.hellobabygo.com").rstrip("/")
        if resolved_api_base.endswith("/v1"):
            return resolved_api_base
        return f"{resolved_api_base}/v1"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        provider_model = self._resolve_provider_model(
            model=model, optional_model=video_create_optional_request_params.get("model")
        )
        seconds = self._resolve_seconds(model, video_create_optional_request_params)
        size = self._resolve_size(model, video_create_optional_request_params)
        n = self._resolve_n(video_create_optional_request_params)

        request_data: Dict[str, Any] = {
            "model": provider_model,
            "prompt": prompt,
            "seconds": str(seconds),
            "size": size,
            "n": str(n),
        }

        reference = self._extract_input_reference(video_create_optional_request_params)
        files: List[Tuple[str, Any]] = []
        if reference is not None:
            part = self._build_reference_multipart(reference)
            files.append(("input_reference", part))

        return request_data, files, f"{api_base}/videos"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        response_data = self._normalize_response_json(self._safe_json(raw_response))
        status = self._normalize_status(response_data.get("status"))

        if status in {"failed", "cancelled"}:
            raise self.get_error_class(
                error_message=self._extract_error_message(response_data),
                status_code=400,
                headers=raw_response.headers,
            )

        video_data: Dict[str, Any] = {
            "id": str(
                response_data.get("id")
                or response_data.get("video_id")
                or response_data.get("task_id")
                or ""
            ),
            "object": "video",
            "status": status,
            "created_at": response_data.get("created_at"),
            "progress": self._parse_progress(response_data.get("progress"), status),
            "model": response_data.get("model", model),
            "seconds": self._safe_seconds(request_data, response_data),
            "size": response_data.get("size"),
            "video_url": self._extract_video_url(response_data),
        }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if video_data.get("video_url"):
            video_obj._hidden_params["video_url"] = str(video_data["video_url"])

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )

        usage_data = {}
        if getattr(video_obj, "seconds", None):
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
        return f"{api_base}/videos/{original_video_id}/content", {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        content_type = raw_response.headers.get("content-type", "").lower()
        if (
            content_type.startswith("video/")
            or "application/octet-stream" in content_type
            or "application/mp4" in content_type
        ):
            return raw_response.content

        response_data = self._normalize_response_json(self._safe_json(raw_response))
        video_url = self._extract_video_url(response_data)
        if not video_url:
            raise ValueError("No downloadable video payload returned by BDWAPI.")

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
        content_type = raw_response.headers.get("content-type", "").lower()
        if (
            content_type.startswith("video/")
            or "application/octet-stream" in content_type
            or "application/mp4" in content_type
        ):
            return raw_response.content

        response_data = self._normalize_response_json(self._safe_json(raw_response))
        video_url = self._extract_video_url(response_data)
        if not video_url:
            raise ValueError("No downloadable video payload returned by BDWAPI.")

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.BDWAPI,
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
        raise NotImplementedError("Video remix is not supported by BDWAPI.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by BDWAPI.")

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
        raise NotImplementedError("Video list is not supported by BDWAPI.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video list is not supported by BDWAPI.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video delete is not supported by BDWAPI.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by BDWAPI.")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        return f"{api_base}/videos/{original_video_id}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        response_data = self._normalize_response_json(self._safe_json(raw_response))
        status = self._normalize_status(response_data.get("status"))
        progress = self._parse_progress(response_data.get("progress"), status)
        video_url = self._extract_video_url(response_data)

        if status in {"failed", "cancelled"}:
            raise self.get_error_class(
                error_message=self._extract_error_message(response_data),
                status_code=400,
                headers=raw_response.headers,
            )

        video_data: Dict[str, Any] = {
            "id": str(
                response_data.get("id")
                or response_data.get("video_id")
                or response_data.get("task_id")
                or ""
            ),
            "object": "video",
            "status": status,
            "created_at": response_data.get("created_at"),
            "completed_at": response_data.get("completed_at"),
            "progress": progress,
            "size": response_data.get("size"),
            "model": response_data.get("model"),
            "seconds": self._safe_seconds(None, response_data),
            "video_url": video_url,
        }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if video_url:
            video_obj._hidden_params["video_url"] = video_url

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, None
            )

        usage_data = {}
        if getattr(video_obj, "seconds", None):
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
            response_json = self._safe_json(raw_response)
            if self._is_business_error(response_json):
                raise self.get_error_class(
                    error_message=self._extract_error_message(response_json),
                    status_code=400,
                    headers=raw_response.headers,
                )
            return

        error_message: Any = raw_response.text
        response_json = self._safe_json(raw_response)
        if response_json:
            error_message = self._extract_error_message(response_json)
        raise self.get_error_class(
            error_message=str(error_message),
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

    def _safe_json(self, raw_response: httpx.Response) -> Dict[str, Any]:
        try:
            payload = raw_response.json()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {}

    def _normalize_response_json(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        data = response_json.get("data")
        if isinstance(data, dict):
            return data
        return response_json

    def _is_business_error(self, response_json: Dict[str, Any]) -> bool:
        if not response_json:
            return False

        status_value = str(response_json.get("status", "")).strip().lower()
        if status_value in {"failed", "failure", "error", "cancelled", "canceled"}:
            return True

        code = response_json.get("code")
        if code is not None and str(code).strip().lower() not in {"0", "200", "ok", "success"}:
            return True

        success = response_json.get("success")
        if success is False:
            return True
        return False

    def _extract_error_message(self, response_json: Dict[str, Any]) -> str:
        return str(
            response_json.get("detail")
            or response_json.get("msg")
            or response_json.get("message")
            or response_json.get("error")
            or response_json.get("error_message")
            or response_json
        )

    def _normalize_status(self, status: Any) -> str:
        value = str(status or "").strip().lower()
        if value in {"completed", "success", "succeeded", "done", "finished"}:
            return "completed"
        if value in {"failed", "failure", "error", "fail"}:
            return "failed"
        if value in {"cancelled", "canceled"}:
            return "cancelled"
        if value in {"running", "in_progress", "processing"}:
            return "in_progress"
        if value in {"pending", "queued", "submitted", "not_start"}:
            return "queued"
        return "queued"

    def _parse_progress(self, progress: Any, status: str) -> Optional[int]:
        if isinstance(progress, int):
            return max(0, min(100, progress))
        if isinstance(progress, float):
            return max(0, min(100, int(progress)))
        if isinstance(progress, str):
            value = progress.strip()
            if value.endswith("%"):
                value = value[:-1].strip()
            try:
                return max(0, min(100, int(float(value))))
            except (TypeError, ValueError):
                pass
        if status == "completed":
            return 100
        if status in {"queued", "in_progress"}:
            return 0
        return None

    def _safe_seconds(
        self, request_data: Optional[Dict[str, Any]], response_data: Dict[str, Any]
    ) -> Optional[str]:
        if isinstance(response_data.get("seconds"), (str, int, float)):
            return str(response_data.get("seconds"))
        if isinstance(response_data.get("duration"), (str, int, float)):
            return str(response_data.get("duration"))
        if isinstance(request_data, dict) and request_data.get("seconds") is not None:
            return str(request_data.get("seconds"))
        return None

    def _resolve_provider_model(self, model: str, optional_model: Any = None) -> str:
        if optional_model:
            return str(optional_model)
        model_token = str(model or "")
        if "/" in model_token:
            model_token = model_token.split("/", 1)[1]
        return model_token

    def _resolve_seconds(self, model: str, params: Dict[str, Any]) -> int:
        if params.get("seconds") is not None:
            return self._safe_int(params.get("seconds"), default=10)
        if params.get("duration") is not None:
            return self._safe_int(params.get("duration"), default=10)

        model_token = self._resolve_provider_model(model).lower()
        if "-25s" in model_token:
            return 25
        if "-15s" in model_token:
            return 15
        if "-10s" in model_token:
            return 10
        return 10

    def _resolve_size(self, model: str, params: Dict[str, Any]) -> str:
        size_value = params.get("size") or params.get("resolution")
        if isinstance(size_value, str) and size_value.strip():
            normalized_size = size_value.strip().lower()
            if normalized_size in {"portrait", "9:16", "720x1280", "720*1280"}:
                return "720x1280"
            if normalized_size in {"landscape", "16:9", "1280x720", "1280*720"}:
                return "1280x720"

            if normalized_size in {"sd", "hd", "fhd", "2k", "4k"}:
                aspect_ratio = str(params.get("aspect_ratio") or "").strip()
                if aspect_ratio in {"9:16", "portrait"}:
                    return "720x1280"
                if aspect_ratio in {"16:9", "landscape"}:
                    return "1280x720"

            return size_value.strip()

        aspect_ratio = params.get("aspect_ratio")
        if isinstance(aspect_ratio, str):
            normalized_aspect = aspect_ratio.strip()
            if normalized_aspect in {"16:9", "landscape"}:
                return "1280x720"
            if normalized_aspect in {"9:16", "portrait"}:
                return "720x1280"

        model_token = self._resolve_provider_model(model).lower()
        if "portrait" in model_token:
            return "720x1280"
        return "1280x720"

    def _resolve_n(self, params: Dict[str, Any]) -> int:
        value = params.get("n")
        if value is None:
            return 1
        return max(1, self._safe_int(value, default=1))

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return default

    def _extract_input_reference(self, params: Dict[str, Any]) -> Optional[Any]:
        candidate = params.get("input_reference")
        if candidate is None:
            candidate = params.get("image_url")

        if isinstance(candidate, list):
            for item in candidate:
                if item is not None:
                    return item
            return None
        return candidate

    def _build_reference_multipart(self, reference: Any) -> Any:
        parsed = self._normalize_reference(reference)
        if isinstance(parsed, str):
            url_bytes = self._download_url_bytes(parsed)
            if url_bytes is not None:
                return (
                    "input_reference.png",
                    url_bytes,
                    ImageEditRequestUtils.get_image_content_type(url_bytes),
                )
            return (None, parsed)

        if isinstance(parsed, tuple):
            return parsed
        raise ValueError("Unsupported input_reference format for BDWAPI.")

    def _normalize_reference(self, reference: Any) -> Union[str, Tuple[str, Any, str]]:
        if isinstance(reference, dict):
            url = reference.get("url")
            if not isinstance(url, str):
                nested = reference.get("image_url")
                if isinstance(nested, dict):
                    url = nested.get("url")
            if isinstance(url, str):
                return self._normalize_reference(url)
            raise ValueError("Unsupported dict input_reference format for BDWAPI.")

        if isinstance(reference, BufferedReader):
            return (
                Path(reference.name).name or "input_reference.png",
                reference,
                ImageEditRequestUtils.get_image_content_type(reference),
            )

        if isinstance(reference, Path):
            bytes_data = reference.read_bytes()
            return (
                reference.name,
                bytes_data,
                ImageEditRequestUtils.get_image_content_type(bytes_data),
            )

        if isinstance(reference, bytes):
            return (
                "input_reference.png",
                reference,
                ImageEditRequestUtils.get_image_content_type(reference),
            )

        if isinstance(reference, str):
            value = reference.strip()
            if value.startswith("http://") or value.startswith("https://"):
                return value

            if value.startswith("data:"):
                header, _, data_body = value.partition(",")
                if not data_body:
                    raise ValueError("Invalid data URL input_reference.")
                mime_type = "image/png"
                if header.startswith("data:") and ";base64" in header:
                    mime_type = header[5:].split(";", 1)[0] or mime_type
                decoded = base64.b64decode(data_body)
                filename = self._filename_from_mime(mime_type)
                return (filename, decoded, mime_type)

            possible_path = Path(value)
            if possible_path.exists() and possible_path.is_file():
                bytes_data = possible_path.read_bytes()
                return (
                    possible_path.name,
                    bytes_data,
                    ImageEditRequestUtils.get_image_content_type(bytes_data),
                )
            return value

        if hasattr(reference, "read"):
            content = reference.read()
            filename = getattr(reference, "name", "input_reference.png")
            return (
                Path(str(filename)).name or "input_reference.png",
                content,
                ImageEditRequestUtils.get_image_content_type(content),
            )

        raise ValueError("Unsupported input_reference type for BDWAPI.")

    def _filename_from_mime(self, mime_type: str) -> str:
        if "/" not in mime_type:
            return "input_reference.png"
        ext = mime_type.split("/", 1)[1].strip() or "png"
        return f"input_reference.{ext}"

    def _download_url_bytes(self, url: str) -> Optional[bytes]:
        try:
            httpx_client: HTTPHandler = _get_httpx_client()
            response = httpx_client.get(url, timeout=20)
            response.raise_for_status()
            return response.content
        except Exception:
            return None

    def _extract_video_url(self, response_data: Dict[str, Any]) -> Optional[str]:
        direct_keys = ("output_url", "video_url", "url", "download_url")
        for key in direct_keys:
            value = response_data.get(key)
            if isinstance(value, str) and value.strip():
                return value

        data_obj = response_data.get("data")
        if isinstance(data_obj, dict):
            for key in direct_keys:
                value = data_obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return None
