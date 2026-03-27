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
from litellm.llms.wtapi.videos.sora_transformation import (
    WTAPISoraVideoRequestTransformer,
)
from litellm.llms.wtapi.videos.veo_transformation import (
    WTAPIVeoVideoRequestTransformer,
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

    def __init__(self):
        super().__init__()
        self._sora_request_transformer = WTAPISoraVideoRequestTransformer()
        self._veo_request_transformer = WTAPIVeoVideoRequestTransformer()

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
        # images (required)
        images: List[str] = []
        images_value = video_create_optional_request_params.get("images")
        if isinstance(images_value, list):
            images.extend([self._image_to_base64(v) for v in images_value if v is not None])
        elif isinstance(images_value, str) and images_value:
            images.append(images_value)

        image_url = video_create_optional_request_params.get("image_url")
        if isinstance(image_url, str) and image_url:
            images.append(image_url)

        input_reference = video_create_optional_request_params.get("input_reference")
        if input_reference is not None:
            if isinstance(input_reference, list):
                for img in input_reference:
                    if img is not None:
                        images.append(self._image_to_base64(img))
            else:
                images.append(self._image_to_base64(input_reference))

        if not images:
            raise ValueError("WTAPI requires at least one image in `images`.")
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
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        task_id = response_data.get("task_id") or response_data.get("id", "")
        status_value = str(response_data.get("status", "")).strip().upper()
        status = "queued"
        if status_value in {"IN_PROGRESS", "PROCESSING", "RUNNING"}:
            status = "in_progress"
        elif status_value in {"SUCCESS", "SUCCEEDED", "COMPLETED", "DONE"}:
            status = "completed"
        elif status_value in {"FAILURE", "FAILED", "ERROR"}:
            status = "failed"

        if status == "failed":
            raise self.get_error_class(
                error_message=str(
                    response_data.get("fail_reason")
                    or response_data.get("error")
                    or response_data.get("message")
                    or "video generation failed"
                ),
                status_code=400,
                headers=raw_response.headers,
            )

        video_data: Dict[str, Any] = {
            "id": task_id,
            "object": "video",
            "status": status,
            "progress": 0 if status == "queued" else None,
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
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/v2/videos/generations/{original_video_id}"
        return url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        video_url = self._extract_video_output_url(response_data)
        if not video_url:
            raise self.get_error_class(
                error_message=self._build_video_not_ready_message(response_data),
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
        video_url = self._extract_video_output_url(response_data)
        if not video_url:
            raise self.get_error_class(
                error_message=self._build_video_not_ready_message(response_data),
                status_code=409,
                headers=raw_response.headers,
            )

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
        # Status polling should return terminal `failed` states instead of raising 500s.
        self._raise_for_status(raw_response, allow_failed_status=True)
        response_data = raw_response.json()
        status_value = response_data.get("status")

        status = "queued"
        if status_value in {"NOT_START", "QUEUED", "PENDING"}:
            status = "queued"
        elif status_value in {"IN_PROGRESS", "RUNNING", "PROCESSING"}:
            status = "in_progress"
        elif status_value in {"SUCCESS", "SUCCEEDED", "COMPLETED"}:
            status = "completed"
        elif status_value in {"FAILURE", "FAILED", "ERROR"}:
            status = "failed"

        video_url = self._extract_video_output_url(response_data)
        progress = response_data.get("progress")
        if isinstance(progress, str):
            progress = progress.strip()
            if progress.endswith("%"):
                try:
                    progress = int(progress.rstrip("%").strip())
                except ValueError:
                    progress = None
            else:
                try:
                    progress = int(progress)
                except ValueError:
                    progress = None

        video_data: Dict[str, Any] = {
            "id": response_data.get("task_id") or response_data.get("id", ""),
            "object": "video",
            "status": status,
            "progress": progress,
            "video_url": video_url,
        }

        if status == "failed":
            video_data["error"] = {
                "code": "FAILURE",
                "message": response_data.get("fail_reason")
                or response_data.get("message")
                or response_data.get("error")
                or "WTAPI video generation failed",
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

    def _raise_for_status(
        self, raw_response: httpx.Response, allow_failed_status: bool = False
    ) -> None:
        response_json: Dict[str, Any] = {}
        error_message: Any = raw_response.text

        try:
            parsed_json = raw_response.json()
            if isinstance(parsed_json, dict):
                response_json = parsed_json
        except Exception:
            response_json = {}

        if raw_response.status_code < 400:
            status_val = str(response_json.get("status", "")).strip().upper()
            if status_val in {"FAILURE", "FAILED", "ERROR"}:
                if allow_failed_status:
                    return
                self.get_error_class(
                    error_message=str(
                        response_json.get("fail_reason")
                        or response_json.get("message")
                        or response_json.get("error")
                        or response_json
                    ),
                    status_code=400,
                    headers=raw_response.headers,
                )
            return

        if response_json:
            error_message = (
                response_json.get("detail")
                or response_json.get("msg")
                or response_json.get("message")
                or response_json.get("error")
                or response_json.get("error_message")
                or response_json.get("fail_reason")
                or error_message
            )

        raise self.get_error_class(
            error_message=str(error_message),
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

    def _extract_video_output_url(self, response_data: Dict[str, Any]) -> Optional[str]:
        data = response_data.get("data")
        if isinstance(data, dict):
            output = data.get("output")
            if isinstance(output, str) and output.strip():
                return output.strip()
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
                    if isinstance(item, dict):
                        for key in ("url", "video_url", "output"):
                            value = item.get(key)
                            if isinstance(value, str) and value.strip():
                                return value.strip()
            for key in ("video_url", "url", "download_url"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        for key in ("output", "video_url", "url", "download_url"):
            value = response_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _build_video_not_ready_message(self, response_data: Dict[str, Any]) -> str:
        status = str(response_data.get("status", "")).strip() or "unknown"
        data = response_data.get("data")
        data_message = data.get("message") if isinstance(data, dict) else None
        fail_reason = (
            response_data.get("fail_reason")
            or response_data.get("message")
            or response_data.get("error")
            or data_message
        )
        if fail_reason:
            return str(fail_reason)
        return f"WTAPI video is not ready yet. status={status}"
