from io import BufferedReader, BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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


class GGAPIVideoConfig(BaseVideoConfig):
    """
    GeminiGen (ggapi) Sora video generation.

    API Base: https://api.geminigen.ai
    Endpoints:
      - POST /uapi/v1/video-gen/sora
      - GET  /uapi/v1/history/{uuid}
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
            "resolution",
            "ref_history",
            "file_urls",
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
            "resolution",
            "ref_history",
            "file_urls",
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
            or get_secret_str("GGAPI_API_KEY")
            or get_secret_str("GEMINIGEN_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "GGAPI API key is required. Set GGAPI_API_KEY env var or pass api_key."
            )

        headers.update({"x-api-key": api_key})
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or "https://api.geminigen.ai"
        return api_base.rstrip("/")

    def _infer_aspect_ratio(self, model: str) -> str:
        if "portrait" in model:
            return "portrait"
        return "landscape"

    def _infer_duration(self, model: str) -> str:
        if "25s" in model:
            return "25"
        if "15s" in model:
            return "15"
        return "10"

    def _infer_model_variant(self, model: str) -> str:
        if "pro-hd" in model or "pro-hd" in model:
            return "sora-2-pro-hd"
        if "pro" in model:
            return "sora-2-pro"
        return "sora-2"

    def _infer_resolution(self, model: str) -> str:
        if "pro-hd" in model:
            return "large"
        return "small"

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
            files_list.append((field_name, ("image.png", image, image_content_type)))

    def _add_form_field(
        self,
        files_list: List[Tuple[str, Any]],
        field_name: str,
        value: Any,
    ) -> None:
        files_list.append((field_name, (None, str(value))))

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
            "duration": int(self._infer_duration(model)),
            "aspect_ratio": self._infer_aspect_ratio(model),
            "resolution": self._infer_resolution(model),
        }

        # Allow overrides
        for key in ("model", "duration", "aspect_ratio", "resolution", "ref_history"):
            value = video_create_optional_request_params.get(key)
            if value is not None:
                request_data[key] = value

        files_list: List[Tuple[str, Any]] = []
        file_urls: List[str] = []

        file_urls_value = video_create_optional_request_params.get("file_urls")
        if isinstance(file_urls_value, list):
            file_urls.extend([str(v) for v in file_urls_value if v])
        elif isinstance(file_urls_value, str) and file_urls_value:
            file_urls.append(file_urls_value)

        images_value = video_create_optional_request_params.get("images")
        if isinstance(images_value, list):
            for img in images_value:
                if isinstance(img, str):
                    file_urls.append(img)
                else:
                    self._add_image_to_files(files_list, img, "files")
        elif images_value is not None:
            if isinstance(images_value, str):
                file_urls.append(images_value)
            else:
                self._add_image_to_files(files_list, images_value, "files")

        image_url = video_create_optional_request_params.get("image_url")
        if isinstance(image_url, str) and image_url:
            file_urls.append(image_url)

        input_reference = video_create_optional_request_params.get("input_reference")
        if input_reference is not None:
            if isinstance(input_reference, list):
                for img in input_reference:
                    self._add_image_to_files(files_list, img, "files")
            else:
                self._add_image_to_files(files_list, input_reference, "files")

        if file_urls:
            # API currently accepts a single URL; send the first one as a form field
            self._add_form_field(files_list, "file_urls", file_urls[0])

        full_api_base = f"{api_base}/uapi/v1/video-gen/sora"
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
        if response_data.get("status") == 3:
            raise self.get_error_class(
                error_message=str(response_data.get("error_message") or "video generation failed"),
                status_code=400,
                headers=raw_response.headers,
            )
        task_id = response_data.get("uuid") or response_data.get("id", "")

        video_data: Dict[str, Any] = {
            "id": task_id,
            "object": "video",
            "status": "queued",
            "progress": response_data.get("status_percentage") or 0,
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
        url = f"{api_base}/uapi/v1/history/{original_video_id}"
        return url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        video_url = self._extract_video_url(response_data)
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
        video_url = self._extract_video_url(response_data)
        if not video_url:
            raise ValueError("video_url not found in response. Video may not be ready.")

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.GGAPI,
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
        raise NotImplementedError("Video remix is not supported by GGAPI.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by GGAPI.")

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
        raise NotImplementedError("Video list is not supported by GGAPI.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video list is not supported by GGAPI.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video delete is not supported by GGAPI.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by GGAPI.")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/uapi/v1/history/{original_video_id}"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        response_data = raw_response.json()
        status_value = response_data.get("status")

        status = "queued"
        if status_value == 1:
            status = "in_progress"
        elif status_value == 2:
            status = "completed"
        elif status_value == 3:
            status = "failed"

        progress = response_data.get("status_percentage")
        if isinstance(progress, str):
            try:
                progress = int(progress)
            except ValueError:
                progress = None

        video_url = self._extract_video_url(response_data)

        video_data: Dict[str, Any] = {
            "id": response_data.get("uuid", ""),
            "object": "video",
            "status": status,
            "progress": progress,
            "video_url": video_url,
        }

        if status == "failed":
            video_data["error"] = {
                "code": response_data.get("error_code"),
                "message": response_data.get("error_message"),
            }
            raise self.get_error_class(
                error_message=str(video_data["error"]["message"]),
                status_code=400,
                headers=raw_response.headers,
            )

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

    def _extract_video_url(self, response_data: Dict[str, Any]) -> Optional[str]:
        generated_videos = response_data.get("generated_video") or []
        if generated_videos and isinstance(generated_videos, list):
            first = generated_videos[0] or {}
            url = first.get("video_url")
            if url:
                return url
        return response_data.get("generate_result")
