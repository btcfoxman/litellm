import base64
import re
import time
from io import BufferedReader
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

import litellm
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.openai.image_edit.transformation import ImageEditRequestUtils
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    from litellm.litellm_core_utils.logging import Logging as LiteLLMLoggingObj


class Flow2APIImageGenerationConfig(BaseImageGenerationConfig):
    """
    Map OpenAI /v1/images/generations requests to flow2api /v1/chat/completions.
    """

    _IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[[^\]]*]\((?P<url>[^)]+)\)")
    _IMAGE_HTML_PATTERN = re.compile(r"<img[^>]*src=['\"](?P<url>[^'\"]+)['\"]", re.I)
    _URL_PATTERN = re.compile(r"(?P<url>https?://[^\s'\"`<>]+)")

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n", "size", "response_format", "quality", "background", "user"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = set(self.get_supported_openai_params(model))
        for key, value in non_default_params.items():
            if key in optional_params:
                continue
            if key in supported_params:
                optional_params[key] = value
            elif not drop_params:
                raise ValueError(
                    f"Parameter {key} is not supported for model {model}. "
                    f"Supported parameters are {sorted(supported_params)}. "
                    "Set drop_params=True to drop unsupported parameters."
                )
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        resolved_api_key = (
            api_key
            or litellm_params.get("api_key")
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
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        resolved_base = (
            api_base
            or litellm_params.get("api_base")
            or "http://127.0.0.1:4020/v1"
        ).rstrip("/")
        if not resolved_base.endswith("/v1"):
            resolved_base = f"{resolved_base}/v1"
        return f"{resolved_base}/chat/completions"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        provider_model = optional_params.get("model") or model
        message_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        # Optional passthrough for img2img style usage from non-standard fields.
        input_reference = optional_params.get("input_reference") or optional_params.get(
            "image"
        )
        image_url = optional_params.get("image_url")
        source_images = input_reference if input_reference is not None else image_url
        if source_images is not None:
            if not isinstance(source_images, list):
                source_images = [source_images]
            for image_item in source_images:
                normalized_url = self._to_image_url(image_item)
                if normalized_url:
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": normalized_url}}
                    )

        payload: Dict[str, Any] = {
            "model": provider_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        prompt if len(message_content) == 1 else message_content
                    ),
                }
            ],
            "stream": False,
        }
        return payload

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        self._raise_for_status(raw_response)
        response_json = raw_response.json()
        content = self._extract_choice_content(response_json)
        image_url = self._extract_image_url(content)
        if image_url is None:
            raise self.get_error_class(
                error_message="flow2api image response did not include a valid image URL.",
                status_code=502,
                headers=raw_response.headers,
            )

        response_format = str(optional_params.get("response_format", "url")).lower()
        image_data: Dict[str, Optional[str]] = {"url": image_url, "b64_json": None}
        if response_format == "b64_json":
            image_data = {"url": None, "b64_json": self._to_b64_json(image_url)}

        response_payload = {"created": int(time.time()), "data": [image_data]}

        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_json,
        )

        image_response: ImageResponse = convert_to_model_response_object(  # type: ignore
            response_object=response_payload,
            model_response_object=model_response,
            response_type="image_generation",
        )
        image_response.size = optional_params.get("size")
        image_response.quality = optional_params.get("quality")
        image_response.output_format = response_format
        return image_response

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

    def _extract_image_url(self, content: str) -> Optional[str]:
        if not content:
            return None
        for pattern in (
            self._IMAGE_MARKDOWN_PATTERN,
            self._IMAGE_HTML_PATTERN,
            self._URL_PATTERN,
        ):
            match = pattern.search(content)
            if match:
                candidate = match.group("url").strip()
                if candidate:
                    return candidate
        return None

    def _to_b64_json(self, image_url: str) -> str:
        if image_url.startswith("data:image"):
            encoded = image_url.split("base64,", 1)
            if len(encoded) == 2:
                return encoded[1]
            raise self.get_error_class(
                error_message="Invalid data URL returned by flow2api image response.",
                status_code=502,
                headers={},
            )
        response = httpx.get(image_url, timeout=90)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")

    def _to_image_url(self, image: Any) -> Optional[str]:
        if image is None:
            return None

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

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if raw_response.status_code < 400:
            return
        error_message = raw_response.text
        try:
            response_json = raw_response.json()
            if isinstance(response_json, dict):
                if isinstance(response_json.get("error"), dict):
                    error_message = response_json["error"].get(
                        "message", error_message
                    )
                else:
                    error_message = (
                        response_json.get("detail")
                        or response_json.get("message")
                        or response_json.get("error")
                        or error_message
                    )
        except Exception:
            pass
        raise self.get_error_class(
            error_message=str(error_message),
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )
