from typing import Any, Dict, List, Union


class YWAPIVeoVideoRequestTransformer:
    @staticmethod
    def get_supported_openai_params() -> List[str]:
        return [
            "model",
            "prompt",
            "input_reference",
            "image_url",
            "images",
            "aspect_ratio",
            "orientation",
            "enhance_prompt",
            "enable_upsample",
            "user",
            "extra_headers",
            "extra_body",
        ]

    @staticmethod
    def infer_provider_model(model: str) -> str:
        model_token = YWAPIVeoVideoRequestTransformer._get_model_token(model).lower()
        return model_token.replace("_", "-")

    @staticmethod
    def is_veo_provider_model(model: str) -> bool:
        return YWAPIVeoVideoRequestTransformer._get_model_token(model).lower().startswith(
            "veo"
        )

    @staticmethod
    def map_openai_params(
        video_create_optional_params: Dict[str, Any], drop_params: bool
    ) -> Dict[str, Any]:
        mapped_params: Dict[str, Any] = {}
        for key in (
            "input_reference",
            "image_url",
            "images",
            "aspect_ratio",
            "orientation",
            "enhance_prompt",
            "enable_upsample",
        ):
            if key in video_create_optional_params:
                mapped_params[key] = video_create_optional_params[key]

        if not drop_params:
            supported_openai_params = set(
                YWAPIVeoVideoRequestTransformer.get_supported_openai_params()
            )
            for key, value in video_create_optional_params.items():
                if key in mapped_params or key in supported_openai_params:
                    continue
                mapped_params[key] = value
        return mapped_params

    @staticmethod
    def build_video_create_request(
        prompt: str,
        provider_model: str,
        video_create_optional_request_params: Dict[str, Any],
        images: List[str],
    ) -> Dict[str, Any]:
        aspect_ratio = YWAPIVeoVideoRequestTransformer._normalize_aspect_ratio(
            video_create_optional_request_params.get("aspect_ratio")
            or video_create_optional_request_params.get("orientation"),
            YWAPIVeoVideoRequestTransformer._infer_aspect_ratio(provider_model),
        )
        request_data: Dict[str, Any] = {
            "model": provider_model,
            "prompt": prompt,
            "enhance_prompt": YWAPIVeoVideoRequestTransformer._coerce_bool(
                video_create_optional_request_params.get("enhance_prompt"), True
            ),
            "enable_upsample": YWAPIVeoVideoRequestTransformer._coerce_enable_upsample(
                video_create_optional_request_params.get("enable_upsample"), False
            ),
        }
        if (
            "aspect_ratio" in video_create_optional_request_params
            or "orientation" in video_create_optional_request_params
            or YWAPIVeoVideoRequestTransformer._get_model_token(provider_model)
            .lower()
            .startswith("veo3")
        ):
            request_data["aspect_ratio"] = aspect_ratio

        if images:
            request_data["images"] = images
        return request_data

    @staticmethod
    def _get_model_token(model: str) -> str:
        if not model:
            return ""
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    @staticmethod
    def _infer_aspect_ratio(model: str) -> str:
        model_token = YWAPIVeoVideoRequestTransformer._get_model_token(model).lower()
        if "portrait" in model_token:
            return "9:16"
        return "16:9"

    @staticmethod
    def _normalize_aspect_ratio(value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        parsed = str(value).strip().lower()
        if parsed in {"16:9", "landscape", "horizontal"}:
            return "16:9"
        if parsed in {"9:16", "portrait", "vertical"}:
            return "9:16"
        return fallback

    @staticmethod
    def _coerce_bool(value: Any, fallback: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return fallback

    @staticmethod
    def _coerce_enable_upsample(value: Any, fallback: bool) -> Union[bool, str]:
        if value is None:
            return fallback
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
            return value.strip()
        return fallback
