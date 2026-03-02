from typing import Any, Dict, List, Optional


class WTAPIVeoVideoRequestTransformer:
    @staticmethod
    def get_supported_openai_params() -> List[str]:
        return [
            "model",
            "prompt",
            "input_reference",
            "image_url",
            "images",
            "orientation",
            "aspect_ratio",
            "enhance_prompt",
            "user",
            "extra_headers",
            "extra_body",
        ]

    @staticmethod
    def infer_provider_model(model: str) -> str:
        return WTAPIVeoVideoRequestTransformer._get_model_token(model)

    @staticmethod
    def is_veo_provider_model(model: str) -> bool:
        return WTAPIVeoVideoRequestTransformer._get_model_token(model).lower().startswith(
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
            "orientation",
            "aspect_ratio",
            "enhance_prompt",
        ):
            if key in video_create_optional_params:
                mapped_params[key] = video_create_optional_params[key]

        if not drop_params:
            supported_openai_params = set(
                WTAPIVeoVideoRequestTransformer.get_supported_openai_params()
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
        request_data: Dict[str, Any] = {
            "prompt": prompt,
            "model": provider_model,
        }
        if "enhance_prompt" in video_create_optional_request_params:
            request_data["enhance_prompt"] = WTAPIVeoVideoRequestTransformer._coerce_bool(
                video_create_optional_request_params.get("enhance_prompt"), False
            )

        aspect_ratio = WTAPIVeoVideoRequestTransformer._normalize_aspect_ratio(
            video_create_optional_request_params.get("aspect_ratio")
        )
        if aspect_ratio is None:
            aspect_ratio = WTAPIVeoVideoRequestTransformer._normalize_aspect_ratio(
                video_create_optional_request_params.get("orientation")
            )
        if aspect_ratio is not None:
            request_data["aspect_ratio"] = aspect_ratio

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
    def _coerce_bool(value: Any, fallback: bool) -> bool:
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
        return fallback

    @staticmethod
    def _normalize_aspect_ratio(value: Any) -> Optional[str]:
        if value is None:
            return None
        parsed = str(value).strip().lower()
        if not parsed:
            return None
        if parsed in {"9:16", "portrait", "vertical"}:
            return "9:16"
        if parsed in {"16:9", "landscape", "horizontal"}:
            return "16:9"
        return str(value)
