from typing import Any, Dict, List


class YWAPISoraVideoRequestTransformer:
    @staticmethod
    def get_supported_openai_params() -> List[str]:
        return [
            "model",
            "prompt",
            "input_reference",
            "image_url",
            "images",
            "orientation",
            "duration",
            "seconds",
            "size",
            "watermark",
            "user",
            "extra_headers",
            "extra_body",
        ]

    @staticmethod
    def infer_provider_model(model: str) -> str:
        model_token = YWAPISoraVideoRequestTransformer._get_model_token(model).lower()
        if "pro" in model_token:
            return "sora-2-pro-all"
        return "sora-2-all"

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
            "duration",
            "seconds",
            "size",
            "watermark",
        ):
            if key in video_create_optional_params:
                mapped_params[key] = video_create_optional_params[key]

        if not drop_params:
            supported_openai_params = set(
                YWAPISoraVideoRequestTransformer.get_supported_openai_params()
            )
            for key, value in video_create_optional_params.items():
                if key in mapped_params or key in supported_openai_params:
                    continue
                mapped_params[key] = value
        return mapped_params

    @staticmethod
    def build_video_create_request(
        model: str,
        prompt: str,
        provider_model: str,
        video_create_optional_request_params: Dict[str, Any],
        images: List[str],
    ) -> Dict[str, Any]:
        inferred_duration = YWAPISoraVideoRequestTransformer._infer_duration(model)
        duration = YWAPISoraVideoRequestTransformer._coerce_int(
            video_create_optional_request_params.get(
                "duration", video_create_optional_request_params.get("seconds")
            ),
            inferred_duration,
        )

        request_data: Dict[str, Any] = {
            "model": provider_model,
            "orientation": str(
                video_create_optional_request_params.get(
                    "orientation",
                    YWAPISoraVideoRequestTransformer._infer_orientation(model),
                )
            ),
            "prompt": prompt,
            "size": str(
                video_create_optional_request_params.get(
                    "size", YWAPISoraVideoRequestTransformer._infer_size(model)
                )
            ),
            "duration": duration,
            "watermark": YWAPISoraVideoRequestTransformer._coerce_bool(
                video_create_optional_request_params.get("watermark"), True
            ),
        }
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
    def _infer_orientation(model: str) -> str:
        model_token = YWAPISoraVideoRequestTransformer._get_model_token(model).lower()
        if "portrait" in model_token:
            return "portrait"
        return "landscape"

    @staticmethod
    def _infer_duration(model: str) -> int:
        model_token = YWAPISoraVideoRequestTransformer._get_model_token(model).lower()
        if "25s" in model_token:
            return 25
        if "15s" in model_token:
            return 15
        return 10

    @staticmethod
    def _infer_size(model: str) -> str:
        model_token = YWAPISoraVideoRequestTransformer._get_model_token(model).lower()
        if "hd" in model_token:
            return "large"
        return "small"

    @staticmethod
    def _coerce_int(value: Any, fallback: int) -> int:
        try:
            if value is None:
                return fallback
            return int(float(value))
        except (TypeError, ValueError):
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
