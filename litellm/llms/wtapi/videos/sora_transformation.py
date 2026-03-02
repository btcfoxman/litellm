from typing import Any, Dict, List


class WTAPISoraVideoRequestTransformer:
    @staticmethod
    def get_supported_openai_params() -> List[str]:
        return [
            "model",
            "prompt",
            "input_reference",
            "image_url",
            "images",
            "aspect_ratio",
            "duration",
            "hd",
            "notify_hook",
            "watermark",
            "private",
            "user",
            "extra_headers",
            "extra_body",
        ]

    @staticmethod
    def infer_provider_model(model: str) -> str:
        model_token = WTAPISoraVideoRequestTransformer._get_model_token(model)
        if "pro" in model_token:
            return "sora-2-pro"
        return "sora-2"

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
            "duration",
            "hd",
            "notify_hook",
            "watermark",
            "private",
        ):
            if key in video_create_optional_params:
                mapped_params[key] = video_create_optional_params[key]

        if not drop_params:
            supported_openai_params = set(
                WTAPISoraVideoRequestTransformer.get_supported_openai_params()
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
        request_data: Dict[str, Any] = {
            "prompt": prompt,
            "model": provider_model,
            "aspect_ratio": WTAPISoraVideoRequestTransformer._infer_aspect_ratio_from_model(
                model
            ),
            "duration": WTAPISoraVideoRequestTransformer._infer_duration_from_model(
                model
            ),
            "hd": WTAPISoraVideoRequestTransformer._infer_hd(model),
        }

        for key in (
            "model",
            "aspect_ratio",
            "duration",
            "hd",
            "notify_hook",
            "watermark",
            "private",
        ):
            value = video_create_optional_request_params.get(key)
            if value is not None:
                request_data[key] = value

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
    def _infer_aspect_ratio_from_model(model: str) -> str:
        model_token = WTAPISoraVideoRequestTransformer._get_model_token(model)
        if "portrait" in model_token:
            return "9:16"
        return "16:9"

    @staticmethod
    def _infer_duration_from_model(model: str) -> str:
        model_token = WTAPISoraVideoRequestTransformer._get_model_token(model)
        if "25s" in model_token:
            return "25"
        if "15s" in model_token:
            return "15"
        return "10"

    @staticmethod
    def _infer_hd(model: str) -> bool:
        return "hd" in WTAPISoraVideoRequestTransformer._get_model_token(model)
