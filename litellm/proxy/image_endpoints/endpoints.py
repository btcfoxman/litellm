import asyncio
import traceback
from typing import Any, List

import orjson
from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.route_llm_request import route_request
from litellm.types.llms.openai import ChatCompletionUserMessage

router = APIRouter()

import io

from fastapi import UploadFile


async def uploadfile_to_bytesio(upload: UploadFile) -> io.BytesIO:
    """
    Read a FastAPI UploadFile into a BytesIO and set .name so OpenAI SDK
    infers filename/content-type correctly.
    """
    data = await upload.read()
    buffer = io.BytesIO(data)
    buffer.name = upload.filename
    return buffer


async def batch_to_bytesio(
    uploads: Optional[List[UploadFile]],
) -> Optional[List[io.BytesIO]]:
    """
    Convert a list of UploadFiles to a list of BytesIO buffers, or None.
    """
    if not uploads:
        return None
    return [await uploadfile_to_bytesio(u) for u in uploads]


def _extract_image_urls_from_messages(messages: Any) -> List[str]:
    """
    Extract image URLs/data-URLs from chat-style message blocks so /images/generations
    can forward them as provider-specific image refs (e.g. flow2api input_reference).
    """
    if not isinstance(messages, list):
        return []

    image_urls: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type not in ("image_url", "input_image"):
                continue

            item_image = item.get("image_url")
            if isinstance(item_image, dict):
                url = item_image.get("url")
            elif isinstance(item_image, str):
                url = item_image
            else:
                url = item.get("url")

            if isinstance(url, str) and url.strip():
                image_urls.append(url.strip())

    return image_urls


@router.post(
    "/v1/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)
@router.post(
    "/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)
@router.post(
    "/openai/deployments/{model:path}/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)  # azure compatible endpoint
async def image_generation(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: Optional[str] = None,
):
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_model,
        version,
    )

    data = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body = await request.body()
        data = orjson.loads(body)

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        data["model"] = (
            model
            or general_settings.get("image_generation_model", None)  # server default
            or user_model  # model name passed via cli args
            or data.get("model", None)  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        # Keep original image refs in case a pre-call hook rewrites request payload.
        original_image_refs = {
            "input_reference": data.get("input_reference"),
            "image": data.get("image"),
            "image_url": data.get("image_url"),
        }

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        prompt_value = data.get("prompt")
        if prompt_value is not None:
            # Reformat the image prompt as a chat message so guardrails can process it.
            user_message: ChatCompletionUserMessage = {
                "role": "user",
                "content": prompt_value,
            }
            data["messages"] = [user_message]
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="image_generation"
        )
        for key, value in original_image_refs.items():
            if value is not None and data.get(key) is None:
                data[key] = value

        messages = data.get("messages")
        if isinstance(messages, list) and messages:
            extracted_image_urls = _extract_image_urls_from_messages(messages)
            data["prompt"] = get_str_from_messages(messages)
            has_existing_images = any(
                data.get(key) is not None for key in ("input_reference", "image", "image_url")
            )
            if extracted_image_urls and not has_existing_images:
                data["input_reference"] = extracted_image_urls
        data.pop("messages", None)

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call = await route_request(
            data=data,
            route_type="aimage_generation",
            llm_router=llm_router,
            user_model=user_model,
        )
        response = await llm_call

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### CALL HOOKS ### - modify outgoing data (guardrails, otel, etc.)
        response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""
        litellm_call_id = hidden_params.get("litellm_call_id", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                call_id=litellm_call_id,
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.image_generation(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                openai_code=getattr(e, "code", None),
                code=getattr(e, "status_code", 500),
            )


@router.post(
    "/v1/images/edits",
    dependencies=[Depends(user_api_key_auth)],
    tags=["images"],
)
@router.post(
    "/images/edits",
    dependencies=[Depends(user_api_key_auth)],
    tags=["images"],
)
@router.post(
    "/openai/deployments/{model:path}/images/edits",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)  # azure compatible endpoint
async def image_edit_api(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    image: Optional[List[UploadFile]] = File(None),
    image_array: Optional[List[UploadFile]] = File(None, alias="image[]"),
    mask: Optional[List[UploadFile]] = File(None),
    mask_array: Optional[List[UploadFile]] = File(None, alias="mask[]"),
    model: Optional[str] = None,
):
    """
    Follows the OpenAI Images API spec: https://platform.openai.com/docs/api-reference/images/create

    ```bash
    curl -s -D >(grep -i x-request-id >&2) \
    -o >(jq -r '.data[0].b64_json' | base64 --decode > gift-basket.png) \
    -X POST "http://localhost:4000/v1/images/edits" \
    -H "Authorization: Bearer sk-1234" \
        -F "model=gpt-image-1" \
        -F "image[]=@soap.png" \
        -F 'prompt=Create a studio ghibli image of this'
    ```
    """
    if image is not None and image_array is not None:
        raise HTTPException(status_code=422, detail="Cannot specify both 'image' and 'image[]'")
    if mask is not None and mask_array is not None:
        raise HTTPException(status_code=422, detail="Cannot specify both 'mask' and 'mask[]'")
    if image is None and image_array is not None:
        image = image_array
    if mask is None and mask_array is not None:
        mask = mask_array

    # if image is None:
    #     raise HTTPException(status_code=422, detail="Field required: image")
    # Note: Image is optional for some models (e.g., Bedrock Stability style-transfer)
    # The validation will be done at the model level if image is truly required

    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    #########################################################
    # Read request body and convert UploadFiles to BytesIO
    #########################################################
    data = await _read_request_body(request=request)
    image_files = await batch_to_bytesio(image)
    mask_files = await batch_to_bytesio(mask)
    if image_files:
        data["image"] = image_files
    if mask_files:
        data["mask"] = mask_files
    
    # Ensure prompt exists in data (default to None for models that don't require it)
    if "prompt" not in data:
        data["prompt"] = None

    data["model"] = (
        model
        or general_settings.get("image_generation_model", None)  # server default
        or user_model  # model name passed via cli args
        or data.get("model", None)  # default passed in http request
    )
    #########################################################
    # Process request
    #########################################################

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aimage_edit",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
