import json
import logging
from decimal import Decimal
from typing import Any, Type, override

import openai
from openai import AzureOpenAI, OpenAI
from openai.lib._pydantic import to_strict_json_schema
from openai.types.responses.response import Response

from llm_toolkit.api_key_rotator.models import APIKeyRotator, BaseAPIKeyRotator
from llm_toolkit.cache.models import LLMResponseCache
from llm_toolkit.llm.errors import (
    LLMAPIConnectionError,
    LLMAPITimeoutError,
    LLMAuthenticationError,
    LLMInternalServerError,
    LLMPermissionDeniedError,
    LLMRateLimitedError,
)
from llm_toolkit.llm.models import (
    LLM,
    LLMInputImage,
    LLMInputMessage,
    LLMMessageRole,
    LLMOutputMode,
    LLMPriceCalculator,
    LLMResponse,
    LLMStopReason,
    LLMTokenBudget,
    PydanticModel,
    StructuredOutputLLM,
)

logger = logging.getLogger(__name__)


class OpenAILLM(LLM, StructuredOutputLLM):
    _api_key: str
    _model: str
    _response_cache: LLMResponseCache | None
    _client: OpenAI

    token_budget: LLMTokenBudget
    price_calculator: LLMPriceCalculator

    def __init__(
        self,
        *,
        api_key: str | BaseAPIKeyRotator,
        model: str,
        token_budget: LLMTokenBudget,
        price_calculator: LLMPriceCalculator,
        response_cache: LLMResponseCache | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self.price_calculator = price_calculator
        self.token_budget = token_budget
        self._response_cache = response_cache

    def get_client(self) -> OpenAI:
        return OpenAI(api_key=self.get_api_key())  # type: ignore

    @override
    def get_api_key(self) -> str:
        if isinstance(self._api_key, APIKeyRotator):
            return self._api_key.get_next_api_key()
        return self._api_key

    @override
    def get_model(self) -> str:
        return self._model

    @override
    def extract(
        self,
        *,
        messages: list[LLMInputMessage],
        schema: Type[PydanticModel],
        system_message: str = "",
        temperature: float = 0,
    ) -> tuple[PydanticModel, LLMResponse]:
        response = self._call(
            system_message=system_message,
            temperature=temperature,
            messages=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": to_strict_json_schema(schema),
                    "strict": True,
                }
            },
        )

        return (
            schema(**json.loads(response.output_text)),
            self._to_llm_response(response=response),
        )

    @override
    def complete_chat(
        self,
        *,
        messages: list[LLMInputMessage],
        system_message: str = "",
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
        temperature: float = 0,
    ) -> LLMResponse:
        output_type = "text"
        if output_mode == LLMOutputMode.JSON:
            output_type = "json_object"

        response = self._call(
            system_message=system_message,
            temperature=temperature,
            messages=messages,
            text={
                "format": {
                    "type": output_type,
                }
            },
        )

        return self._to_llm_response(response=response)

    def _call(
        self,
        *,
        system_message: str,
        temperature: float,
        messages: list[LLMInputMessage],
        text: dict[str, Any],
    ) -> Response:
        llm_input = []
        if system_message:
            llm_input.append(
                {
                    "role": "system",
                    "content": system_message,
                }
            )

        llm_input += [
            self._convert_llm_input_message_to_raw_message(message=message)
            for message in messages
        ]

        logger.debug(
            "Calling OpenAI LLM, model: %s, temperature: %s",
        )

        try:
            response = self.get_client().responses.create(
                model=self.get_model(),
                temperature=temperature,
                input=llm_input,
                text=text,  # type: ignore
            )
        except openai.RateLimitError as error:
            raise LLMRateLimitedError from error
        except openai.APITimeoutError as error:
            raise LLMAPITimeoutError from error
        except openai.APIConnectionError as error:
            raise LLMAPIConnectionError from error
        except openai.InternalServerError as error:
            raise LLMInternalServerError from error
        except openai.AuthenticationError as error:
            raise LLMAuthenticationError from error
        except openai.PermissionDeniedError as error:
            raise LLMPermissionDeniedError from error

        logger.debug("LLM Returned Response: %s", response)

        return response

    def _to_llm_response(self, *, response: Response) -> LLMResponse:
        stop_reason: LLMStopReason = LLMStopReason.END_TURN
        if incomplete_details := response.incomplete_details:
            match incomplete_details.reason:
                case "max_output_tokens":
                    stop_reason = LLMStopReason.MAX_TOKENS
                case _:
                    raise NotImplementedError

        assert response.usage is not None, "Usage metadata is not available"

        return LLMResponse(
            llm_model=self.get_model(),
            answer=response.output_text,
            prompt_tokens_used=response.usage.input_tokens,
            completion_tokens_used=response.usage.output_tokens,
            cost=self.price_calculator.calculate_price(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=stop_reason,
        )

    def _convert_llm_input_message_to_raw_message(
        self, message: LLMInputMessage
    ) -> dict[str, Any]:
        match message.role:
            case LLMMessageRole.USER:
                role = "user"
            case LLMMessageRole.ASSISTANT:
                role = "assistant"
            case _:
                raise NotImplementedError(
                    f"{message.role} is not supported for OpenAI"
                )
        content = {}
        if isinstance(message.content, str):
            content = message.content
        elif isinstance(message.content, LLMInputImage):
            image = message.content.image
            content = [
                {"type": "input_text", "text": message.content.text},
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:{image.mime_type};base64,{image.base64_data}"
                    ),
                },
            ]
        else:
            raise NotImplementedError(f"Unhandled message type: {message}")

        return {
            "role": role,
            "content": content,
        }


class AzureOpenAILLM(OpenAILLM):
    def __init__(
        self,
        *,
        api_key: str,
        api_version: str,
        endpoint: str,
        deployment_name: str,
        token_budget: LLMTokenBudget,
        price_calculator: LLMPriceCalculator,
        response_cache: LLMResponseCache | None = None,
    ) -> None:
        self.api_version = api_version
        self.endpoint = endpoint
        super().__init__(
            response_cache=response_cache,
            api_key=api_key,
            model=deployment_name,
            token_budget=token_budget,
            price_calculator=price_calculator,
        )

    @override
    def get_client(self) -> OpenAI:
        return AzureOpenAI(
            api_key=self._api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )


class GPT35TurboLLM(OpenAILLM):
    def __init__(
        self,
        api_key: str,
        response_cache: LLMResponseCache | None = None,
    ) -> None:
        super().__init__(
            response_cache=response_cache,
            api_key=api_key,
            model="gpt-3.5-turbo",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal(0.50),
                output_tokens=Decimal(1.50),
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=16_385,
                max_tokens_for_input=12000,
                max_tokens_for_output=4000,
            ),
        )


class GPT4oLLM(OpenAILLM):
    def __init__(
        self,
        api_key: str,
        model_suffix: str = "",
        response_cache: LLMResponseCache | None = None,
    ) -> None:
        super().__init__(
            response_cache=response_cache,
            model="gpt-4o" + model_suffix,
            api_key=api_key,
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal(5.0),
                output_tokens=Decimal(15.0),
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=128_000,
                max_tokens_for_input=124_00,
                max_tokens_for_output=4_000,
            ),
        )
