import json
import logging
from decimal import Decimal
from hashlib import md5
from typing import Any, override

import openai
import tiktoken
from openai import AzureOpenAI, OpenAI

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
    ImageDataExtractorLLM,
    LLMInputImage,
    LLMInputMessage,
    LLMMessageRole,
    LLMOutputMode,
    LLMPriceCalculator,
    LLMResponse,
    LLMStopReason,
    LLMTokenBudget,
)

logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    _api_key: str
    _model: str
    _response_cache: LLMResponseCache | None
    _client: OpenAI

    token_budget: LLMTokenBudget
    price_calculator: LLMPriceCalculator
    temperature: float

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        token_budget: LLMTokenBudget,
        price_calculator: LLMPriceCalculator,
        temperature: float,
        response_cache: LLMResponseCache | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self.price_calculator = price_calculator
        self.token_budget = token_budget
        self.temperature = temperature
        self._client = self.get_client()
        self._response_cache = response_cache

    def get_client(self) -> OpenAI:
        return OpenAI(api_key=self._api_key)  # type: ignore

    @override
    def get_model(self) -> str:
        return self._model

    @override
    def count_tokens(self, *, text: str) -> int:
        encoding = tiktoken.encoding_for_model(model_name=self.get_model())
        return len(encoding.encode(text=text))

    @override
    def truncate_text_to_max_tokens(
        self,
        *,
        text: str,
    ) -> str:
        encoding = tiktoken.encoding_for_model(model_name=self.get_model())
        return encoding.decode(
            encoding.encode(text=text)[
                : self.token_budget.max_tokens_for_context
            ]
        )

    @override
    def complete_chat(
        self,
        *,
        messages: list[LLMInputMessage],
        system_message: str = "",
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
    ) -> LLMResponse:
        llm_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_message,
            },
            *[
                self._convert_llm_input_message_to_raw_message(message=message)
                for message in messages
            ],
        ]

        extra_kwargs = {}
        if output_mode == LLMOutputMode.JSON:
            extra_kwargs["response_format"] = {"type": "json_object"}

        logger.debug(
            "Calling OpenAI LLM, model: %s, temperature: %s",
        )

        cache_key = ""

        if self._response_cache:
            md5_hash = md5()
            md5_hash.update(self._model.encode("utf-8"))
            md5_hash.update(str(self.temperature).encode("utf-8"))
            md5_hash.update(json.dumps(llm_messages).encode("utf-8"))
            md5_hash.update(json.dumps(extra_kwargs).encode("utf-8"))
            cache_key_suffix = md5_hash.hexdigest()
            cache_key = f"openai_response::{cache_key_suffix}"

            cached_response = self._response_cache.get(cache_key)
            if cached_response:
                logger.debug(
                    (
                        "Using cached response, "
                        "cache_key: %s, cached_response: %s"
                    ),
                    cache_key,
                    str(cached_response),
                )
                return cached_response

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=llm_messages,  # type: ignore
                temperature=self.temperature,
                **extra_kwargs,
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

        if answer := response.choices[0]:
            stop_reason: LLMStopReason
            match answer.finish_reason:
                case "stop":
                    stop_reason = LLMStopReason.END_TURN
                case "length":
                    stop_reason = LLMStopReason.MAX_TOKENS
                case "tool_calls":
                    stop_reason = LLMStopReason.TOOL_USE
                case _:
                    raise NotImplementedError

            llm_response = LLMResponse(
                llm_model=self.get_model(),
                answer=answer.message.content,
                prompt_tokens_used=response.usage.prompt_tokens,
                completion_tokens_used=response.usage.completion_tokens,
                cost=self.price_calculator.calculate_price(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                ),
                stop_reason=stop_reason,
            )
            logger.debug("LLM Returned Response: %s", llm_response)

            if self._response_cache:
                assert cache_key != ""
                self._response_cache.set(cache_key, llm_response)

            return llm_response

        raise NotImplementedError(
            "Something went wrong with OpenAI Completion"
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
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{image.mime_type};base64,{image.base64_data}"
                        ),
                    },
                },
                {"type": "text", "text": message.content.text},
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
        temperature: float,
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
            temperature=temperature,
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
        temperature: float,
        response_cache: LLMResponseCache | None = None,
    ) -> None:
        super().__init__(
            response_cache=response_cache,
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=temperature,
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


class GPT4oLLM(OpenAILLM, ImageDataExtractorLLM):
    def __init__(
        self,
        api_key: str,
        temperature: float,
        model_suffix: str = "",
        response_cache: LLMResponseCache | None = None,
    ) -> None:
        super().__init__(
            response_cache=response_cache,
            model="gpt-4o" + model_suffix,
            api_key=api_key,
            temperature=temperature,
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
