from decimal import Decimal
from typing import Any, override

import openai
import structlog
import tiktoken
from openai import AzureOpenAI, OpenAI

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
    LLMInputMessage,
    LLMMessageBuilderInterface,
    LLMMessageRole,
    LLMOutputMode,
    LLMPriceCalculator,
    LLMResponse,
    LLMStopReason,
    LLMTokenBudget,
)

logger = structlog.get_logger(__name__)


def _ai_agent_message_to_openai_message(
    *, message: LLMInputMessage
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

    return {"role": role, "content": message.content}


class OpenAIMessageBuilder(LLMMessageBuilderInterface):
    def __init__(self) -> None:
        self.content: list[dict[str, Any]] = []

    def add_base64_image(
        self,
        *,
        mime_type: str,  # noqa
        content: str,
    ) -> "OpenAIMessageBuilder":
        self.content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": content,
                },
            }
        )
        return self

    def add_text(self, *, text: str) -> "OpenAIMessageBuilder":
        self.content.append({"type": "text", "text": text})
        return self

    def build_message(self, role: LLMMessageRole) -> LLMInputMessage:
        return LLMInputMessage(role=role, content=self.content)


class OpenAILLM(LLM):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        token_budget: LLMTokenBudget,
        price_calculator: LLMPriceCalculator,
        temperature: float,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.price_calculator = price_calculator
        self.token_budget = token_budget
        self.temperature = temperature
        self.client = self.get_client()

    def get_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key)  # type: ignore

    @override
    def get_model(self) -> str:
        return self.model

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
                _ai_agent_message_to_openai_message(message=message)
                for message in messages
            ],
        ]

        try:
            extra_kwargs = {}
            if output_mode == LLMOutputMode.JSON:
                extra_kwargs["response_format"] = {"type": "json_object"}

            logger.debug(
                "Calling OpenAI LLM",
                model=self.model,
                temperature=self.temperature,
            )

            response = self.client.chat.completions.create(
                model=self.model,
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
            logger.debug("LLM Returned Response", llm_response=llm_response)
            return llm_response

        raise NotImplementedError(
            "Something went wrong with OpenAI Completion"
        )


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
    ) -> None:
        self.api_version = api_version
        self.endpoint = endpoint
        super().__init__(
            api_key=api_key,
            model=deployment_name,
            token_budget=token_budget,
            price_calculator=price_calculator,
            temperature=temperature,
        )

    @override
    def get_client(self) -> OpenAI:
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )


class GPT35TurboLLM(OpenAILLM):
    def __init__(self, api_key: str, temperature: float) -> None:
        super().__init__(
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


class GPT4oLLM(OpenAILLM):
    def __init__(
        self, api_key: str, temperature: float, model_suffix: str = ""
    ) -> None:
        super().__init__(
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
