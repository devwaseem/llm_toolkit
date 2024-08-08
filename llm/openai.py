from decimal import Decimal
from typing import override

import openai
from openai import AzureOpenAI, OpenAI

from llm_toolkit.llm.models import (
    LLM,
    LLMAPIConnectionError,
    LLMAPITimeoutError,
    LLMAuthenticationError,
    LLMInternalServerError,
    LLMMessage,
    LLMMessageBuilderInterface,
    LLMMessageRole,
    LLMOutputMode,
    LLMPermissionDeniedError,
    LLMPriceCalculator,
    LLMRateLimitedError,
    LLMResponse,
    LLMStopReason,
    LLMTokenBudget,
)
from llm_toolkit.types import JSON


def _ai_agent_message_to_openai_message(*, message: LLMMessage) -> JSON:
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
        self.content: list[JSON] = []

    def add_base64_image(
        self,
        *,
        mime_type: str,  # noqa
        content: str,
    ) -> "OpenAIMessageBuilder":
        self.content.append(
            {"type": "image_url", "image_url": {"url": content}}
        )
        return self

    def add_text(self, *, text: str) -> "OpenAIMessageBuilder":
        self.content.append({"type": "text", "text": text})
        return self

    def build_message(self, role: LLMMessageRole) -> LLMMessage:
        return LLMMessage(role=role, content=self.content)


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
        self.system_message = ""
        self.messages: list[JSON] = []
        self.client = self.get_client()

    def get_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key)  # type: ignore

    @override
    def get_model(self) -> str:
        return self.model

    @override
    def set_system_message(self, message: str) -> None:
        self.system_message = message

    @override
    def add_message(self, message: "LLMMessage") -> None:
        return self.messages.append(
            _ai_agent_message_to_openai_message(message=message)
        )

    @override
    def complete_chat(
        self,
        *,
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
    ) -> LLMResponse:
        messages: list[JSON] = [
            {
                "role": "system",
                "content": self.system_message,
            },
            *self.messages,
        ]
        try:
            extra_kwargs = {}
            if output_mode == LLMOutputMode.JSON:
                extra_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
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

            return LLMResponse(
                llm_model=self.get_model(),
                answer=LLMMessage(
                    role=LLMMessageRole.USER,
                    content=answer.message.content,
                ),
                prompt_tokens_used=response.usage.prompt_tokens,
                completion_tokens_used=response.usage.completion_tokens,
                cost=self.price_calculator.calculate_price(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                ),
                stop_reason=stop_reason,
            )

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
                max_tokens_for_output=4000,
                max_tokens_for_context=12000,
            ),
        )


class GPT4oLLM(OpenAILLM):
    def __init__(self, api_key: str, temperature: float) -> None:
        super().__init__(
            model="gpt-4o",
            api_key=api_key,
            temperature=temperature,
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal(5.0),
                output_tokens=Decimal(15.0),
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=16_385,
                max_tokens_for_output=4000,
                max_tokens_for_context=12000,
            ),
        )
