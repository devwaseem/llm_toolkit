import json
from typing import Literal

import anthropic
from anthropic.types import MessageParam

from ..types import JSON  # noqa
from .models import (
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
    LLMPrice,
    LLMRateLimitedError,
    LLMResponse,
    LLMTokenBudget,
)


class AnthropicLLM(LLM):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        token_budget: LLMTokenBudget,
        price: LLMPrice,
    ) -> None:
        self.client = anthropic.Anthropic(
            api_key=api_key,
        )
        self.model = model
        self.temperature = temperature
        self.token_budget = token_budget
        self.price = price

    def get_model(self) -> str:
        return self.model

    def __llm_message_to_anthropic_message(
        self,
        message: LLMMessage,
    ) -> MessageParam:
        role: Literal["user", "assistant"]
        match message.role:
            case LLMMessageRole.USER:
                role = "user"
            case LLMMessageRole.ASSISTANT:
                role = "assistant"
            case _:
                raise NotImplementedError(
                    f"{message.role} is not supported for Anthropic"
                )

        if isinstance(message.content, (dict, list)):
            return MessageParam(role=role, content=json.dumps(message.content))
        return MessageParam(role=role, content=str(message.content))

    def complete_chat(
        self,
        *,
        message_list: list[LLMMessage],
        system_message: str,
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
    ) -> LLMResponse:
        messages = [
            self.__llm_message_to_anthropic_message(message=message)
            for message in message_list
        ]
        if output_mode == LLMOutputMode.JSON:
            messages.append(
                self.__llm_message_to_anthropic_message(
                    message=LLMMessage(role=LLMMessageRole.USER, content="{"),
                )
            )
        try:
            assistant_message = self.client.messages.create(
                model=self.model,
                max_tokens=self.token_budget.max_tokens_for_output,
                temperature=self.temperature,
                system=system_message,
                messages=messages,
            )
        except anthropic.RateLimitError as error:
            raise LLMRateLimitedError from error
        except anthropic.APITimeoutError as error:
            raise LLMAPITimeoutError from error
        except anthropic.APIConnectionError as error:
            raise LLMAPIConnectionError from error
        except anthropic.InternalServerError as error:
            raise LLMInternalServerError from error
        except anthropic.AuthenticationError as error:
            raise LLMAuthenticationError from error
        except anthropic.PermissionDeniedError as error:
            raise LLMPermissionDeniedError from error

        if assistant_message and assistant_message.content:
            answer_text = assistant_message.content[0].text
            if output_mode == LLMOutputMode.JSON:
                answer_text = "{\n" + answer_text
            return LLMResponse(
                llm_model=self.get_model(),
                answer=LLMMessage(
                    role=LLMMessageRole.ASSISTANT,
                    content=answer_text,
                ),
                prompt_tokens_used=assistant_message.usage.input_tokens,
                completion_tokens_used=assistant_message.usage.output_tokens,
                price=self.price.calculate_price(
                    input_tokens=assistant_message.usage.input_tokens,
                    output_tokens=assistant_message.usage.output_tokens,
                ),
            )

        raise NotImplementedError(
            "Something went wrong with Anthropic Completion"
        )


class AnthropicVisionMessageBuilder(LLMMessageBuilderInterface):
    def __init__(self) -> None:
        self.content: list[JSON] = []

    def add_base64_image(
        self, *, mime_type: str, content: str
    ) -> "AnthropicVisionMessageBuilder":
        self.content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": f"'{content}'",
                },
            }
        )
        return self

    def add_text(self, *, text: str) -> "AnthropicVisionMessageBuilder":
        self.content.append({"type": "text", "text": text})
        return self

    def build_message(self, role: LLMMessageRole) -> LLMMessage:
        return LLMMessage(role=role, content=self.content)


class ClaudeHaikuLLM(AnthropicLLM):
    def __init__(
        self,
        *,
        api_key: str,
        temperature: float = 1,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="claude-3-haiku-20240307",
            temperature=temperature,
            token_budget=LLMTokenBudget(
                llm_max_token=200_000,
                max_tokens_for_context=150_000,
                max_tokens_for_output=4000,
            ),
            price=LLMPrice(
                tokens=1_000_000,
                input_tokens=0.25,
                output_tokens=1.25,
            ),
        )


class ClaudeSonnetLLM(AnthropicLLM):
    def __init__(
        self,
        *,
        api_key: str,
        temperature: float = 1,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="claude-3-sonnet-20240229",
            temperature=temperature,
            token_budget=LLMTokenBudget(
                llm_max_token=200_000,
                max_tokens_for_context=150_000,
                max_tokens_for_output=4000,
            ),
            price=LLMPrice(
                tokens=1_000_000,
                input_tokens=3.0,
                output_tokens=15.0,
            ),
        )