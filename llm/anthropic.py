import json
from decimal import Decimal
from typing import Literal, override

import anthropic
from anthropic.types import Message, MessageParam

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
    LLMMessage,
    LLMMessageBuilderInterface,
    LLMMessageRole,
    LLMOutputMode,
    LLMPriceCalculator,
    LLMResponse,
    LLMStopReason,
    LLMTokenBudget,
)
from llm_toolkit.types import JSON  # noqa


class AnthropicLLM(LLM):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        token_budget: LLMTokenBudget,
        price_calculator: LLMPriceCalculator,
    ) -> None:
        self.client = anthropic.Anthropic(
            api_key=api_key,
        )
        self.model = model
        self.temperature = temperature
        self.token_budget = token_budget
        self.price_calculator = price_calculator
        self.system_message = ""
        self.messages: list[MessageParam] = []

    @override
    def get_model(self) -> str:
        return self.model

    @override
    def set_system_message(self, *, message: str) -> None:
        self.system_message = message

    @override
    def add_message(self, *, message: "LLMMessage") -> None:
        self.messages.append(self.__llm_message_to_anthropic_message(message))

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

    @override
    def complete_chat(
        self,
        *,
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
    ) -> LLMResponse:
        if output_mode == LLMOutputMode.JSON:
            self.messages.append(
                self.__llm_message_to_anthropic_message(
                    message=LLMMessage(
                        role=LLMMessageRole.ASSISTANT,
                        content="{",
                    ),
                )
            )
        try:
            assistant_message = self.client.messages.create(
                model=self.model,
                max_tokens=self.token_budget.max_tokens_for_output,
                temperature=self.temperature,
                system=self.system_message,
                messages=self.messages,
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

        if assistant_message.content:
            answer_text = assistant_message.content[0].text  # type: ignore
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
                cost=self.price_calculator.calculate_price(
                    input_tokens=assistant_message.usage.input_tokens,
                    output_tokens=assistant_message.usage.output_tokens,
                ),
                stop_reason=self._generate_stop_reason(
                    message=assistant_message
                ),
            )

        raise NotImplementedError(
            "Something went wrong with Anthropic Completion"
        )

    def _generate_stop_reason(self, *, message: Message) -> LLMStopReason:
        match message.stop_reason:
            case "end_turn":
                stop_reason = LLMStopReason.END_TURN
            case "max_tokens":
                stop_reason = LLMStopReason.MAX_TOKENS
            case "tool_use":
                stop_reason = LLMStopReason.TOOL_USE
            case "stop_sequence":
                stop_reason = LLMStopReason.STOP_SEQUENCE
            case _:
                raise NotImplementedError

        return stop_reason


class AnthropicMessageBuilder(LLMMessageBuilderInterface):
    def __init__(self) -> None:
        self.content: list[JSON] = []

    def add_base64_image(
        self, *, mime_type: str, content: str
    ) -> "AnthropicMessageBuilder":
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

    def add_text(self, *, text: str) -> "AnthropicMessageBuilder":
        self.content.append({"type": "text", "text": text})
        return self

    def build_message(self, role: LLMMessageRole) -> LLMMessage:
        return LLMMessage(role=role, content=self.content)


class Claude3HaikuLLM(AnthropicLLM):
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
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal(0.25),
                output_tokens=Decimal(1.25),
            ),
        )


class Claude3SonnetLLM(AnthropicLLM):
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
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal(3.0),
                output_tokens=Decimal(15.0),
            ),
        )


class Claude3OpusLLM(AnthropicLLM):
    def __init__(
        self,
        *,
        api_key: str,
        temperature: float = 1,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="claude-3-opus-20240229",
            temperature=temperature,
            token_budget=LLMTokenBudget(
                llm_max_token=200_000,
                max_tokens_for_context=195_000,
                max_tokens_for_output=4000,
            ),
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal(15.0),
                output_tokens=Decimal(75.0),
            ),
        )


class Claude3P5SonnetLLM(AnthropicLLM):
    def __init__(
        self,
        *,
        api_key: str,
        temperature: float = 1,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="claude-3-5-sonnet-20240620",
            temperature=temperature,
            token_budget=LLMTokenBudget(
                llm_max_token=200_000,
                max_tokens_for_context=195_000,
                max_tokens_for_output=4000,
            ),
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal(3.0),
                output_tokens=Decimal(15.0),
            ),
        )
