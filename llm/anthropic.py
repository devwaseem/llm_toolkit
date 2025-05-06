try:
    import anthropic
except ImportError as exc:
    raise RuntimeError(
        "Anthropic is not installed: install it using `anthropic` package"
    ) from exc


from decimal import Decimal
from typing import Literal, override

from anthropic.types import (
    ImageBlockParam,
    Message,
    MessageParam,
    TextBlockParam,
)
from anthropic.types.image_block_param import Source

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
)


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

    @override
    def get_model(self) -> str:
        return self.model

    @override
    def complete_chat(
        self,
        *,
        messages: list[LLMInputMessage],
        system_message: str = "",
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
    ) -> LLMResponse:
        llm_messages: list[MessageParam] = [
            self.__llm_message_to_anthropic_message(message)
            for message in messages
        ]

        if output_mode == LLMOutputMode.JSON:
            llm_messages.append(
                self.__llm_message_to_anthropic_message(
                    message=LLMInputMessage(
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
                system=system_message,
                messages=llm_messages,
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
                answer=answer_text,
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

    def __llm_message_to_anthropic_message(
        self,
        message: LLMInputMessage,
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

        if isinstance(message.content, str):
            content = message.content
        elif isinstance(message.content, LLMInputImage):
            image = message.content.image
            content = [
                ImageBlockParam(  # type: ignore
                    source=Source(  # type: ignore
                        media_type=image.mime_type,
                        data=image.base64_data,
                    )
                ),
                TextBlockParam(  # type: ignore
                    text=message.content.text,
                ),
            ]
            return MessageParam(
                role=role,
                content=content,
            )
        else:
            raise NotImplementedError(f"Unhandled message type: {message}")

        return MessageParam(role=role, content=message.content)

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
                max_tokens_for_input=190_000,
                max_tokens_for_output=4_000,
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
                max_tokens_for_input=190_000,
                max_tokens_for_output=4_000,
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
                max_tokens_for_input=190_000,
                max_tokens_for_output=4_000,
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
                max_tokens_for_input=190_000,
                max_tokens_for_output=4_000,
            ),
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal(3.0),
                output_tokens=Decimal(15.0),
            ),
        )
