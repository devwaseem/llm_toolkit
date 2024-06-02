from typing import Literal

import anthropic
from anthropic.types import MessageParam

from .models import (
    LLM,
    LLMMessage,
    LLMMessageRole,
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

    def __llm_message_to_anthropic_message(self, message: LLMMessage) -> MessageParam:
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

        return MessageParam(role=role, content=message.content)

    def complete_chat(
        self,
        *,
        message_list: list[LLMMessage],
        system_message: str,
    ) -> LLMResponse:
        messages = [
            self.__llm_message_to_anthropic_message(message=message)
            for message in message_list
        ]
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
        if assistant_message and assistant_message.content:
            return LLMResponse(
                llm_model=self.get_model(),
                answer=LLMMessage(
                    role=LLMMessageRole.ASSISTANT,
                    content=assistant_message.content[0].text,
                ),
                prompt_tokens_used=assistant_message.usage.input_tokens,
                completion_tokens_used=assistant_message.usage.output_tokens,
                price=self.price.calculate_price(
                    input_tokens=assistant_message.usage.input_tokens,
                    output_tokens=assistant_message.usage.output_tokens,
                ),
            )

        raise NotImplementedError("Something went wrong with Anthropic Completion")


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
