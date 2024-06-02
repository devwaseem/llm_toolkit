from enum import StrEnum

import openai
from django.conf import settings
from openai import OpenAI

from .models import (
    LLM,
    LLMMessage,
    LLMMessageRole,
    LLMPrice,
    LLMRateLimitedError,
    LLMResponse,
    LLMTokenBudget,
)


def openai_client_factory() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)  # type: ignore


class OpenAIEmbeddingModels(StrEnum):
    TEXT_ADA_002 = "text-embedding-ada-002"
    TEXT_3_LARGE = "text-embedding-3-large"
    TEXT_3_SMALL = "text-embedding-3-small"


def _ai_agent_message_to_openai_message(
    *, message: LLMMessage
) -> dict[str, str]:
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


class OpenAILLM(LLM):
    def __init__(
        self,
        *,
        model: str,
        token_budget: LLMTokenBudget,
        price: LLMPrice,
        temperature: float = 0.5,
    ) -> None:
        self.client = openai_client_factory()
        self.model = model
        self.price = price
        self.token_budget = token_budget
        self.temperature = temperature

    def get_model(self) -> str:
        return self.model

    def complete_chat(
        self, *, system_message: str, message_list: list[LLMMessage]
    ) -> LLMResponse:
        messages = [
            {
                "role": "system",
                "content": system_message,
            }
        ]
        messages += [
            _ai_agent_message_to_openai_message(message=message)
            for message in message_list
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=self.temperature,
            )
        except openai.RateLimitError as error:
            raise LLMRateLimitedError from error

        answer = response.choices[0].message.content
        if answer and response.usage:
            return LLMResponse(
                llm_model=self.get_model(),
                answer=LLMMessage(
                    role=LLMMessageRole.USER,
                    content=answer,
                ),
                prompt_tokens_used=response.usage.prompt_tokens,
                completion_tokens_used=response.usage.completion_tokens,
                price=self.price.calculate_price(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                ),
            )

        raise NotImplementedError(
            "Something went wrong with OpenAI Completion"
        )


class GPT35TurboLLM(OpenAILLM):
    def __init__(self) -> None:
        super().__init__(
            model="gpt-3.5-turbo",
            price=LLMPrice(
                tokens=1_000_000,
                input_tokens=0.50,
                output_tokens=1.50,
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=16_385,
                max_tokens_for_output=4000,
                max_tokens_for_context=12000,
            ),
        )


class GPT4oLLM(OpenAILLM):
    def __init__(self) -> None:
        super().__init__(
            model="gpt-4o",
            price=LLMPrice(
                tokens=1_000_000,
                input_tokens=0.50,
                output_tokens=1.50,
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=16_385,
                max_tokens_for_output=4000,
                max_tokens_for_context=12000,
            ),
        )
