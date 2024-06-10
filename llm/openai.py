import openai
from django.conf import settings
from openai import OpenAI

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


def openai_client_factory() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)  # type: ignore


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


class OpenAIVisionMessageBuilder(LLMMessageBuilderInterface):
    def __init__(self) -> None:
        self.content: list[JSON] = []

    def add_image(self, *, url: str) -> "OpenAIVisionMessageBuilder":
        self.content.append({"type": "image_url", "image_url": {"url": url}})
        return self

    def add_text(self, *, text: str) -> "OpenAIVisionMessageBuilder":
        self.content.append({"type": "text", "text": text})
        return self

    def build_message(self, role: LLMMessageRole) -> LLMMessage:
        return LLMMessage(role=role, content=self.content)


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
        self,
        *,
        system_message: str,
        message_list: list[LLMMessage],
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
    ) -> LLMResponse:
        messages: list[JSON] = [
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
                input_tokens=5.0,
                output_tokens=15.0,
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=16_385,
                max_tokens_for_output=4000,
                max_tokens_for_context=12000,
            ),
        )
