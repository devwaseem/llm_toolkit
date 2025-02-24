from decimal import Decimal
from typing import Any, override

from google.genai.types import Candidate, FinishReason, GenerateContentConfig
import structlog
from llm_toolkit.cache.models import LLMResponseCache
from llm_toolkit.llm.errors import LLMEmptyResponseError
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
from google import genai

logger = structlog.get_logger(__name__)


class GoogleLLM(LLM):
    def __init__(
        self,
        api_key: str,
        model: str,
        price_calculator: LLMPriceCalculator,
        temperature: float,
        response_cache: LLMResponseCache | None = None,
    ):
        self.api_key = api_key
        self.model = model

        self.price_calculator = price_calculator
        self.temperature = temperature
        self.response_cache = response_cache
        self.client = self.get_client()

        model_info = self.get_client().models.get(model=model)
        self.token_budget = LLMTokenBudget(
            llm_max_token=(
                (model_info.input_token_limit or 0)
                + (model_info.output_token_limit or 0)
            ),
            max_tokens_for_input=model_info.input_token_limit or 0,
            max_tokens_for_output=model_info.output_token_limit or 0,
        )

    @override
    def get_model(self) -> str:
        return self.model

    @override
    def count_tokens(self, *, text: str) -> int:
        raise NotImplementedError

    @override
    def truncate_text_to_max_tokens(
        self,
        *,
        text: str,
    ) -> str:
        raise NotImplementedError

    def get_client(self) -> genai.Client:
        client = genai.Client(api_key=self.api_key)
        return client

    @override
    def complete_chat(
        self,
        *,
        messages: list[LLMInputMessage],
        system_message: str = "",
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
    ) -> LLMResponse:
        llm_messages = [
            self._convert_llm_input_message_to_raw_message(message=message)
            for message in messages
        ]

        match output_mode:
            case LLMOutputMode.JSON:
                response_mime_type = "application/json"
            case _:
                response_mime_type = "text/plain"

        response = self.get_client().models.generate_content(
            model=self.model,
            contents=llm_messages,  # type: ignore
            config=GenerateContentConfig(
                max_output_tokens=8192,
                system_instruction=system_message,
                response_mime_type=response_mime_type,
            ),
        )

        logger.debug("Data from LLM", response=response)

        if not response.candidates:
            raise LLMEmptyResponseError

        candidate = response.candidates[0]

        if (
            not candidate.content
            or not candidate.content.parts
            or not candidate.content.parts[0].text
        ):
            raise LLMEmptyResponseError

        answer_text = candidate.content.parts[0].text

        usage = response.usage_metadata

        assert usage is not None, "Usage metadata is not available"

        prompt_tokens_used = usage.prompt_token_count or 0
        completion_tokens_used = usage.candidates_token_count or 0

        return LLMResponse(
            llm_model=self.get_model(),
            answer=answer_text,
            prompt_tokens_used=prompt_tokens_used,
            completion_tokens_used=completion_tokens_used,
            cost=self.price_calculator.calculate_price(
                input_tokens=prompt_tokens_used,
                output_tokens=completion_tokens_used,
            ),
            stop_reason=self._get_stop_reason(response_candidate=candidate),
        )

    def _get_stop_reason(self, response_candidate: Candidate) -> LLMStopReason:
        match response_candidate.finish_reason:
            case FinishReason.STOP:
                return LLMStopReason.END_TURN
            case FinishReason.MAX_TOKENS:
                return LLMStopReason.MAX_TOKENS
            case _:
                raise ValueError(
                    f"Unknown finish reason: {response_candidate.finish_reason}"
                )

    def _convert_llm_input_message_to_raw_message(
        self, *, message: LLMInputMessage
    ) -> dict[str, Any]:
        match message.role:
            case LLMMessageRole.USER:
                role = "user"
            case LLMMessageRole.ASSISTANT:
                role = "model"
            case _:
                raise ValueError(f"{message.role} is not supported for Google AI")

        if isinstance(message.content, str):
            return {
                "role": role,
                "parts": [{"text": message.content}],
            }

        elif isinstance(message.content, LLMInputImage):
            image = message.content.image
            return {
                "role": role,
                "parts": [
                    {"text": message.content.text},
                    {
                        "inline_data": {
                            "mime_type": image.mime_type,
                            "data": image.base64_data,
                        }
                    },
                ],
            }

        raise NotImplementedError(f"Unhandled message type: {message}")


class Gemini2_0_Flash(GoogleLLM):
    def __init__(self, api_key: str, temperature: float) -> None:
        super().__init__(
            api_key=api_key,
            model="gemini-2.0-flash",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal(0.10),
                output_tokens=Decimal(0.40),
            ),
            temperature=temperature,
        )
