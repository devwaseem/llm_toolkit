try:
    from google import genai
except ImportError as exc:
    raise ImportError("Please install google-genai to use Google LLM") from exc

import json
import logging
from decimal import Decimal
from typing import Any, Type, cast, override

from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    Candidate,
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    GoogleSearch,
    Tool,
    ToolListUnion,
)

from llm_toolkit.cache.models import LLMResponseCache
from llm_toolkit.llm.errors import (
    LLMAuthenticationError,
    LLMEmptyResponseError,
    LLMInternalServerError,
    LLMJsonResponseDecodingError,
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
    PydanticModel,
    StructuredOutputLLM,
)

logger = logging.getLogger(__name__)


class GoogleLLM(LLM, StructuredOutputLLM):
    def __init__(
        self,
        api_key: str,
        model: str,
        price_calculator: LLMPriceCalculator,
        token_budget: LLMTokenBudget,
        response_cache: LLMResponseCache | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model

        self.price_calculator = price_calculator
        self.response_cache = response_cache
        self.client = self.get_client()
        self.token_budget = token_budget

    @override
    def get_model(self) -> str:
        return self.model

    def get_client(self) -> genai.Client:
        return genai.Client(api_key=self.api_key)

    @override
    def extract(
        self,
        *,
        messages: list[LLMInputMessage],
        schema: Type[PydanticModel],
        system_message: str = "",
        temperature: float = 0,
    ) -> tuple[PydanticModel, LLMResponse]:
        llm_messages = [
            self._convert_llm_input_message_to_raw_message(message=message)
            for message in messages
        ]
        response = self._call(
            system_message=system_message,
            temperature=temperature,
            contents=llm_messages,
            response_mime_type="application/json",
            response_schema=schema,
        )

        response_json = {}
        try:
            response_json = json.loads(response.text)
        except json.JSONDecodeError as exc:
            logger.exception(
                "%s: Failed to decode JSON response: %s",
                self.get_model(),
                response.text,
                exc_info=exc,
                stack_info=False,
            )
            raise LLMJsonResponseDecodingError from exc

        if response.parsed is None:
            raise LLMEmptyResponseError

        return (
            cast(PydanticModel, schema(**response_json)),
            self._to_llm_response(response=response),
        )

    @override
    def complete_chat(
        self,
        *,
        messages: list[LLMInputMessage],
        system_message: str = "",
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
        temperature: float = 0,
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

        response = self._call(
            system_message=system_message,
            temperature=temperature,
            contents=llm_messages,
            response_mime_type=response_mime_type,
        )
        return self._to_llm_response(response=response)

    def _get_stop_reason(self, response_candidate: Candidate) -> LLMStopReason:
        match response_candidate.finish_reason:
            case FinishReason.STOP:
                return LLMStopReason.END_TURN
            case FinishReason.MAX_TOKENS:
                return LLMStopReason.MAX_TOKENS
            case _:
                raise ValueError(
                    "Unknown finish reason: "
                    + str(response_candidate.finish_reason)
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
                raise ValueError(
                    f"{message.role} is not supported for Google AI"
                )

        parts: list[dict[str, Any]]
        if isinstance(message.content, str):
            parts = [{"text": message.content}]
        elif isinstance(message.content, LLMInputImage):
            image = message.content.image
            parts = [
                {"text": message.content.text},
                {
                    "inline_data": {
                        "mime_type": image.mime_type,
                        "data": image.base64_data,
                    }
                },
            ]
        else:
            raise NotImplementedError(f"Unhandled message type: {message}")

        return {
            "role": role,
            "parts": parts,
        }

    def _call(
        self,
        system_message: str,
        temperature: float,
        contents: str | list[dict[str, Any]],
        response_mime_type: str,
        response_schema: Type[PydanticModel] | None = None,
    ) -> GenerateContentResponse:
        try:
            response = self._call_llm(
                system_message=system_message,
                temperature=temperature,
                contents=contents,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
            )
        except ClientError as exc:
            logger.exception(
                "%s: Client Error calling LLM, code: %d, status: %s",
                self.get_model(),
                exc.code,
                exc.status,
                exc_info=exc,
                stack_info=False,
            )
            if exc.code == 429:
                raise LLMRateLimitedError from exc

            if exc.code == 401:
                raise LLMAuthenticationError from exc

            if exc.code == 403:
                raise LLMPermissionDeniedError from exc

            raise exc from exc

        except ServerError as exc:
            logger.exception(
                "%s: Server Error calling LLM",
                self.get_model(),
                exc_info=exc,
                stack_info=False,
            )
            raise LLMInternalServerError from exc

        return response

    def _call_llm(
        self,
        system_message: str,
        temperature: float,
        contents: str | list[dict[str, Any]],
        response_mime_type: str,
        response_schema: Type[PydanticModel] | None = None,
    ) -> GenerateContentResponse:
        return self.get_client().models.generate_content(
            model=self.model,
            contents=contents,  # type: ignore
            config=GenerateContentConfig(
                max_output_tokens=self.token_budget.max_tokens_for_output,
                system_instruction=system_message if system_message else None,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
                temperature=temperature,
                tools=self.get_tools(),
            ),
        )

    def get_tools(self) -> ToolListUnion | None:
        return None

    def _to_llm_response(
        self, response: GenerateContentResponse
    ) -> LLMResponse:
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


class Gemini2_0_Flash(GoogleLLM):  # noqa
    def __init__(self, api_key: str) -> None:
        super().__init__(
            api_key=api_key,
            model="gemini-2.0-flash",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal(0.10),
                output_tokens=Decimal(0.40),
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=1_000_000,
                max_tokens_for_input=900_000,
                max_tokens_for_output=400_000,
            ),
        )


class Gemini2_0_FlashWithGroundingSearch(  # noqa
    Gemini2_0_Flash
):
    def get_tools(self) -> ToolListUnion | None:
        return [
            Tool(google_search=GoogleSearch()),
        ]
