try:
    from google import genai
except ImportError as exc:
    raise ImportError("Please install google-genai to use Google LLM") from exc

import json
import logging
from decimal import Decimal
from typing import Any, cast, override

from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    Blob,
    Candidate,
    Content,
    ContentListUnion,
    ContentListUnionDict,
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    GoogleSearch,
    Part,
    ThinkingConfig,
    Tool,
    ToolListUnion,
)

from llm_toolkit.api_key_rotator.models import APIKeyRotator
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
        api_key: str | APIKeyRotator,
        model: str,
        price_calculator: LLMPriceCalculator,
        token_budget: LLMTokenBudget,
        response_cache: LLMResponseCache | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model

        self.price_calculator = price_calculator
        self.response_cache = response_cache
        self.token_budget = token_budget

    @override
    def get_model(self) -> str:
        return self.model

    @override
    def get_api_key(self) -> str:
        if isinstance(self.api_key, APIKeyRotator):
            return self.api_key.get_next_api_key()
        return self.api_key

    def get_client(self) -> genai.Client:
        return genai.Client(api_key=self.get_api_key())

    def _get_extra_config(self) -> dict[str, Any]:
        return {}

    @override
    def extract(
        self,
        *,
        messages: list[LLMInputMessage],
        schema: type[PydanticModel],
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
            contents=cast(ContentListUnion, llm_messages),
            response_mime_type="application/json",
            response_schema=schema,
        )

        if not response.text:
            raise LLMEmptyResponseError

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
            schema(**response_json),
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
            contents=cast(ContentListUnion, llm_messages),
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
    ) -> Content:
        match message.role:
            case LLMMessageRole.USER:
                role = "user"
            case LLMMessageRole.ASSISTANT:
                role = "model"
            case _:
                raise ValueError(
                    f"{message.role} is not supported for Google AI"
                )

        parts: list[Part]
        if isinstance(message.content, str):
            parts = [Part(text=message.content)]
        elif isinstance(message.content, LLMInputImage):
            image = message.content.image
            parts = [
                Part(text=message.content.text),
                Part(
                    inline_data=Blob(
                        data=image.base64_data,
                        mime_type=image.mime_type,
                    )
                ),
            ]
        else:
            raise NotImplementedError(f"Unhandled message type: {message}")

        return Content(role=role, parts=parts)

    def _call(
        self,
        system_message: str,
        temperature: float,
        contents: ContentListUnion | ContentListUnionDict,
        response_mime_type: str,
        response_schema: type[PydanticModel] | None = None,
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
                retry_after: int | None = None
                error_details = exc.details or []
                retry_info_list = filter(
                    lambda x: isinstance(x, dict)
                    and x.get("@type", "")
                    == "type.googleapis.com/google.rpc.RetryInfo",
                    error_details,
                )
                if retry_info := next(iter(retry_info_list), None):
                    retry_after = retry_info.get("retryDelay")

                raise LLMRateLimitedError(retry_after=retry_after) from exc

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
        contents: ContentListUnion | ContentListUnionDict,
        response_mime_type: str,
        response_schema: type[PydanticModel] | None = None,
    ) -> GenerateContentResponse:
        return self.get_client().models.generate_content(
            model=self.model,
            contents=contents,
            config=GenerateContentConfig(
                max_output_tokens=self.token_budget.max_tokens_for_output,
                system_instruction=system_message if system_message else None,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
                temperature=temperature,
                tools=self.get_tools(),
                **self._get_extra_config(),
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
    def __init__(self, api_key: str | APIKeyRotator) -> None:
        super().__init__(
            api_key=api_key,
            model="gemini-2.0-flash",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal("0.10"),
                output_tokens=Decimal("0.40"),
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


class Gemini2_5_FlashPreview(GoogleLLM):  # noqa
    def __init__(self, api_key: str | APIKeyRotator) -> None:
        super().__init__(
            api_key=api_key,
            model="gemini-2.5-flash-preview-04-17",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=Decimal("0.15"),
                output_tokens=Decimal("0.60"),
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=1_000_000,
                max_tokens_for_input=900_000,
                max_tokens_for_output=400_000,
            ),
        )

    def _get_extra_config(self) -> dict[str, Any]:
        data = super()._get_extra_config()
        data.update({"thinking_config": ThinkingConfig(thinking_budget=1024)})
        return data
