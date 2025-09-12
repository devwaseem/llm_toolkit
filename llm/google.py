try:
    from google import genai
except ImportError as exc:
    raise ImportError("Please install google-genai to use Google LLM") from exc

import json
import logging
from typing import Any, override
from uuid import uuid4

from google import genai
from google.genai.errors import ClientError, ServerError

from llm_toolkit.api_key_rotator.models import APIKeyRotator
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
    LLMInputFile,
    LLMInputMessage,
    LLMMessageRole,
    LLMOutputMode,
    LLMPriceCalculator,
    LLMResponse,
    LLMStopReason,
    LLMTokenBudget,
    LLMToolCall,
    LLMToolCallResponse,
    LLMToolRegistry,
    LLMTools,
    PydanticModel,
    StructuredOutputLLM,
)

logger = logging.getLogger(__name__)


class GoogleLLM(LLM, StructuredOutputLLM):
    tool_registry: LLMToolRegistry

    def __init__(
        self,
        api_key: str | APIKeyRotator,
        model: str,
        price_calculator: LLMPriceCalculator,
        token_budget: LLMTokenBudget,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.price_calculator = price_calculator
        self.token_budget = token_budget

    @override
    def get_model(self) -> str:
        return self.model

    @override
    def get_api_key(self) -> str:
        if isinstance(self.api_key, APIKeyRotator):
            return self.api_key.get_next_api_key()
        return self.api_key

    @override
    def extract(
        self,
        *,
        messages: list[LLMInputMessage],
        schema: type[PydanticModel],
        system_message: str = "",
        temperature: float = 0,
        tools: LLMTools | None = None,
    ) -> tuple[PydanticModel | None, LLMResponse]:
        tool_registry = LLMToolRegistry()
        if tools:
            for tool in tools.tools:
                tool_registry.add(tool=tool)

        llm_messages = [
            self._convert_llm_input_message_to_genai_content(message=message)
            for message in messages
        ]

        response = self._call(
            system_message=system_message,
            temperature=temperature,
            contents=llm_messages,
            response_mime_type="application/json",
            response_schema=schema,
            tool_registry=tool_registry,
        )

        if response.function_calls:
            return None, self._to_llm_response(response=response)

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

        data = schema(**response_json)
        llm_response = self._to_llm_response(response=response)

        if (
            tools
            and llm_response.stop_reason == LLMStopReason.TOOL_USE
            and llm_response.function_calls
            and tools.call_automatically
        ):
            messages = tool_registry.process_tool_calls(
                messages=messages,
                tool_calls=llm_response.function_calls,
            )
            return self.extract(
                messages=messages,
                schema=schema,
                system_message=system_message,
                temperature=temperature,
                tools=tools,
            )

        return (
            data,
            llm_response,
        )

    @override
    def complete_chat(
        self,
        *,
        messages: list[LLMInputMessage],
        system_message: str = "",
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
        temperature: float = 0,
        tools: LLMTools | None = None,
    ) -> LLMResponse:
        tool_registry = LLMToolRegistry()
        if tools:
            for tool in tools.tools:
                tool_registry.add(tool=tool)

        llm_messages = [
            self._convert_llm_input_message_to_genai_content(message=message)
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
            tool_registry=tool_registry,
        )

        llm_response = self._to_llm_response(response=response)
        print("input_tokens: ", llm_response.prompt_tokens_used)

        if (
            tools
            and llm_response.stop_reason == LLMStopReason.TOOL_USE
            and llm_response.function_calls
            and tools.call_automatically
        ):
            messages = tool_registry.process_tool_calls(
                messages=messages,
                tool_calls=llm_response.function_calls,
            )
            return self.complete_chat(
                messages=messages,
                system_message=system_message,
                output_mode=output_mode,
                temperature=temperature,
                tools=tools,
            )

        return llm_response

    def get_client(self) -> genai.Client:
        return genai.Client(api_key=self.get_api_key())

    def get_gemini_tools(
        self,
        *,
        tool_registry: LLMToolRegistry,
        parallel_tool_calls: bool,
    ) -> list[genai.types.Tool]:
        definitions = tool_registry.definitions

        if not definitions:
            return []

        if parallel_tool_calls:
            return [
                genai.types.Tool(
                    function_declarations=[
                        tool.get_tool_dict()  # type: ignore
                        for tool in tool_registry.definitions
                    ]
                )
            ]

        return [
            genai.types.Tool(
                function_declarations=[
                    tool.model_dump()  # type: ignore
                ]
            )
            for tool in tool_registry.definitions
        ]

    def _get_stop_reason(
        self,
        response_candidate: genai.types.Candidate,
    ) -> LLMStopReason:
        match response_candidate.finish_reason:
            case genai.types.FinishReason.STOP:
                return LLMStopReason.END_TURN
            case genai.types.FinishReason.MAX_TOKENS:
                return LLMStopReason.MAX_TOKENS
            case _:
                raise ValueError(
                    "Unknown finish reason: "
                    + str(response_candidate.finish_reason)
                )

    def _convert_llm_input_message_to_genai_content(
        self, *, message: LLMInputMessage
    ) -> genai.types.Content:
        match message.role:
            case LLMMessageRole.USER:
                role = "user"
            case LLMMessageRole.ASSISTANT:
                role = "model"
            case LLMMessageRole.TOOL_CALL:
                role = "model"
            case LLMMessageRole.TOOL_OUTPUT:
                role = "function"
            case _:
                raise TypeError(f"Unknown role: {message.role}")

        parts: list[genai.types.Part]
        if isinstance(message.content, str):
            parts = [genai.types.Part(text=message.content)]
        elif isinstance(message.content, LLMInputFile):
            image = message.content.file
            parts = [
                genai.types.Part(text=message.content.text),
                genai.types.Part(
                    inline_data=genai.types.Blob(
                        mime_type=image.mime_type,
                        data=image.base64_data,
                    )
                ),
            ]
        elif isinstance(message.content, LLMToolCall):
            tool_call = message.content
            parts = [
                genai.types.Part(
                    function_call=genai.types.FunctionCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        args=tool_call.arguments,
                    )
                )
            ]
        elif isinstance(message.content, LLMToolCallResponse):
            tool_call = message.content.tool_call
            parts = [
                genai.types.Part(
                    function_response=genai.types.FunctionResponse(
                        id=tool_call.id,
                        name=tool_call.name,
                        response={"result": message.content.output},
                    )
                )
            ]
        else:
            raise TypeError(f"Unsupported message type: {message}")

        return genai.types.Content(
            role=role,
            parts=parts,
        )

    def _call(
        self,
        *,
        system_message: str,
        temperature: float,
        contents: list[genai.types.Content],
        response_mime_type: str,
        response_schema: type[PydanticModel] | None = None,
        tool_registry: LLMToolRegistry | None = None,
        parallel_tool_calls: bool = True,
    ) -> genai.types.GenerateContentResponse:
        try:
            response = self._call_llm(
                system_message=system_message,
                temperature=temperature,
                contents=contents,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
                tool_registry=tool_registry,
                parallel_tool_calls=parallel_tool_calls,
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
        *,
        system_message: str,
        temperature: float,
        contents: list[genai.types.Content],
        response_mime_type: str,
        response_schema: type[PydanticModel] | None = None,
        tool_registry: LLMToolRegistry | None = None,
        parallel_tool_calls: bool = True,
    ) -> genai.types.GenerateContentResponse:
        return self.get_client().models.generate_content(
            model=self.model,
            contents=contents,  # type: ignore
            config=genai.types.GenerateContentConfig(
                max_output_tokens=self.token_budget.max_tokens_for_output,
                system_instruction=system_message if system_message else None,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
                temperature=temperature,
                tools=self.get_gemini_tools(
                    parallel_tool_calls=parallel_tool_calls,
                    tool_registry=tool_registry,
                )
                if tool_registry
                else [],  # type: ignore
                automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                **self._get_extra_config(),
            ),
        )

    def _to_llm_response(
        self, response: genai.types.GenerateContentResponse
    ) -> LLMResponse:
        usage = response.usage_metadata

        assert usage is not None, "Usage metadata is not available"

        prompt_tokens_used = usage.prompt_token_count or 0
        completion_tokens_used = usage.candidates_token_count or 0
        cost = self.price_calculator.calculate_price(
            input_tokens=prompt_tokens_used,
            output_tokens=completion_tokens_used,
        )

        if response.function_calls:
            return LLMResponse(
                llm_model=self.get_model(),
                answer="",
                prompt_tokens_used=prompt_tokens_used,
                completion_tokens_used=completion_tokens_used,
                cost=cost,
                stop_reason=LLMStopReason.TOOL_USE,
                function_calls=[
                    LLMToolCall(
                        id=call.id or "call_" + uuid4().hex,
                        name=call.name,
                        arguments=call.args or {},
                    )
                    for call in response.function_calls
                    if call.name
                ],
            )

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

    def _get_extra_config(self) -> dict[str, Any]:
        return {}


class Gemini2_0_Flash(GoogleLLM):  # noqa
    def __init__(
        self,
        api_key: str | APIKeyRotator,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="gemini-2.0-flash",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=0.10,
                output_tokens=0.40,
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
    def get_gemini_tools(
        self,
        *,
        tool_registry: LLMToolRegistry,
        parallel_tool_calls: bool,
    ) -> list[genai.types.Tool]:
        return [
            *super().get_gemini_tools(
                tool_registry=tool_registry,
                parallel_tool_calls=parallel_tool_calls,
            ),
            genai.types.Tool(google_search=genai.types.GoogleSearch()),
        ]


class Gemini2_5_Flash(GoogleLLM):  # noqa
    def __init__(self, api_key: str | APIKeyRotator) -> None:
        super().__init__(
            api_key=api_key,
            model="gemini-2.5-flash",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=0.15,
                output_tokens=0.60,
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=1_000_000,
                max_tokens_for_input=900_000,
                max_tokens_for_output=400_000,
            ),
        )

    def _get_extra_config(self) -> dict[str, Any]:
        data = super()._get_extra_config()
        data.update(
            {
                "thinking_config": genai.types.ThinkingConfig(
                    thinking_budget=1024
                )
            }
        )
        return data


class Gemini20FlashLite(GoogleLLM):
    def __init__(
        self,
        api_key: str,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="gemini-2.0-flash-lite",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=0.075,
                output_tokens=0.30,
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=1_000_000,
                max_tokens_for_input=900_000,
                max_tokens_for_output=400_000,
            ),
        )


class Gemini25FlashLite(GoogleLLM):
    def __init__(
        self,
        api_key: str,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="gemini-2.5-flash-lite",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=0.1,
                output_tokens=0.4,
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=1_000_000,
                max_tokens_for_input=900_000,
                max_tokens_for_output=400_000,
            ),
        )


class Gemma3(GoogleLLM):
    def __init__(
        self,
        api_key: str,
    ) -> None:
        super().__init__(
            model="gemma-3-7b-it",
            api_key=api_key,
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=0,
                output_tokens=0,
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=1_000_000,
                max_tokens_for_input=900_000,
                max_tokens_for_output=400_000,
            ),
        )
