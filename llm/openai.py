import json
import logging
from typing import Any, Literal, cast, override

import openai
from openai import AzureOpenAI, OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    ParsedResponse,
    ResponseCustomToolCallOutputParam,
    ResponseCustomToolCallParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseInputTextParam,
)
from openai.types.responses.response import Response
from openai.types.shared.responses_model import ResponsesModel

from llm_toolkit.api_key_rotator.models import APIKeyRotator
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
    LLMInputFile,
    LLMInputMessage,
    LLMMessageRole,
    LLMOutputMode,
    LLMPriceCalculator,
    LLMResponse,
    LLMStopReason,
    LLMTokenBudget,
    LLMToolCallRequest,
    LLMToolCallResponse,
    LLMToolRegistry,
    LLMTools,
    PydanticModel,
    StructuredOutputLLM,
)

logger = logging.getLogger(__name__)


class OpenAILLM(LLM, StructuredOutputLLM):
    _api_key: str | APIKeyRotator
    _model: ResponsesModel
    _client: OpenAI

    token_budget: LLMTokenBudget
    price_calculator: LLMPriceCalculator

    def __init__(
        self,
        *,
        api_key: str | APIKeyRotator,
        model: ResponsesModel,
        token_budget: LLMTokenBudget,
        price_calculator: LLMPriceCalculator,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self.price_calculator = price_calculator
        self.token_budget = token_budget

    def get_client(self) -> OpenAI:
        return OpenAI(api_key=self.get_api_key())

    @override
    def get_api_key(self) -> str:
        if isinstance(self._api_key, APIKeyRotator):
            return self._api_key.get_next_api_key()
        return self._api_key

    @override
    def get_model(self) -> str:
        return self._model

    @override
    def extract(
        self,
        *,
        messages: list[LLMInputMessage],
        schema: type[PydanticModel],
        system_message: str = "",
        temperature: float = 0,
        tools: LLMTools | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[PydanticModel | None, LLMResponse]:
        tool_registry = LLMToolRegistry()
        if tools:
            for tool in tools.tools:
                tool_registry.add(tool=tool)

        response = cast(
            ParsedResponse[PydanticModel],
            self._call(
                system_message=system_message,
                temperature=temperature,
                messages=messages,
                schema=schema,
                text={},
                tool_registry=tool_registry,
            ),
        )

        data = response.output_parsed
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
                metadata=metadata or {},
            )
            return self.extract(
                messages=messages,
                schema=schema,
                system_message=system_message,
                temperature=temperature,
                tools=tools,
                metadata=metadata,
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
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        tool_registry = LLMToolRegistry()
        if tools:
            for tool in tools.tools:
                tool_registry.add(tool=tool)

        output_type = "text"
        if output_mode == LLMOutputMode.JSON:
            output_type = "json_object"

        response = self._call(
            system_message=system_message,
            temperature=temperature,
            messages=messages,
            text={
                "format": {
                    "type": output_type,
                }
            },
        )

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
                metadata=metadata or {},
            )
            return self.complete_chat(
                messages=messages,
                system_message=system_message,
                output_mode=output_mode,
                temperature=temperature,
                tools=tools,
                metadata=metadata,
            )

        return llm_response

    def _call(
        self,
        *,
        system_message: str,
        temperature: float,
        messages: list[LLMInputMessage],
        text: dict[str, Any],
        schema: type[PydanticModel] | None = None,
        tool_registry: LLMToolRegistry | None = None,
        parallel_tool_calls: bool = True,
    ) -> Response | ParsedResponse[Any]:
        llm_input: ResponseInputParam = []
        if system_message:
            llm_input.append(
                EasyInputMessageParam(
                    role="system",
                    content=system_message,
                )
            )

        llm_input += [
            self._convert_llm_input_message_to_raw_message(message=message)
            for message in messages
        ]

        model = self.get_model()

        logger.debug(
            "Calling OpenAI LLM, model: %s, temperature: %s",
            model,
            temperature,
        )

        tools = (
            [
                {"type": "function", **tool.get_tool_dict()}
                for tool in tool_registry.definitions
            ]
            if tool_registry
            else []
        )

        try:
            if schema:
                return self.get_client().responses.parse(
                    model=model,
                    temperature=temperature,
                    input=llm_input,
                    parallel_tool_calls=parallel_tool_calls,
                    tools=tools,  # type: ignore
                    text_format=schema,
                )

            return self.get_client().responses.create(  # type: ignore
                model=model,
                temperature=temperature,
                input=llm_input,
                parallel_tool_calls=parallel_tool_calls,
                tools=tools,
                text=text,  # type: ignore
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

    def _to_llm_response(
        self, *, response: Response | ParsedResponse[Any]
    ) -> LLMResponse:
        stop_reason = LLMStopReason.END_TURN
        if incomplete_details := response.incomplete_details:
            match incomplete_details.reason:
                case "max_output_tokens":
                    stop_reason = LLMStopReason.MAX_TOKENS
                case _:
                    raise NotImplementedError

        if response.usage is None:
            raise ValueError("Usage metadata is not available")

        prompt_tokens_used = response.usage.input_tokens
        completion_tokens_used = response.usage.output_tokens
        cost = self.price_calculator.calculate_price(
            input_tokens=prompt_tokens_used,
            output_tokens=completion_tokens_used,
        )

        tool_calls = []
        answer = ""
        for output in response.output:
            if output.type == "function_call":
                tool_calls.append(
                    LLMToolCallRequest(
                        id=output.call_id,
                        name=output.name,
                        arguments=json.loads(output.arguments),
                    )
                )
            elif output.type == "message":
                for message in output.content:
                    if message.type == "output_text":
                        answer += message.text

        if tool_calls:
            return LLMResponse(
                llm_model=self.get_model(),
                answer=answer,
                prompt_tokens_used=prompt_tokens_used,
                completion_tokens_used=completion_tokens_used,
                cost=float(cost),
                stop_reason=LLMStopReason.TOOL_USE,
                function_calls=tool_calls,
            )

        return LLMResponse(
            llm_model=self.get_model(),
            answer=answer,
            prompt_tokens_used=prompt_tokens_used,
            completion_tokens_used=completion_tokens_used,
            cost=float(cost),
            stop_reason=stop_reason,
        )

    def _convert_llm_input_message_to_raw_message(
        self, message: LLMInputMessage
    ) -> ResponseInputItemParam:
        role: Literal["user", "assistant", "system", "developer"]
        match message.role:
            case LLMMessageRole.USER:
                role = "user"
            case LLMMessageRole.ASSISTANT:
                role = "assistant"
            case LLMMessageRole.TOOL_CALL:
                role = "assistant"
            case LLMMessageRole.TOOL_OUTPUT:
                role = "user"
            case _:
                raise TypeError(f"Unknown role: {message.role}")

        if isinstance(message.content, str):
            return EasyInputMessageParam(role=role, content=message.content)

        if isinstance(message.content, LLMInputFile):
            image = message.content.file
            return EasyInputMessageParam(
                role=role,
                content=[
                    ResponseInputTextParam(
                        type="input_text",
                        text=message.content.text,
                    ),
                    ResponseInputImageParam(
                        detail="auto",
                        type="input_image",
                        image_url=(
                            f"data:{image.mime_type};base64,{image.base64_data_str}"
                        ),
                    ),
                ],
            )

        if isinstance(message.content, LLMToolCallRequest):
            tool_call = message.content
            return ResponseCustomToolCallParam(
                type="custom_tool_call",
                call_id=tool_call.id,
                name=tool_call.name,
                input=json.dumps(tool_call.arguments),
            )

        if isinstance(message.content, LLMToolCallResponse):
            tool_call = message.content.tool_call
            return ResponseCustomToolCallOutputParam(
                type="custom_tool_call_output",
                call_id=tool_call.id,
                output=message.content.output,
            )

        raise TypeError(f"Unhandled message type: {message}")


class AzureOpenAILLM(OpenAILLM):
    def __init__(
        self,
        *,
        api_key: str,
        api_version: str,
        endpoint: str,
        deployment_name: str,
        token_budget: LLMTokenBudget,
        price_calculator: LLMPriceCalculator,
    ) -> None:
        self.api_version = api_version
        self.endpoint = endpoint
        super().__init__(
            api_key=api_key,
            model=deployment_name,
            token_budget=token_budget,
            price_calculator=price_calculator,
        )

    @override
    def get_client(self) -> OpenAI:
        return AzureOpenAI(
            api_key=self.get_api_key(),
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )


class GPT35TurboLLM(OpenAILLM):
    def __init__(
        self,
        api_key: str,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="gpt-3.5-turbo",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=0.50,
                output_tokens=1.50,
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=16_385,
                max_tokens_for_input=12000,
                max_tokens_for_output=4000,
            ),
        )


class GPT4oLLM(OpenAILLM):
    def __init__(
        self,
        api_key: str,
        model_suffix: str = "",
    ) -> None:
        super().__init__(
            model="gpt-4o" + model_suffix,
            api_key=api_key,
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=5.0,
                output_tokens=15.0,
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=128_000,
                max_tokens_for_input=124_00,
                max_tokens_for_output=4_000,
            ),
        )


class GPT5Nano(OpenAILLM):
    def __init__(
        self,
        api_key: str,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="gpt-5-nano",
            price_calculator=LLMPriceCalculator(
                tokens=1_000_000,
                input_tokens=0.05,
                output_tokens=0.45,
            ),
            token_budget=LLMTokenBudget(
                llm_max_token=128_000,
                max_tokens_for_input=124_00,
                max_tokens_for_output=4_000,
            ),
        )
