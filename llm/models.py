import json
import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, NamedTuple, Type, TypeVar, cast

from pydantic import BaseModel

from llm_toolkit.models import LLMImageData
from llm_toolkit.schema_generator.models import (
    LLMSchemaGenerator,
    LLMSchemaModel,
)

logger = logging.getLogger(__name__)


class LLMOutputMode(StrEnum):
    TEXT = "TEXT"
    JSON = "JSON"


class LLMMessageRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class LLMInputImage(NamedTuple):
    text: str
    image: LLMImageData


class LLMInputMessage(NamedTuple):
    role: LLMMessageRole
    content: str | LLMInputImage


PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


class LLM(ABC):
    @abstractmethod
    def get_model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, *, text: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def truncate_text_to_max_tokens(
        self,
        *,
        text: str,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete_chat(
        self,
        *,
        messages: list[LLMInputMessage],
        system_message: str = "",
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
        tools: list[dict[str, Any]] | None = None,
    ) -> "LLMResponse":
        raise NotImplementedError


class LLMStopReason(StrEnum):
    MAX_TOKENS = "MAX_TOKENS"
    END_TURN = "END_TURN"
    TOOL_USE = "TOOL_USE"
    STOP_SEQUENCE = "STOP_SEQUENCE"


class LLMFunctionCall(NamedTuple):
    name: str
    arguments: dict[str, Any]


class LLMResponse(NamedTuple):
    llm_model: str
    answer: str
    prompt_tokens_used: int
    completion_tokens_used: int
    cost: Decimal
    function_call: LLMFunctionCall | None = None
    stop_reason: LLMStopReason = LLMStopReason.END_TURN


class LLMPriceCalculator(NamedTuple):
    tokens: int
    input_tokens: Decimal
    output_tokens: Decimal

    def cost_per_input_token(self) -> Decimal:
        return self.input_tokens / max(1, self.tokens)

    def cost_per_output_token(self) -> Decimal:
        return self.output_tokens / max(1, self.tokens)

    def calculate_price(
        self, input_tokens: int, output_tokens: int
    ) -> Decimal:
        cost_of_input_tokens = self.cost_per_input_token() * input_tokens
        cost_of_output_tokens = self.cost_per_output_token() * output_tokens
        return cost_of_input_tokens + cost_of_output_tokens


class LLMTokenBudget:
    llm_max_token: int
    max_tokens_for_context: int
    _max_tokens_for_output: int | None

    def __init__(
        self,
        llm_max_token: int,
        max_tokens_for_input: int,
        max_tokens_for_output: int | None = None,
    ) -> None:
        self.llm_max_token = llm_max_token
        self._max_tokens_for_output = max_tokens_for_output
        if max_tokens_for_input > llm_max_token:
            raise ValueError(
                "max_tokens_for_context cannot be larger than llm_max_token"
            )
        self.max_tokens_for_context = max_tokens_for_input

    @property
    def max_tokens_for_output(self) -> int:
        if not self._max_tokens_for_output:
            return self.llm_max_token - self.max_tokens_for_context
        return self._max_tokens_for_output


class StructuredOutputLLM:
    @abstractmethod
    def extract(
        self,
        *,
        messages: list[LLMInputMessage],
        schema: Type[PydanticModel],
        system_message: str = "",
    ) -> tuple[PydanticModel, LLMResponse]:
        raise NotImplementedError


T = TypeVar("T", bound=LLMSchemaModel)


class LLMExtractedImageData(NamedTuple, Generic[T]):
    schema: T
    llm_response: LLMResponse


class ImageDataExtractorLLM(LLM):
    def extract_image_data(
        self,
        *,
        system_message: str,
        image_file: Path,
        schema_generator: LLMSchemaGenerator[T],
        pre_image_llm_messages: list[LLMInputMessage] | None = None,
        post_image_llm_messages: list[LLMInputMessage] | None = None,
    ) -> LLMExtractedImageData[T]:
        schema_dict = schema_generator.build_schema()

        llm_messages = pre_image_llm_messages or []

        llm_messages.append(
            LLMInputMessage(
                role=LLMMessageRole.USER,
                content=LLMInputImage(
                    image=LLMImageData(
                        image_path=str(image_file),
                    ),
                    text=json.dumps(schema_dict),
                ),
            )
        )

        if post_image_llm_messages:
            llm_messages.extend(post_image_llm_messages)

        llm_response = self.complete_chat(
            system_message=(system_message + schema_generator.get_example()),
            messages=llm_messages,
            output_mode=LLMOutputMode.JSON,
        )

        try:
            json_data = cast(
                dict[str, Any], json.loads(str(llm_response.answer))
            )
        except json.JSONDecodeError as exc:
            logger.exception(
                "Invalid JSON returned by LLM: %s",
                llm_response.answer,
            )
            raise exc from exc

        return LLMExtractedImageData(
            schema=schema_generator.schema(
                data=(
                    schema_generator.decode_json(data=json_data)
                    if schema_generator.encoded
                    else json_data
                )
            ),
            llm_response=llm_response,
        )
