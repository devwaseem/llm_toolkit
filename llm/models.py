import logging
from decimal import Decimal
from enum import StrEnum
from typing import Any, NamedTuple, Protocol, Type, TypeVar

from pydantic import BaseModel

from llm_toolkit.models import LLMImageData

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


class LLM(Protocol):
    def get_api_key(self) -> str:
        raise NotImplementedError

    def get_model(self) -> str:
        raise NotImplementedError

    def complete_chat(
        self,
        *,
        messages: list[LLMInputMessage],
        system_message: str = "",
        temperature: float = 0,
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
    ) -> LLMResponse:
        raise NotImplementedError


PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


class StructuredOutputLLM(Protocol):
    def extract(
        self,
        *,
        messages: list[LLMInputMessage],
        schema: Type[PydanticModel],
        system_message: str = "",
        temperature: float = 0,
    ) -> tuple[PydanticModel, LLMResponse]:
        raise NotImplementedError
