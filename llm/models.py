from abc import ABC, abstractmethod
from decimal import Decimal
from enum import StrEnum
from typing import NamedTuple

from llm_toolkit.models import LLMImageData


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
    ) -> "LLMResponse":
        raise NotImplementedError


class LLMMessageBuilderInterface(ABC):
    @abstractmethod
    def add_base64_image(
        self, *, mime_type: str, content: str
    ) -> "LLMMessageBuilderInterface":
        raise NotImplementedError

    @abstractmethod
    def add_text(self, *, text: str) -> "LLMMessageBuilderInterface":
        raise NotImplementedError

    @abstractmethod
    def build_message(self, role: LLMMessageRole) -> LLMInputMessage:
        raise NotImplementedError


class LLMStopReason(StrEnum):
    MAX_TOKENS = "MAX_TOKENS"
    END_TURN = "END_TURN"
    TOOL_USE = "TOOL_USE"
    STOP_SEQUENCE = "STOP_SEQUENCE"


class LLMResponse(NamedTuple):
    llm_model: str
    answer: str
    prompt_tokens_used: int
    completion_tokens_used: int
    cost: Decimal
    stop_reason: LLMStopReason = LLMStopReason.END_TURN


class LLMPriceCalculator(NamedTuple):
    tokens: int
    input_tokens: Decimal
    output_tokens: Decimal

    def cost_per_input_token(self) -> Decimal:
        return self.input_tokens / max(1, self.tokens)

    def cost_per_output_token(self) -> Decimal:
        return self.output_tokens / max(1, self.tokens)

    def calculate_price(self, input_tokens: int, output_tokens: int) -> Decimal:
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
