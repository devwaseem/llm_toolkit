from abc import ABC, abstractmethod
from decimal import Decimal
from enum import StrEnum
from typing import Any, NamedTuple


class LLMOutputMode(StrEnum):
    TEXT = "TEXT"
    JSON = "JSON"


class LLM(ABC):
    @abstractmethod
    def get_model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def set_system_message(self, *, message: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_message(self, *, message: "LLMInputMessage") -> None:
        raise NotImplementedError

    @abstractmethod
    def complete_chat(
        self,
        *,
        output_mode: LLMOutputMode = LLMOutputMode.TEXT,
    ) -> "LLMResponse":
        raise NotImplementedError


class LLMMessageRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class LLMInputMessage(NamedTuple):
    role: LLMMessageRole
    content: str | dict[str, Any] | list[dict[str, Any]]


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

    def calculate_price(
        self, input_tokens: int, output_tokens: int
    ) -> Decimal:
        cost_of_input_tokens = self.cost_per_input_token() * input_tokens
        cost_of_output_tokens = self.cost_per_output_token() * output_tokens
        return cost_of_input_tokens + cost_of_output_tokens


class LLMTokenBudget:
    llm_max_token: int
    max_tokens_for_context: int
    max_tokens_for_output: int

    def __init__(
        self,
        llm_max_token: int,
        max_tokens_for_context: int,
        max_tokens_for_output: int,
    ) -> None:
        self.llm_max_token = llm_max_token
        self.max_tokens_for_output = max_tokens_for_output
        if max_tokens_for_context > llm_max_token:
            raise ValueError(
                "max_tokens_for_context cannot be larger than llm_max_token"
            )
        self.max_tokens_for_context = max_tokens_for_context
