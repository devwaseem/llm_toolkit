from abc import ABC, abstractmethod
from enum import StrEnum
from typing import NamedTuple


class LLMRateLimitedError(Exception):
    ...


class LLM(ABC):
    @abstractmethod
    def get_model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete_chat(
        self, *, system_message: str, message_list: list["LLMMessage"]
    ) -> "LLMResponse":
        raise NotImplementedError


class LLMMessageRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class LLMMessage(NamedTuple):
    role: LLMMessageRole
    content: str


class LLMResponse(NamedTuple):
    llm_model: str
    answer: LLMMessage
    prompt_tokens_used: int
    completion_tokens_used: int
    price: float


class LLMPrice(NamedTuple):
    tokens: int
    input_tokens: float
    output_tokens: float

    def calculate_price(self, input_tokens: int, output_tokens: int) -> float:
        input_price = (self.input_tokens / max(self.tokens, 1)) * input_tokens
        output_price = (
            self.output_tokens / max(self.tokens, 1)
        ) * output_tokens
        return input_price + output_price


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
