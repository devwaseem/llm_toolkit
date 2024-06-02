from abc import ABC, abstractmethod
from typing import NamedTuple

from ..token_counter.models import (
    EmbeddingTokenCounterInterface,
)


class EmbeddingRateLimitedError(Exception): ...


class EmbeddingAPIConnectionError(Exception): ...


class EmbeddingAPIError(Exception): ...


class EmbeddingServerError(Exception): ...


class EmbeddingResult(NamedTuple):
    embedding: list[float]
    tokens_used: int


class EmbeddingGeneratorInterface(ABC):
    @abstractmethod
    def get_embedding(self, *, text: str) -> EmbeddingResult:
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_context_limit(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_token_counter(self) -> EmbeddingTokenCounterInterface:
        raise NotImplementedError
