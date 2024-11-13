from typing import Protocol

from llm_toolkit.llm.models import LLMResponse


class LLMResponseCache(Protocol):
    def get(self, key: str) -> LLMResponse | None:
        ...

    def set(self, key: str, value: LLMResponse) -> None:
        ...


class LLMInMemoryCache(LLMResponseCache):
    def __init__(self) -> None:
        self.cache: dict[str, LLMResponse] = {}

    def get(self, key: str) -> LLMResponse | None:
        return self.cache.get(key)

    def set(self, key: str, value: LLMResponse) -> None:
        self.cache[key] = value
