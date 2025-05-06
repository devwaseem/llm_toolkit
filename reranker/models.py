from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class ReRanker(ABC, Generic[T]):
    @abstractmethod
    def rerank(self) -> list[T]:
        raise NotImplementedError
