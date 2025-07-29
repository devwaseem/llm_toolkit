from abc import ABC, abstractmethod
from typing import Any


class ReRanker[T: Any](ABC):
    @abstractmethod
    def rerank(self) -> list[T]:
        raise NotImplementedError
