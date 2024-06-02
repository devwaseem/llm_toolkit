from abc import ABC, abstractmethod


class EmbeddingTokenCounterInterface(ABC):
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError
