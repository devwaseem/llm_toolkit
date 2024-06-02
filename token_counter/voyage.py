from typing import cast

import voyageai

from .models import (
    EmbeddingTokenCounterInterface,
)


class VoyageEmbeddingTokenCounter(EmbeddingTokenCounterInterface):
    def __init__(self) -> None:
        self.client = voyageai.Client(api_key=None)

    def count_tokens(self, text: str) -> int:
        return cast(int, self.client.count_tokens(texts=[text]))
