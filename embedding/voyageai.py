import logging
from typing import Literal

import voyageai
from voyageai import error as voyageai_error

from .models import (
    EmbeddingAPIConnectionError,
    EmbeddingAPIError,
    EmbeddingGeneratorInterface,
    EmbeddingRateLimitedError,
    EmbeddingResult,
    EmbeddingServerError,
)
from ..token_counter.models import (
    EmbeddingTokenCounterInterface,
)
from ..token_counter.voyage import (
    VoyageEmbeddingTokenCounter,
)
from ..voyage import VoyageEmbeddingModel

logger = logging.getLogger(__name__)


class VoyageAIEmbeddingGenerator(EmbeddingGeneratorInterface):
    def __init__(
        self,
        *,
        api_key: str,
        model: VoyageEmbeddingModel,
        context_limit: int,
        dimensions: Literal[1536, 1024],
    ) -> None:
        self.client = voyageai.Client(
            api_key=api_key,
            max_retries=3,
        )
        self.model = model
        self.dimensions = dimensions
        self.context_limit = context_limit

    def get_embedding(self, *, text: str) -> EmbeddingResult:
        try:
            response = self.client.embed([text], model=self.model)
        except voyageai_error.RateLimitError as error:
            logger.exception("Rate limited by Voyage AI while generating embedding")
            raise EmbeddingRateLimitedError from error
        except voyageai.error.APIConnectionError as error:
            raise EmbeddingAPIConnectionError from error
        except voyageai.error.APIError as error:
            raise EmbeddingAPIError from error
        except voyageai.error.ServerError as error:
            raise EmbeddingServerError from error

        return EmbeddingResult(
            embedding=response.embeddings[0],
            tokens_used=response.total_tokens,
        )

    def get_context_limit(self) -> int:
        return self.context_limit

    def get_model(self) -> str:
        return self.model

    def get_token_counter(self) -> EmbeddingTokenCounterInterface:
        return VoyageEmbeddingTokenCounter()


class VoyageAILaw2EmbeddingGenerator(VoyageAIEmbeddingGenerator):
    def __init__(self, *, api_key: str) -> None:
        super().__init__(
            api_key=api_key,
            model=VoyageEmbeddingModel.LAW_2,
            dimensions=1024,
            context_limit=16000,
        )
