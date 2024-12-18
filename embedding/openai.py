import logging
from enum import StrEnum
from typing import Literal

from openai import (
    APIConnectionError,
    APIError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from llm_toolkit.embedding.errors import (
    EmbeddingAPIConnectionError,
    EmbeddingAPIError,
    EmbeddingRateLimitedError,
    EmbeddingServerError,
)
from llm_toolkit.embedding.models import (
    EmbeddingGeneratorInterface,
    EmbeddingResult,
)
from llm_toolkit.token_counter.models import (
    EmbeddingTokenCounterInterface,
)


class OpenAIEmbeddingModels(StrEnum):
    TEXT_ADA_002 = "text-embedding-ada-002"
    TEXT_3_LARGE = "text-embedding-3-large"
    TEXT_3_SMALL = "text-embedding-3-small"


logger = logging.getLogger(__name__)


class OpenAIEmbeddingGenerator(EmbeddingGeneratorInterface):
    def __init__(
        self,
        api_key: str,
        model: str,
        context_limit: int,
        dimensions: Literal[1536, 1024, 3072, None] = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.context_limit = context_limit

    def get_embedding(self, *, text: str) -> EmbeddingResult:
        extra_args = {}
        if self.dimensions is not None:
            extra_args["dimensions"] = self.dimensions
        try:
            response = self.client.embeddings.create(
                **extra_args,  # type: ignore
                input=text,
                model=self.model,
            )
        except RateLimitError as error:
            logger.exception(
                "Rate limited by OpenAI while generating embedding",
            )
            raise EmbeddingRateLimitedError from error

        except APIConnectionError as error:
            raise EmbeddingAPIConnectionError from error

        except InternalServerError as error:
            raise EmbeddingServerError from error

        except APIError as error:
            raise EmbeddingAPIError from error

        return EmbeddingResult(
            embedding=response.data[0].embedding,
            tokens_used=response.usage.total_tokens,
        )

    def get_context_limit(self) -> int:
        return self.context_limit

    def get_model(self) -> str:
        return self.model

    def get_token_counter(self) -> EmbeddingTokenCounterInterface:
        raise NotImplementedError


class OpenAIADAEmbeddingGenerator(OpenAIEmbeddingGenerator):
    def __init__(self, api_key: str) -> None:
        super().__init__(
            model=OpenAIEmbeddingModels.TEXT_ADA_002,
            api_key=api_key,
            context_limit=8191,
        )


class OpenAITextLarge3072EmbeddingGenerator(OpenAIEmbeddingGenerator):
    def __init__(self, api_key: str) -> None:
        super().__init__(
            model=OpenAIEmbeddingModels.TEXT_3_LARGE,
            api_key=api_key,
            dimensions=3072,
            context_limit=8191,
        )


class OpenAITextSmall1536EmbeddingGenerator(OpenAIEmbeddingGenerator):
    def __init__(self, api_key: str) -> None:
        super().__init__(
            model=OpenAIEmbeddingModels.TEXT_3_SMALL,
            api_key=api_key,
            dimensions=1536,
            context_limit=8191,
        )
