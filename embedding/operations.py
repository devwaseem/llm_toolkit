import string
from hashlib import md5
from typing import cast

from django.core.cache import cache

from .models import (
    EmbeddingGeneratorInterface,
    EmbeddingResult,
)


def get_embedding(
    *,
    embedding_generator: EmbeddingGeneratorInterface,
    text: str,
) -> EmbeddingResult:
    text = text.replace("\n", " ").lower()
    text = text.translate(
        str.maketrans({key: None for key in string.punctuation})  # type: ignore
    )
    md5_hash = md5(text.encode("utf-8")).hexdigest()

    cache_key = "embedding:" + md5_hash + embedding_generator.__class__.__name__
    if cached_data := cache.get(cache_key):
        return cast(EmbeddingResult, cached_data)

    embedding_result = embedding_generator.get_embedding(text=text)
    cache.set(cache_key, embedding_result)
    return embedding_result
