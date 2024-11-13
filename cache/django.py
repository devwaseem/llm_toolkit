from llm_toolkit.cache.models import LLMResponseCache
from llm_toolkit.llm.models import LLMResponse

try:
    from django.core.cache import cache
except ImportError as exc:
    raise ImportError(
        "Could not import django.core.cache.cache. "
        "Please install django to use this cache."
    ) from exc


class DjangoLLMResponseCache(LLMResponseCache):
    def get(self, key: str) -> LLMResponse | None:
        if not cache.has_key(key):
            return None

        return cache.get(key)

    def set(self, key: str, value: LLMResponse) -> None:
        cache.set(key, value)
