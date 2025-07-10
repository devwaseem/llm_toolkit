from hashlib import md5

from llm_toolkit.api_key_rotator.models import APIKeyRotator

try:
    import redis
except ImportError as exc:
    raise ImportError(
        "Please install redis to use RedisBasedAPIKeyRotator"
    ) from exc

import logging

logger = logging.getLogger(__name__)


class RedisBasedAPIKeyRotator(APIKeyRotator):
    key_prefix = "api_key_rotator"

    def __init__(
        self,
        api_key_list: list[str],
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
    ) -> None:
        super().__init__(api_key_list=api_key_list)
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
        )
        self.cache_key = (
            self.key_prefix
            + md5(
                str(api_key_list).encode(),
                usedforsecurity=False,
            ).hexdigest()
        )

    def _rotate_index(self) -> int:
        current_index = self.redis_client.get(self.cache_key)
        if current_index is None:
            self.redis_client.set(self.cache_key, 0)
            current_index = 0
        self.redis_client.incrby(self.cache_key, 1)
        return int(current_index)

    def get_next_api_key(self) -> str:
        index = self._rotate_index()
        api_key = self.api_key_list[index % len(self.api_key_list)]
        logger.debug(
            "[RedisBasedAPIKeyRotator] Using API Key Index: %d", index
        )
        return api_key
