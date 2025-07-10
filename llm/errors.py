class LLMJsonResponseDecodingError(Exception): ...


class LLMRateLimitedError(Exception):
    def __init__(self, retry_after: int | None) -> None:
        self.retry_after = retry_after

    def __str__(self) -> str:
        if self.retry_after is None:
            return "Rate limited"
        return f"Rate limited, retry after {self.retry_after} seconds"


class LLMAPIConnectionError(Exception): ...


class LLMInternalServerError(Exception): ...


class LLMAPITimeoutError(Exception): ...


class LLMAuthenticationError(Exception): ...


class LLMPermissionDeniedError(Exception): ...


class LLMEmptyResponseError(Exception): ...
