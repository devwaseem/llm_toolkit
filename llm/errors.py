class LLMJsonResponseDecodingError(Exception):
    ...


class LLMRateLimitedError(Exception):
    ...


class LLMAPIConnectionError(Exception):
    ...


class LLMInternalServerError(Exception):
    ...


class LLMAPITimeoutError(Exception):
    ...


class LLMAuthenticationError(Exception):
    ...


class LLMPermissionDeniedError(Exception):
    ...