class APIKeyRotator:
    def __init__(self, api_key_list: list[str]) -> None:
        self.api_key_list = api_key_list
        self._current_index = 0

    def _rotate(self) -> int:
        next_index = self._current_index
        self._current_index += 1
        return next_index % len(self.api_key_list)

    def get_next_api_key(self) -> str:
        current_index = self.rotator.rotate()
        return self.api_key_list[current_index]
