import base64
import mimetypes
from functools import cached_property
from pathlib import Path


class LLMFileData:
    def __init__(self, *, path: str | Path, fallback_mime_type: str) -> None:
        self.image_path = Path(path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        self.fallback_mime_type: str = fallback_mime_type

    @cached_property
    def mime_type(self) -> str:
        mime_type, _ = mimetypes.guess_type(str(self.image_path))
        return mime_type or self.fallback_mime_type

    @cached_property
    def base64_data(self) -> bytes:
        with self.image_path.open("rb") as f:
            return base64.standard_b64encode(f.read())

    @cached_property
    def base64_data_str(self) -> str:
        return self.base64_data.decode("utf-8")
