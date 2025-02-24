import base64
from functools import cached_property
import mimetypes
from pathlib import Path


class LLMImageData:
    def __init__(
        self,
        *,
        image_path: str | Path,
        fallback_mime_type: str = "image/jpeg",
    ) -> None:
        self.image_path: str = str(image_path)
        self.fallback_mime_type: str = fallback_mime_type

    @cached_property
    def mime_type(self) -> str:
        mime_type, _ = mimetypes.guess_file_type(self.image_path)
        return mime_type or self.fallback_mime_type

    @cached_property
    def base64_data(self) -> str:
        with open(self.image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
