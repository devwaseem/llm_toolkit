import base64
import mimetypes
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class LLMFileData(BaseModel):
    image_path: Path
    fallback_mime_type: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.image_path.exists():
            raise ValueError(f"File {self.image_path} does not exist")

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

    def __repr__(self) -> str:
        return (
            f"LLMFileData(path='{self.image_path.name}', "
            f"mime_type='{self.mime_type}')"
        )

    def __dict__(self) -> dict[str, Any]:
        return {
            "path": str(self.image_path),
            "fallback_mime_type": self.fallback_mime_type,
        }
