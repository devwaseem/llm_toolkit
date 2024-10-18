from pathlib import Path
from typing import Generator, NamedTuple

try:
    import pymupdf
except ImportError as exc:
    raise ImportError(
        "PyMuPDF is required to extract images from PDF. "
        "Please install it using `pip install pymupdf`"
        "or poetry install pymupdf"
    ) from exc


class ExtractedImage(NamedTuple):
    image_path: str
    order: int


def extract_images_from_pdf(
    *,
    file_path: str,
    out_dir: str,
    page_prefix: str = "page",
    grayscale: bool = False,
) -> Generator[Path, None, None]:
    doc = pymupdf.open(file_path)
    for page in doc:
        pix = page.get_pixmap(dpi=150)  # type: ignore
        page_number = page.number
        image_path = Path(out_dir) / f"{page_prefix}_{page_number}.jpeg"
        pix.save(image_path)
        if grayscale:
            try:
                from PIL import Image
            except ImportError as exc:
                raise ImportError(
                    "Pillow is required to convert images to grayscale. "
                    "Please install it using `pip install Pillow` "
                    "or poetry install Pillow"
                ) from exc

            image = Image.open(image_path)
            image.convert("L")
            image.save(image_path)
        yield image_path
