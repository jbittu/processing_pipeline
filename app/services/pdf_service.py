import fitz  # PyMuPDF
from dataclasses import dataclass
from app.utils.helpers import (
    image_bytes_to_base64,
    validate_pdf_header,
    resize_image_if_needed,
)


@dataclass
class PageImage:
    page_number: int          # 1-indexed
    image_bytes: bytes        # raw PNG bytes
    base64_image: str         # base64-encoded for LLM
    width: int
    height: int


class PDFService:
    """Handles PDF validation, page splitting, and image conversion."""

    def __init__(self, dpi: int = 200, max_dimension: int = 2048):
        self.dpi = dpi
        self.max_dimension = max_dimension

    def validate(self, file_bytes: bytes, max_size_mb: int = 20) -> None:
        if not validate_pdf_header(file_bytes):
            raise ValueError("Invalid file: not a PDF")
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(
                f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)"
            )

    def split_to_page_images(self, file_bytes: bytes) -> list[PageImage]:
        """Convert each PDF page into a PNG image."""
        self.validate(file_bytes)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []

        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            zoom = self.dpi / 72  # 72 is default PDF DPI
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix)
            image_bytes = pixmap.tobytes("png")

            image_bytes = resize_image_if_needed(
                image_bytes, self.max_dimension
            )
            b64 = image_bytes_to_base64(image_bytes)

            pages.append(
                PageImage(
                    page_number=page_idx + 1,
                    image_bytes=image_bytes,
                    base64_image=b64,
                    width=pixmap.width,
                    height=pixmap.height,
                )
            )

        doc.close()
        return pages

    def extract_pages_subset(
        self, all_pages: list[PageImage], page_numbers: list[int]
    ) -> list[PageImage]:
        """Return only the requested pages (for routing to agents)."""
        return [p for p in all_pages if p.page_number in page_numbers]
