import logging
from collections.abc import Generator
from typing import Any
import io

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.file.file import File
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class ToolParameters(BaseModel):
    files: list[File]


class Pdf2imageTool(Tool):
    """
    A tool for converting PDF files to images using PyMuPDF and Pillow
    """

    def _invoke(
            self, tool_parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        if tool_parameters.get("files") is None:
            yield self.create_text_message("No files provided. Please upload PDF files for processing.")
            return

        params = ToolParameters(**tool_parameters)
        files = params.files

        try:
            # Try both import methods to ensure compatibility
            try:
                import pymupdf
                fitz_module = pymupdf
            except ImportError:
                import fitz
                fitz_module = fitz

            try:
                from PIL import Image
            except ImportError:
                error_msg = "Error: Pillow library not installed. Please install it with 'pip install Pillow'."
                logger.error(error_msg)
                yield self.create_text_message(error_msg)
                return

            for file in files:
                try:
                    logger.info(f"Processing file: {file.filename}")

                    # Process PDF file
                    file_bytes = io.BytesIO(file.blob)
                    doc = fitz_module.open(stream=file_bytes, filetype="pdf")

                    page_count = doc.page_count
                    images = []

                    # Convert each page to an image
                    for page_num in range(page_count):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        images.append(img)

                    # Close the document to free resources
                    doc.close()

                    if not images:
                        yield self.create_text_message(f"No pages found in {file.filename}")
                        continue

                    # Merge all images vertically
                    total_width = max(img.width for img in images)
                    total_height = sum(img.height for img in images)
                    combined_image = Image.new('RGB', (total_width, total_height))

                    y_offset = 0
                    for img in images:
                        combined_image.paste(img, (0, y_offset))
                        y_offset += img.height

                    # Save the combined image to a bytes buffer
                    img_buffer = io.BytesIO()
                    combined_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    image_bytes = img_buffer.getvalue()

                    # Yield image as blob with mime type
                    yield self.create_blob_message(
                        image_bytes,
                        meta={
                            "mime_type": "image/png",
                            "filename": f"{file.filename.rsplit('.', 1)[0]}.png"
                        },
                    )

                except Exception as e:
                    error_msg = f"Error processing {file.filename}: {str(e)}"
                    logger.error(error_msg)
                    yield self.create_text_message(error_msg)
                    yield self.create_json_message({
                        file.filename: {"error": str(e)}
                    })

        except ImportError as e:
            error_msg = f"Error: Required library not installed. {str(e)}"
            logger.error(error_msg)
            yield self.create_text_message(error_msg)
