from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentExtractionError(Exception):
    """Raised when extraction fails irrecoverably for a document."""

    def __init__(self, file_path: Path, errors: list):
        self.file_path = file_path
        self.errors = errors
        super().__init__(f"Extraction failed for {file_path}")


def build_converter() -> DocumentConverter:
    """
    Construct a converter with formats explicitly allowed.

    Table structure is preserved by default. OCR is left off for
    Phase 1 — born-digital PDFs only. Enable it in Phase 2.
    """

    pdf_options = PdfPipelineOptions(
        generate_picture_images=False,
        document_timeout=120.0,
        do_ocr=False,
        do_table_structure=True,
        do_code_enrichment=False,
        do_formula_enrichment=False,
    )
    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.XLSX,
            InputFormat.HTML,
            InputFormat.MD,
            InputFormat.ASCIIDOC,
            InputFormat.IMAGE,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        },
    )


def extract_document(
    source: str | Path,
) -> DoclingDocument:
    """
    Extract a DoclingDocument from a file.

    Raises:
        FileNotFoundError: source does not exist.
        DocumentExtractionError: conversion failed or was skipped.
    """
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Document not found: {source_path.resolve()}")

    conv = build_converter()
    result = conv.convert(source_path, raises_on_error=False)

    if result.status == ConversionStatus.SUCCESS:
        logger.info(
            "extraction_success",
            extra={"file": str(source_path), "pages": len(result.pages or [])},
        )
        return result.document

    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        logger.warning(
            "extraction_partial_success",
            extra={
                "file": str(source_path),
                "errors": [e.error_message for e in result.errors],
            },
        )
        return result.document

    # SKIPPED or FAILURE
    logger.error(
        "extraction_failed",
        extra={
            "file": str(source_path),
            "status": result.status.value,
            "errors": [
                {
                    "component": e.component_type.value,
                    "module": e.module_name,
                    "message": e.error_message,
                }
                for e in result.errors
            ],
        },
    )
    raise DocumentExtractionError(source_path, result.errors)
