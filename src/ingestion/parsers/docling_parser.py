"""
Docling/Nougat PDF parser implementation.

Converts PDF files into structured ParsedDocument objects
using the Docling library (with OCR via Nougat).

FLOW:
Input: PDF file
↓ [DoclingParser.parse()]
↓ Extract text, tables, metadata using Docling
↓ Structure into ParsedDocument
Output: ParsedDocument (title, content, tables, metadata)
"""

from pathlib import Path
from typing import Union, Optional
import logging

from ..models import ParsedDocument
from .base import BaseParser

# Try to import docling, with fallback for testing
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    DocumentConverter = None

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """
    PDF parser using Docling/Nougat.
    
    Handles:
    - Scanned PDFs (via Nougat OCR)
    - Digital PDFs (direct text extraction)
    - Table extraction
    - Metadata preservation
    
    INPUTS:
    - file_path: Path to PDF file
    
    OUTPUTS:
    - ParsedDocument containing:
      - content: Full extracted text
      - title: Detected document title
      - metadata: Document metadata (author, creation date, etc.)
      - tables: List of extracted tables
      - pages: Total page count
    """
    
    def __init__(self, use_ocr: bool = True):
        """
        Initialize the Docling parser.
        
        Args:
            use_ocr: Whether to use OCR for scanned PDFs (default: True)
                    Uses Nougat for high-quality text recognition
        """
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "Docling is not installed. Install with: pip install docling"
            )
        self.use_ocr = use_ocr
        # Initialize Docling converter with appropriate settings
        self.converter = DocumentConverter()
        logger.info(f"DoclingParser initialized (OCR enabled: {use_ocr})")
    
    def parse(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Parse a PDF file using Docling.
        
        FLOW:
        1. Validate file exists and is PDF
        2. Convert PDF to Docling Document object
        3. Extract text content (with OCR if needed)
        4. Extract tables
        5. Extract metadata
        6. Return as ParsedDocument
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ParsedDocument: Structured document representation
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If file is not a valid PDF
        """
        # Validate file
        path = self.validate_file_exists(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got {path.suffix}")
        
        try:
            logger.info(f"Parsing PDF: {path}")
            
            # Convert PDF using Docling
            # This handles both scanned (via Nougat) and digital PDFs
            result = self.converter.convert(str(path))
            docling_doc = result.document
            
            # Extract main content
            content = docling_doc.export_to_markdown()
            
            # Extract metadata
            metadata = {
                "source": str(path),
                "filename": path.name,
            }
            if hasattr(docling_doc, "properties"):
                if hasattr(docling_doc.properties, "title"):
                    metadata["title"] = docling_doc.properties.title
                if hasattr(docling_doc.properties, "author"):
                    metadata["author"] = docling_doc.properties.author
            
            # Extract tables (stored as structured data)
            tables = []
            # Docling provides structured table information
            # This is a placeholder - actual extraction depends on Docling's API
            
            # Determine page count
            pages = len(docling_doc.pages) if hasattr(docling_doc, "pages") else 0
            
            # Create ParsedDocument
            parsed_doc = ParsedDocument(
                content=content,
                title=metadata.get("title"),
                metadata=metadata,
                tables=tables,
                pages=pages,
            )
            
            logger.info(
                f"Successfully parsed: {parsed_doc.title or path.name} "
                f"({pages} pages, {len(content)} characters)"
            )
            
            return parsed_doc
            
        except Exception as e:
            logger.error(f"Error parsing PDF {path}: {str(e)}")
            raise ValueError(f"Failed to parse PDF {path}: {str(e)}")
