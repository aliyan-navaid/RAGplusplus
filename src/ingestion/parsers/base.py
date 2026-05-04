"""
Abstract base class for document parsers.

Defines the interface that all parsers must implement.
Input: Raw file (PDF, DOCX, etc.)
Output: ParsedDocument with extracted content and metadata
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from ..models import ParsedDocument


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.
    
    Each parser implementation converts a file format (PDF, DOCX, etc.)
    into a standardized ParsedDocument representation.
    
    FLOW:
    Input: PDF file path
    ↓ [parser.parse()]
    Output: ParsedDocument (structured text, tables, metadata)
    """
    
    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Parse a document file and return structured content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ParsedDocument: Structured representation of the document
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        pass
    
    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> Path:
        """Helper to validate that file exists."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path
