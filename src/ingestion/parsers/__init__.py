"""
Parsers module for converting different document formats to structured data.

This module provides:
- Base parser interface following strategy pattern
- Docling/Nougat PDF parser implementation
- Parser factory for creating appropriate parsers
"""

from .base import BaseParser
from .docling_parser import DoclingParser

__all__ = ["BaseParser", "DoclingParser"]
