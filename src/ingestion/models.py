"""
Data models for the ingestion pipeline.

These models represent the data flowing through the pipeline:
- ParsedDocument: Raw output from PDF parser (Docling)
- ConvertedDocument: Markdown-formatted output ready for storage
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ParsedDocument:
    """
    Represents a document parsed from PDF using Docling/Nougat.
    
    This is the initial output from the parser containing:
    - Raw extracted text and metadata
    - Extracted tables and structured data
    - Document structure information
    """
    content: str  # Raw extracted text content
    title: Optional[str] = None  # Document title if detected
    metadata: Dict[str, Any] = field(default_factory=dict)  # Raw metadata (author, creation date, etc.)
    tables: List[Dict[str, Any]] = field(default_factory=list)  # Extracted tables
    pages: int = 0  # Total number of pages
    parsed_at: datetime = field(default_factory=datetime.now)  # Parsing timestamp
    
    def __repr__(self) -> str:
        return f"ParsedDocument(title={self.title}, pages={self.pages}, content_len={len(self.content)})"


@dataclass
class ConvertedDocument:
    """
    Represents a document converted to Markdown format.
    
    This is the output from markdown conversion, ready for:
    - Saving to RAG (vector database)
    - Saving to Graph RAG (knowledge graph)
    
    The markdown format is a universal intermediate representation
    that both storage strategies can consume.
    """
    markdown_content: str  # Full content in Markdown format
    title: Optional[str] = None  # Document title
    source: Optional[str] = None  # Original file path/name
    metadata: Dict[str, Any] = field(default_factory=dict)  # Enriched metadata
    sections: List[Dict[str, str]] = field(default_factory=list)  # [{"title": "...", "content": "..."}]
    converted_at: datetime = field(default_factory=datetime.now)  # Conversion timestamp
    
    def __repr__(self) -> str:
        return f"ConvertedDocument(title={self.title}, sections={len(self.sections)}, content_len={len(self.markdown_content)})"
