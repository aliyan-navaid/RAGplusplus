"""
Markdown converter for normalized document formatting.

Converts ParsedDocument (raw parser output) to ConvertedDocument (Markdown format).

FLOW:
Input: ParsedDocument from parser
↓ [MarkdownConverter.convert()]
↓ Format to standard Markdown
↓ Split into sections
↓ Add metadata
Output: ConvertedDocument (ready for RAG/Graph storage)
"""

import logging
import re
from typing import List, Dict, Optional, Tuple

from ..models import ParsedDocument, ConvertedDocument

logger = logging.getLogger(__name__)


class MarkdownConverter:
    """
    Converts parsed documents to standardized Markdown format.
    
    INPUTS:
    - parsed_doc: ParsedDocument from Docling parser
    
    OUTPUTS:
    - ConvertedDocument with:
      - markdown_content: Full Markdown-formatted content
      - sections: [{"title": "...", "content": "..."}]
      - metadata: Enriched metadata
    
    This universal Markdown format is consumed by both:
    - RAG Strategy (for vector embedding)
    - Graph Strategy (for entity extraction)
    """
    
    def __init__(self, section_marker: str = "#"):
        """
        Initialize the converter.
        
        Args:
            section_marker: Markdown heading marker for sections (default: "#" for H1)
        """
        self.section_marker = section_marker
        logger.info("MarkdownConverter initialized")
    
    def convert(self, parsed_doc: ParsedDocument, source_path: Optional[str] = None) -> ConvertedDocument:
        """
        Convert ParsedDocument to Markdown format.
        
        FLOW:
        1. Take raw extracted content from ParsedDocument
        2. Normalize and format as Markdown
        3. Extract sections based on heading structure
        4. Enrich metadata
        5. Return ConvertedDocument
        
        Args:
            parsed_doc: ParsedDocument from parser
            source_path: Optional source file path for metadata
            
        Returns:
            ConvertedDocument: Markdown-formatted document ready for storage
        """
        try:
            logger.info(f"Converting document: {parsed_doc.title or 'Untitled'}")
            
            # The parsed_doc.content from Docling is already in Markdown
            # but we normalize and enhance it
            markdown_content = self._normalize_markdown(parsed_doc.content)
            
            # Extract sections from the Markdown content
            sections = self._extract_sections(markdown_content)
            
            # Enrich metadata
            metadata = {
                **parsed_doc.metadata,
                "pages": parsed_doc.pages,
                "sections_count": len(sections),
            }
            
            # Create ConvertedDocument
            converted_doc = ConvertedDocument(
                markdown_content=markdown_content,
                title=parsed_doc.title,
                source=source_path or parsed_doc.metadata.get("source"),
                metadata=metadata,
                sections=sections,
            )
            
            logger.info(
                f"Conversion complete: {converted_doc.title or 'Untitled'} "
                f"({len(sections)} sections, {len(markdown_content)} chars)"
            )
            
            return converted_doc
            
        except Exception as e:
            logger.error(f"Error converting document: {str(e)}")
            raise ValueError(f"Failed to convert document: {str(e)}")
    
    def _normalize_markdown(self, content: str) -> str:
        """
        Helper function for the  follwing things:
        
        Fixes:
         Multiple consecutive newlines → max 2 newlines
         Inconsistent spacing
         Missing newlines around headings
        
        Args:
            content: Raw Markdown content
            
        Returns:
            Normalized Markdown content
        """
        # Remove multiple consecutive newlines (keep max 2)
        content = re.sub(r"\n\n\n+", "\n\n", content)
        
        # Ensure newlines around Markdown headings
        content = re.sub(r"\n(#{1,6}\s)", r"\n\n\1", content)
        content = re.sub(r"(#{1,6}\s.*?)\n([^\n#])", r"\1\n\n\2", content)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in content.split("\n")]
        content = "\n".join(lines)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        return content
    
    def _extract_sections(self, markdown_content: str) -> List[Dict[str, str]]:
        """
        Extract sections from Markdown content based on heading hierarchy.
        
        Sections are:
        - Title: H1, H2, H3 headings
        - Content: Text until next heading
        
        EXAMPLE:
        Input Markdown:
        ```
        # Introduction
        This is the intro...
        
        ## Background
        Some background...
        ```
        
        Output:
        ```
        [
            {"title": "Introduction", "content": "This is the intro..."},
            {"title": "Background", "content": "Some background..."}
        ]
        ```
        
        Args:
            markdown_content: Markdown formatted content
            
        Returns:
            List of sections with title and content
        """
        sections: List[Dict[str, str]] = []
        
        # Split by headings (H1-H6)
        # This regex captures heading level and text
        heading_pattern = r"^(#{1,6})\s+(.+?)$"
        
        lines = markdown_content.split("\n")
        current_section: Optional[Dict[str, str]] = None
        current_content: List[str] = []
        
        for line in lines:
            heading_match = re.match(heading_pattern, line)
            
            if heading_match:
                # Save previous section if exists
                if current_section is not None:
                    current_section["content"] = "\n".join(current_content).strip()
                    sections.append(current_section)
                
                # Start new section
                level = heading_match.group(1)
                title = heading_match.group(2)
                current_section = {"title": title, "content": ""}
                current_content = []
            else:
                # Add line to current section content
                if current_section is not None:
                    current_content.append(line)
        
        # Save last section
        if current_section is not None:
            current_section["content"] = "\n".join(current_content).strip()
            sections.append(current_section)
        
        # Filter out empty sections
        sections = [s for s in sections if s.get("content", "").strip()]
        
        logger.debug(f"Extracted {len(sections)} sections from Markdown")
        
        return sections
