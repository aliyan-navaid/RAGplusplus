"""
Test script for RAGplusplus PDF parsing pipeline.

Uses the parser and converter objects from the project:
- DoclingParser: Extracts text from PDF
- MarkdownConverter: Converts to standardized Markdown format

Usage:
    python parse_pdf.py <path_to_pdf>
    
Example:
    python parse_pdf.py Faizan_Jawaid_Resume.pdf
"""

import sys
from pathlib import Path

# Add parent directories to path to import RAGplusplus modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_pdf.py <path_to_pdf>")
        print("\nExample:")
        print("  python parse_pdf.py Faizan_Jawaid_Resume.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Parsing: {pdf_path.name}")
    print(f"File size: {pdf_path.stat().st_size / (1024*1024):.2f} MB")
    print("-" * 70)
    
    try:
        # Import parser and converter objects
        from src.ingestion.parsers.docling_parser import DoclingParser
        from src.ingestion.parsers.markdown_converter import MarkdownConverter
        
        # Step 1: Parse PDF using DoclingParser
        print("Parsing PDF with DoclingParser...")
        parser = DoclingParser(use_ocr=True)
        parsed_doc = parser.parse(pdf_path)
        print("PDF parsing complete")
        
        # Step 2: Convert to Markdown using MarkdownConverter
        print("Converting to Markdown...")
        converter = MarkdownConverter()
        converted_doc = converter.convert(parsed_doc, source_path=str(pdf_path))
        print("Markdown conversion complete")
        
        # Display results
        print("\n" + "="*70)
        print("DOCUMENT METADATA")
        print("="*70)
        print(f"Title:       {parsed_doc.title or 'N/A'}")
        print(f"Pages:       {parsed_doc.pages or 'N/A'}")
        print(f"Author:      {parsed_doc.metadata.get('author', 'N/A')}")
        print(f"Created:     {parsed_doc.metadata.get('creation_date', 'N/A')}")
        
        # Show content preview
        print("\n" + "="*70)
        print("EXTRACTED MARKDOWN CONTENT (first 1000 chars)")
        print("="*70)
        print(converted_doc.markdown_content[:1000])
        print("\n... (truncated) ...")
        
        # Show sections
        print("\n" + "="*70)
        print("SECTIONS EXTRACTED")
        print("="*70)
        for i, section in enumerate(converted_doc.sections[:5], 1):
            print(f"{i}. {section.get('title', 'Untitled')}")
        if len(converted_doc.sections) > 5:
            print(f"... and {len(converted_doc.sections) - 5} more sections")
        
        # Show statistics
        print("\n" + "="*70)
        print("STATISTICS")
        print("="*70)
        print(f"Total characters: {len(converted_doc.markdown_content):,}")
        print(f"Total lines:      {len(converted_doc.markdown_content.splitlines()):,}")
        print(f"Total sections:   {len(converted_doc.sections)}")
        
        # Save to file
        output_path = pdf_path.parent / f"{pdf_path.stem}_extracted.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(converted_doc.markdown_content)
        
        print(f"\nMarkdown saved to: {output_path}")
        print(f"To view: cat '{output_path}'")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure you're running from the correct directory:")
        print("  cd d:\\AI_Project\\RAGplusplus\\tests\\Ingestion_pipline")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
