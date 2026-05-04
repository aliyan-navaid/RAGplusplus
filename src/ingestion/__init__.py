"""
Ingestion Pipeline Module

This module handles the entire document ingestion process:
1. Parse PDF using Docling/Nougat
2. Convert parsed content to Markdown
3. Save to either standard RAG or Graph RAG using strategy pattern
"""
