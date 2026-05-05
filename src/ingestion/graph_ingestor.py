"""Graph ingestion helper for Markdown documents."""
from pathlib import Path
from typing import Dict, List, Optional
import re

# Absolute imports (relative to 'src')
from ingestion.models import ConvertedDocument
from ingestion.parsers.docling_parser import DoclingParser
from ingestion.parsers.markdown_converter import MarkdownConverter
from repositories.graph import (
    Neo4jRepository,
    DocumentNode,
    ChunkNode,
    EntityNode,
)

class GraphIngestor:
    """Ingests Markdown content into Neo4j graph database."""

    def __init__(self):
        self.repo = Neo4jRepository()
        self.converter = MarkdownConverter()

    def ingest_markdown_file(self, markdown_path: str, title: Optional[str] = None) -> dict:
        """Read a Markdown file and ingest it into Neo4j."""
        path = Path(markdown_path)
        content = path.read_text(encoding="utf-8")
        document = ConvertedDocument(
            markdown_content=content,
            title=title or path.stem,
            source=str(path),
            metadata={"source": str(path)},
            sections=self.converter._extract_sections(content),
        )
        return self.ingest_converted_document(document)

    def ingest_pdf_file(self, pdf_path: str, title: Optional[str] = None, use_ocr: bool = True) -> dict:
        """Parse a PDF, convert it to Markdown, and ingest it into Neo4j."""
        parser = DoclingParser(use_ocr=use_ocr)
        parsed_doc = parser.parse(pdf_path)
        converted_doc = self.converter.convert(parsed_doc, source_path=str(pdf_path))
        if title:
            converted_doc.title = title
        return self.ingest_converted_document(converted_doc)

    def ingest_converted_document(self, document: ConvertedDocument) -> dict:
        """Ingest a ConvertedDocument into Neo4j with dynamic flattening."""
        
        # 1. Prepare metadata dict with sections_count and other metadata
        metadata = document.metadata.copy() if document.metadata else {}
        metadata["sections_count"] = len(document.sections)
        
        # 2. Create DocumentNode with only the fields it expects
        doc_node = DocumentNode(
            title=document.title or "Untitled",
            source=document.source or "",
            file_type="markdown",
            metadata=metadata,
        )

        # 3. Pass the document node to repository
        doc_id = self.repo.create_document(doc_node)

        # Create chunks from sections or fallback to one chunk
        chunks = document.sections or [{"title": document.title or "document", "content": document.markdown_content}]
        section_entities = self._extract_entities(document)
        relationships = self._extract_relationships(document, section_entities)

        entity_ids: Dict[str, str] = {}
        created_entities = 0
        created_relations = 0

        for idx, section in enumerate(chunks):
            chunk_content = section.get("content", "").strip()
            if not chunk_content:
                continue

            chunk_node = ChunkNode(
                content=chunk_content,
                chunk_index=idx,
                tokens_count=len(chunk_content.split()),
                metadata={"section_title": section.get("title", "")},
            )
            chunk_id = self.repo.create_chunk(chunk_node)
            self.repo.create_relationship(doc_id, chunk_id, "CONTAINS")

            # Link entities mentioned in this section to the chunk
            for entity in [e for e in section_entities if e["section"] == section.get("title")]:
                entity_key = entity["entity"]
                entity_id = entity_ids.get(entity_key)
                if entity_id is None:
                    entity_id = self._get_or_create_entity(entity)
                    if entity_id is not None:
                        entity_ids[entity_key] = entity_id
                if entity_id:
                    self.repo.create_relationship(chunk_id, entity_id, "MENTIONS")
                    created_relations += 1

        # Create entity-to-entity relationships
        for relation in relationships:
            subject_id = entity_ids.get(relation["subject"])
            object_id = entity_ids.get(relation["object"])
            if subject_id and object_id and subject_id != object_id:
                if self.repo.create_relationship(subject_id, object_id, relation["predicate"]):
                    created_relations += 1

        return {
            "success": True,
            "doc_id": doc_id,
            "sections_created": len(chunks),
            "entities_created": len(entity_ids),
            "relationships_created": created_relations,
            "message": f"Ingested markdown file into Neo4j: {doc_id}",
        }

    def _get_or_create_entity(self, entity: Dict[str, str]) -> Optional[str]:
        """Find or create an EntityNode in Neo4j."""
        name = entity["entity"]
        ent_type = entity.get("type", "ENTITY")
        existing = self.repo.get_entity_by_name_and_type(name, ent_type)
        if existing is not None:
            return existing.id

        created_id = self.repo.create_entity(EntityNode(name=name, type=ent_type))
        return created_id

    def _extract_entities(self, document: ConvertedDocument) -> List[Dict[str, str]]:
        """Extract entity candidates from document sections."""
        entities: List[Dict[str, str]] = []
        seen = set()

        for section in document.sections:
            section_title = section.get("title", "")
            content = section.get("content", "")
            phrases = re.findall(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b", content)

            for phrase in phrases:
                if len(phrase) > 2 and phrase not in seen:
                    seen.add(phrase)
                    entities.append({
                        "entity": phrase,
                        "type": "ENTITY",
                        "section": section_title,
                    })

        return entities

    def _extract_relationships(
        self,
        document: ConvertedDocument,
        entities: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Extract simple co-occurrence relationships between entities."""
        relationships: List[Dict[str, str]] = []

        for section in document.sections:
            section_entities = [e for e in entities if e["section"] == section.get("title")]
            for i, source in enumerate(section_entities):
                for target in section_entities[i + 1 :]:
                    relationships.append({
                        "subject": source["entity"],
                        "predicate": "RELATED_TO",
                        "object": target["entity"],
                        "section": section.get("title"),
                    })

        return relationships
