"""Graph indexing utilities: extract entities/relations and store in Neo4j."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.ingestion.models import ParsedDocument
from src.ingestion.parsers.markdown_converter import MarkdownConverter
from src.repositories.graph.models import ChunkNode, DocumentNode, EntityNode


_ORG_KEYWORDS = {
    "university",
    "college",
    "institute",
    "inc",
    "ltd",
    "llc",
    "corp",
    "company",
    "systems",
    "technologies",
}

_STOPWORDS = {
    "The",
    "And",
    "For",
    "With",
    "From",
    "This",
    "That",
    "These",
    "Those",
    "A",
    "An",
    "In",
    "On",
    "To",
    "Of",
    "By",
    "At",
}

_TITLE_CASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
_UPPER_RE = re.compile(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,})?)\b")


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _classify_entity(name: str) -> str:
    lowered = name.lower()
    for key in _ORG_KEYWORDS:
        if key in lowered:
            return "ORG"
    if len(name.split()) == 2:
        return "PERSON"
    return "ENTITY"


def _is_valid_entity(name: str) -> bool:
    if not name or len(name) < 3:
        return False
    if name in _STOPWORDS:
        return False
    if name.lower() in (s.lower() for s in _STOPWORDS):
        return False
    return True


def _extract_entities(text: str) -> List[str]:
    candidates: List[str] = []
    for phrase in _TITLE_CASE_RE.findall(text or ""):
        if _is_valid_entity(phrase):
            candidates.append(phrase.strip())
    for phrase in _UPPER_RE.findall(text or ""):
        if _is_valid_entity(phrase):
            candidates.append(phrase.strip())
    # preserve order but remove duplicates (case-insensitive)
    seen = set()
    ordered: List[str] = []
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def index_markdown_graph(
    markdown: str,
    graph_repo,
    source: Optional[str] = None,
    title: Optional[str] = None,
    max_entities: int = 100,
    max_relationships: int = 200,
) -> Dict[str, int | str]:
    """Extract entities/relations from markdown and store in Neo4j.

    Returns counts and the created document id.
    """
    converter = MarkdownConverter()
    parsed = ParsedDocument(content=markdown, title=title, metadata={"source": source or ""})
    converted = converter.convert(parsed, source_path=source)

    if converted.sections:
        sections = converted.sections
    else:
        sections = [{"title": title or "Document", "content": markdown}]

    file_type = ""
    if source:
        file_type = Path(source).suffix.lstrip(".")

    doc_node = DocumentNode(
        title=title or converted.title or (Path(source).stem if source else "Document"),
        source=source or "",
        file_type=file_type,
        metadata=converted.metadata,
    )
    doc_id = graph_repo.create_document(doc_node)

    entity_id_by_key: Dict[str, str] = {}
    created_entities = 0
    created_relationships = 0
    created_chunks = 0
    related_pairs: set[Tuple[str, str]] = set()

    for idx, section in enumerate(sections):
        content = (section.get("content") or "").strip()
        section_title = section.get("title") or ""

        chunk_node = ChunkNode(
            content=content,
            chunk_index=idx,
            tokens_count=_word_count(content),
            metadata={"title": section_title, "source": source or ""},
        )
        chunk_id = graph_repo.create_chunk(chunk_node)
        created_chunks += 1
        graph_repo.create_relationship(doc_id, chunk_id, "CONTAINS")

        section_text = f"{section_title}\n{content}".strip()
        entities = _extract_entities(section_text)
        if len(entity_id_by_key) >= max_entities:
            entities = entities[: max(0, max_entities - len(entity_id_by_key))]

        section_entity_ids: List[str] = []
        for name in entities:
            ent_type = _classify_entity(name)
            key = f"{name.lower()}::{ent_type}"
            if key in entity_id_by_key:
                entity_id = entity_id_by_key[key]
            else:
                existing = graph_repo.get_entity_by_name_and_type(name, ent_type)
                if existing is not None:
                    entity_id = existing.id
                else:
                    entity_node = EntityNode(name=name, type=ent_type, metadata={"source": source or ""})
                    entity_id = graph_repo.create_entity(entity_node)
                    created_entities += 1
                entity_id_by_key[key] = entity_id
            section_entity_ids.append(entity_id)
            graph_repo.create_relationship(chunk_id, entity_id, "MENTIONS")

        # Create RELATED_TO links for co-occurring entities in the same section
        for i in range(len(section_entity_ids)):
            for j in range(i + 1, len(section_entity_ids)):
                if created_relationships >= max_relationships:
                    break
                pair = tuple(sorted((section_entity_ids[i], section_entity_ids[j])))
                if pair in related_pairs:
                    continue
                related_pairs.add(pair)
                graph_repo.create_relationship(pair[0], pair[1], "RELATED_TO")
                created_relationships += 1

    return {
        "doc_id": doc_id,
        "chunks_created": created_chunks,
        "entities_created": created_entities,
        "relationships_created": created_relationships,
    }
