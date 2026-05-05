"""
Data models for Neo4j graph database entities
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid


@dataclass
class DocumentNode:
    """Represents a Document node in Neo4j."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    source: str = ""  # file path
    file_type: str = ""  # pdf, docx, md, etc.
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    embedding_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j."""
        data = asdict(self)
        # Filter out None values and empty dicts (Neo4j doesn't allow nested maps)
        return {k: v for k, v in data.items() if v is not None and not (isinstance(v, dict) and not v)}


@dataclass
class ChunkNode:
    """Represents a Chunk node in Neo4j."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    chunk_index: int = 0  # position in document
    tokens_count: int = 0
    embedding: Optional[List[float]] = None  # vector embedding
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j."""
        data = asdict(self)
        # Filter out None values and empty dicts (Neo4j doesn't allow nested maps)
        return {k: v for k, v in data.items() if v is not None and not (isinstance(v, dict) and not v)}


@dataclass
class EntityNode:
    """Represents an Entity node in Neo4j (extracted entities)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = ""  # PERSON, ORG, LOCATION, CONCEPT, etc.
    embedding: Optional[List[float]] = None  # vector embedding
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j."""
        data = asdict(self)
        # Filter out None values and empty dicts (Neo4j doesn't allow nested maps)
        return {k: v for k, v in data.items() if v is not None and not (isinstance(v, dict) and not v)}


@dataclass
class Relationship:
    """Represents a relationship between nodes."""
    source_id: str  # id of source node
    target_id: str  # id of target node
    relationship_type: str  # CONTAINS, MENTIONS, RELATED_TO, SIMILAR_TO, etc.
    properties: Dict[str, Any] = field(default_factory=dict)  # optional properties
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
