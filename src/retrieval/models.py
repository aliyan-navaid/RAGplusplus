from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalHit:
    id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResponse:
    query: str
    answer: str
    prompt: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    hits: List[RetrievalHit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["hits"] = [hit.to_dict() for hit in self.hits]
        return payload
