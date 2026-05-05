"""
Neo4j implementation of graph database repository
"""
import logging
from typing import Optional, List, Dict, Any
from neo4j import Session

from .base_repository import BaseGraphRepository
from .models import DocumentNode, ChunkNode, EntityNode
from .connection import Neo4jConnectionManager

logger = logging.getLogger(__name__)


class Neo4jRepository(BaseGraphRepository):
    """Neo4j implementation of graph database repository."""
    
    def __init__(self):
        """Initialize Neo4j repository with connection manager."""
        self.conn_manager = Neo4jConnectionManager()
    
    def _get_session(self) -> Session:
        """Get a new Neo4j session."""
        return self.conn_manager.get_session()
    
    # ==================== Document Operations ====================
    
    def create_document(self, document: DocumentNode) -> str:
        """Create a document node in Neo4j."""
        session = self._get_session()
        try:
            doc_dict = document.to_dict()
            query = """
            CREATE (d:Document $props)
            RETURN d.id as id
            """
            result = session.run(query, props=doc_dict)
            doc_id = result.single()["id"]
            logger.info(f"Created document: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            raise
        finally:
            session.close()
    
    def get_document(self, document_id: str) -> Optional[DocumentNode]:
        """Get a document by ID."""
        session = self._get_session()
        try:
            query = "MATCH (d:Document {id: $id}) RETURN d"
            result = session.run(query, id=document_id)
            record = result.single()
            if record:
                doc_props = dict(record["d"])
                return DocumentNode(**doc_props)
            return None
        finally:
            session.close()
    
    def update_document(self, document_id: str, document: DocumentNode) -> bool:
        """Update a document node."""
        session = self._get_session()
        try:
            doc_dict = document.to_dict()
            # Remove id from update properties to avoid changing it
            doc_dict.pop("id", None)
            
            query = """
            MATCH (d:Document {id: $id})
            SET d += $props
            RETURN COUNT(d) as count
            """
            result = session.run(query, id=document_id, props=doc_dict)
            count = result.single()["count"]
            logger.info(f"Updated document: {document_id}")
            return count > 0
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise
        finally:
            session.close()
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its relationships."""
        session = self._get_session()
        try:
            query = """
            MATCH (d:Document {id: $id})
            DETACH DELETE d
            RETURN COUNT(d) as count
            """
            result = session.run(query, id=document_id)
            count = result.single()["count"]
            if count > 0:
                logger.info(f"Deleted document: {document_id}")
            return count > 0
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise
        finally:
            session.close()
    
    def list_documents(self, limit: int = 100, skip: int = 0) -> List[DocumentNode]:
        """List all documents with pagination."""
        session = self._get_session()
        try:
            query = """
            MATCH (d:Document)
            RETURN d
            SKIP $skip
            LIMIT $limit
            """
            result = session.run(query, skip=skip, limit=limit)
            documents = []
            for record in result:
                doc_props = dict(record["d"])
                documents.append(DocumentNode(**doc_props))
            return documents
        finally:
            session.close()
    
    # ==================== Chunk Operations ====================
    
    def create_chunk(self, chunk: ChunkNode) -> str:
        """Create a chunk node."""
        session = self._get_session()
        try:
            chunk_dict = chunk.to_dict()
            query = """
            CREATE (c:Chunk $props)
            RETURN c.id as id
            """
            result = session.run(query, props=chunk_dict)
            chunk_id = result.single()["id"]
            logger.info(f"Created chunk: {chunk_id}")
            return chunk_id
        except Exception as e:
            logger.error(f"Error creating chunk: {e}")
            raise
        finally:
            session.close()
    
    def get_chunk(self, chunk_id: str) -> Optional[ChunkNode]:
        """Get a chunk by ID."""
        session = self._get_session()
        try:
            query = "MATCH (c:Chunk {id: $id}) RETURN c"
            result = session.run(query, id=chunk_id)
            record = result.single()
            if record:
                chunk_props = dict(record["c"])
                return ChunkNode(**chunk_props)
            return None
        finally:
            session.close()
    
    def update_chunk(self, chunk_id: str, chunk: ChunkNode) -> bool:
        """Update a chunk node."""
        session = self._get_session()
        try:
            chunk_dict = chunk.to_dict()
            chunk_dict.pop("id", None)
            
            query = """
            MATCH (c:Chunk {id: $id})
            SET c += $props
            RETURN COUNT(c) as count
            """
            result = session.run(query, id=chunk_id, props=chunk_dict)
            count = result.single()["count"]
            logger.info(f"Updated chunk: {chunk_id}")
            return count > 0
        except Exception as e:
            logger.error(f"Error updating chunk: {e}")
            raise
        finally:
            session.close()
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk node."""
        session = self._get_session()
        try:
            query = """
            MATCH (c:Chunk {id: $id})
            DETACH DELETE c
            RETURN COUNT(c) as count
            """
            result = session.run(query, id=chunk_id)
            count = result.single()["count"]
            if count > 0:
                logger.info(f"Deleted chunk: {chunk_id}")
            return count > 0
        except Exception as e:
            logger.error(f"Error deleting chunk: {e}")
            raise
        finally:
            session.close()
    
    def get_chunks_by_document(self, document_id: str) -> List[ChunkNode]:
        """Get all chunks of a document, ordered by chunk_index."""
        session = self._get_session()
        try:
            query = """
            MATCH (d:Document {id: $doc_id})-[r:CONTAINS]->(c:Chunk)
            RETURN c
            ORDER BY c.chunk_index ASC
            """
            result = session.run(query, doc_id=document_id)
            chunks = []
            for record in result:
                chunk_props = dict(record["c"])
                chunks.append(ChunkNode(**chunk_props))
            return chunks
        finally:
            session.close()
    
    # ==================== Entity Operations ====================
    
    def create_entity(self, entity: EntityNode) -> str:
        """Create an entity node."""
        session = self._get_session()
        try:
            entity_dict = entity.to_dict()
            query = """
            CREATE (e:Entity $props)
            RETURN e.id as id
            """
            result = session.run(query, props=entity_dict)
            entity_id = result.single()["id"]
            logger.info(f"Created entity: {entity_id}")
            return entity_id
        except Exception as e:
            logger.error(f"Error creating entity: {e}")
            raise
        finally:
            session.close()
    
    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        """Get an entity by ID."""
        session = self._get_session()
        try:
            query = "MATCH (e:Entity {id: $id}) RETURN e"
            result = session.run(query, id=entity_id)
            record = result.single()
            if record:
                entity_props = dict(record["e"])
                return EntityNode(**entity_props)
            return None
        finally:
            session.close()
    
    def get_entity_by_name_and_type(self, name: str, entity_type: str) -> Optional[EntityNode]:
        """Get an entity by name and type (respects unique constraint)."""
        session = self._get_session()
        try:
            query = """
            MATCH (e:Entity {name: $name, type: $type})
            RETURN e
            """
            result = session.run(query, name=name, type=entity_type)
            record = result.single()
            if record:
                entity_props = dict(record["e"])
                return EntityNode(**entity_props)
            return None
        finally:
            session.close()
    
    def update_entity(self, entity_id: str, entity: EntityNode) -> bool:
        """Update an entity node."""
        session = self._get_session()
        try:
            entity_dict = entity.to_dict()
            entity_dict.pop("id", None)
            
            query = """
            MATCH (e:Entity {id: $id})
            SET e += $props
            RETURN COUNT(e) as count
            """
            result = session.run(query, id=entity_id, props=entity_dict)
            count = result.single()["count"]
            logger.info(f"Updated entity: {entity_id}")
            return count > 0
        except Exception as e:
            logger.error(f"Error updating entity: {e}")
            raise
        finally:
            session.close()
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity node."""
        session = self._get_session()
        try:
            query = """
            MATCH (e:Entity {id: $id})
            DETACH DELETE e
            RETURN COUNT(e) as count
            """
            result = session.run(query, id=entity_id)
            count = result.single()["count"]
            if count > 0:
                logger.info(f"Deleted entity: {entity_id}")
            return count > 0
        except Exception as e:
            logger.error(f"Error deleting entity: {e}")
            raise
        finally:
            session.close()
    
    def list_entities(self, entity_type: Optional[str] = None, limit: int = 100) -> List[EntityNode]:
        """List entities, optionally filtered by type."""
        session = self._get_session()
        try:
            if entity_type:
                query = """
                MATCH (e:Entity {type: $type})
                RETURN e
                LIMIT $limit
                """
                result = session.run(query, type=entity_type, limit=limit)
            else:
                query = """
                MATCH (e:Entity)
                RETURN e
                LIMIT $limit
                """
                result = session.run(query, limit=limit)
            
            entities = []
            for record in result:
                entity_props = dict(record["e"])
                entities.append(EntityNode(**entity_props))
            return entities
        finally:
            session.close()
    
    # ==================== Relationship Operations ====================
    
    def create_relationship(self, source_id: str, target_id: str, rel_type: str,
                           properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between two nodes."""
        session = self._get_session()
        try:
            properties = properties or {}
            # Find nodes by id (could be any type)
            query = f"""
            MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
            CREATE (source)-[r:{rel_type} $props]->(target)
            RETURN COUNT(r) as count
            """
            result = session.run(query, source_id=source_id, target_id=target_id, props=properties)
            count = result.single()["count"]
            if count > 0:
                logger.info(f"Created relationship: {source_id} -[{rel_type}]-> {target_id}")
            return count > 0
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            raise
        finally:
            session.close()
    
    def delete_relationship(self, source_id: str, target_id: str, rel_type: str) -> bool:
        """Delete a relationship between two nodes."""
        session = self._get_session()
        try:
            query = f"""
            MATCH (source {{id: $source_id}})-[r:{rel_type}]->(target {{id: $target_id}})
            DELETE r
            RETURN COUNT(r) as count
            """
            result = session.run(query, source_id=source_id, target_id=target_id)
            count = result.single()["count"]
            if count > 0:
                logger.info(f"Deleted relationship: {source_id} -[{rel_type}]-> {target_id}")
            return count > 0
        except Exception as e:
            logger.error(f"Error deleting relationship: {e}")
            raise
        finally:
            session.close()
    
    # ==================== Graph Query Operations ====================
    
    def get_document_context(self, chunk_id: str, depth: int = 1) -> Dict[str, Any]:
        """Get document context for a chunk (document -> chunks -> entities)."""
        session = self._get_session()
        try:
            query = """
            MATCH (d:Document)-[:CONTAINS]->(c:Chunk {id: $chunk_id})
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            WITH d, c, COLLECT(e) as entities
            RETURN d, c, entities
            """
            result = session.run(query, chunk_id=chunk_id)
            record = result.single()
            if record:
                d = record["d"]
                c = record["c"]
                entities = record["entities"]
                return {
                    "document": dict(d) if d else None,
                    "chunk": dict(c) if c else None,
                    "entities": [dict(e) for e in (entities or [])]
                }
            return {"document": None, "chunk": None, "entities": []}
        finally:
            session.close()
    
    def search_entities_by_name(self, name: str, limit: int = 10) -> List[EntityNode]:
        """Search for entities by name (substring search)."""
        session = self._get_session()
        try:
            query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $name
            RETURN e
            LIMIT $limit
            """
            result = session.run(query, name=name, limit=limit)
            entities = []
            for record in result:
                entity_props = dict(record["e"])
                entities.append(EntityNode(**entity_props))
            return entities
        finally:
            session.close()
    
    def get_related_entities(self, entity_id: str, limit: int = 10) -> List[EntityNode]:
        """Get entities related to a given entity."""
        session = self._get_session()
        try:
            query = """
            MATCH (e:Entity {id: $entity_id})-[r:RELATED_TO]-(related:Entity)
            RETURN related
            LIMIT $limit
            """
            result = session.run(query, entity_id=entity_id, limit=limit)
            entities = []
            for record in result:
                entity_props = dict(record["related"])
                entities.append(EntityNode(**entity_props))
            return entities
        finally:
            session.close()
    
    # ==================== Utility Operations ====================
    
    def clear_all(self) -> bool:
        """Delete all nodes and relationships (for testing)."""
        session = self._get_session()
        try:
            query = "MATCH (n) DETACH DELETE n"
            session.run(query)
            logger.warning("Cleared all nodes from Neo4j database")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
        finally:
            session.close()
    
    def close(self) -> None:
        """Close database connection."""
        self.conn_manager.close()
