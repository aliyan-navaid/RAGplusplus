"""
Neo4j Connection Manager - Singleton pattern for connection pooling
"""
import os
from typing import Optional
from neo4j import GraphDatabase, Driver
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Neo4jConnectionManager:
    """Singleton manager for Neo4j database connections."""
    
    _instance: Optional["Neo4jConnectionManager"] = None
    _driver: Optional[Driver] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jConnectionManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Neo4j connection parameters from environment variables."""
        if self._driver is None:
            self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
            self.username = os.getenv("NEO4J_USERNAME", "neo4j")
            self.password = os.getenv("NEO4J_PASSWORD", "")
            self.database = os.getenv("NEO4J_DATABASE", "neo4j")
            self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                encrypted=False  # Use True for production
            )
            # Test connection
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def get_driver(self) -> Driver:
        """Get the Neo4j driver instance."""
        if self._driver is None:
            self._connect()
        return self._driver
    
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
    
    def get_session(self):
        """Get a new session for database operations."""
        return self.get_driver().session(database=self.database)
    
    def __del__(self):
        """Ensure connection is closed on garbage collection."""
        self.close()
