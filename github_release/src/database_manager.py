"""
Neo4j Database Manager for MACM graphs.
"""
import logging
from typing import Optional, List, Any
from neo4j import GraphDatabase, Session


class DatabaseManager:
    """Manages Neo4j database connections and operations."""
    
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize the Neo4j manager.
        
        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            username: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
                
            self.logger.info(f"Connected to Neo4j at {self.uri}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            self.logger.info("Disconnected from Neo4j")
    
    def execute_query(self, query: str) -> List[Any]:
        """
        Execute a Cypher query.
        
        Args:
            query: Cypher query string
            
        Returns:
            List of query results
        """
        if not self.driver:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        with self.driver.session() as session:
            result = session.run(query)
            return list(result)
    
    def clear_database(self) -> None:
        """Delete all nodes and relationships from the database."""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)
        self.logger.info("Database cleared")
