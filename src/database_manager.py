"""
Neo4j Database Manager module.

This module provides a high-level interface for interacting with Neo4j databases,
including connection management, query execution, and data retrieval.
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Iterator

import neo4j
from neo4j import GraphDatabase, Session, Result

from config import DatabaseConfig


class Neo4jManager:
    """Manages Neo4j database connections and operations."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize the Neo4j manager with configuration.
        
        Args:
            config: Database configuration settings
        """
        self.config = config
        self.driver: Optional[neo4j.Driver] = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
                
            self.logger.info(f"Successfully connected to Neo4j at {self.config.uri}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            self.logger.info("Disconnected from Neo4j")
    
    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """
        Context manager for database sessions.
        
        Yields:
            Neo4j session object
        """
        if not self.driver:
            raise RuntimeError("Database not connected. Call connect() first.")
            
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def execute_cypher_file(self, file_path: str) -> None:
        """
        Execute Cypher commands from a file.
        
        Args:
            file_path: Path to the Cypher file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                cypher_content = file.read()
            
            # Remove comments and split by semicolons to handle multiple statements
            lines = []
            for line in cypher_content.split('\n'):
                # Remove comments (lines starting with //)
                if not line.strip().startswith('//'):
                    lines.append(line)
            
            cleaned_content = '\n'.join(lines)
            
            # Split by semicolon and execute each statement separately
            statements = [stmt.strip() for stmt in cleaned_content.split(';') if stmt.strip()]
            
            with self.get_session() as session:
                for statement in statements:
                    session.run(statement)
                
            self.logger.info(f"Successfully executed Cypher file: {file_path}")
            
        except FileNotFoundError:
            self.logger.error(f"Cypher file not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error executing Cypher file {file_path}: {e}")
            raise
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """
        Retrieve all nodes from the database.
        
        Returns:
            List of node dictionaries
        """
        with self.get_session() as session:
            result = session.run("MATCH (n) RETURN n")
            return [record['n'] for record in result]
    
    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """
        Retrieve all relationships from the database.
        
        Returns:
            List of relationship dictionaries
        """
        with self.get_session() as session:
            result = session.run("MATCH ()-[r]-() RETURN r")
            return [record['r'] for record in result]
    
    def get_graph_statistics(self) -> Dict[str, int]:
        """
        Get basic statistics about the graph.
        
        Returns:
            Dictionary with node count, relationship count, and other metrics
        """
        with self.get_session() as session:
            # Get node count
            node_result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = node_result.single()['node_count']
            
            # Get relationship count
            rel_result = session.run("MATCH ()-[r]-() RETURN count(r) as rel_count")
            rel_count = rel_result.single()['rel_count']
            
            # Get label statistics
            label_result = session.run("""
                MATCH (n)
                UNWIND labels(n) as label
                RETURN label, count(*) as count
                ORDER BY count DESC
            """)
            label_stats = {record['label']: record['count'] for record in label_result}
            
            return {
                'node_count': node_count,
                'relationship_count': rel_count,
                'label_statistics': label_stats
            }
    
    def clear_database(self) -> None:
        """Clear all data from the database."""
        with self.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        self.logger.info("Database cleared")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
