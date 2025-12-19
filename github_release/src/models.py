"""
Minimal data models for MACM graph comparison.
"""
from dataclasses import dataclass, field
from typing import Dict, Set, Any, Optional
from collections import defaultdict


@dataclass
class GraphNode:
    """Represents a node in the graph."""
    id: str
    labels: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)


@dataclass
class GraphRelationship:
    """Represents a relationship in the graph."""
    id: str
    start_node_id: str
    end_node_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    """Represents a complete graph structure."""
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    relationships: Dict[str, GraphRelationship] = field(default_factory=dict)
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_relationship(self, relationship: GraphRelationship) -> None:
        """Add a relationship to the graph."""
        self.relationships[relationship.id] = relationship
    
    @staticmethod
    def from_cypher_file(filepath: str) -> 'Graph':
        """
        Load a graph from a Cypher (.macm) file.
        
        Requires Neo4j running on bolt://localhost:7687
        with default credentials (neo4j/password).
        """
        from .database_manager import DatabaseManager
        from .utils import load_graph_from_cypher_string
        
        with open(filepath, 'r') as f:
            cypher = f.read()
        
        db = DatabaseManager('bolt://localhost:7687', 'neo4j', 'password')
        db.connect()
        graph = load_graph_from_cypher_string(db, cypher)
        db.disconnect()
        
        return graph
