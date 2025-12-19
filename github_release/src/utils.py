"""
Utility functions for loading MACM graphs from Cypher files.
"""
from typing import Optional
from .models import Graph, GraphNode, GraphRelationship
from .database_manager import DatabaseManager


def load_graph_from_cypher_string(db: DatabaseManager, cypher: str, clear_db: bool = True) -> Graph:
    """
    Load a graph from a Cypher string by executing it in Neo4j.
    
    Args:
        db: Neo4j database manager
        cypher: Cypher CREATE statements
        clear_db: Whether to clear database before loading
        
    Returns:
        Graph object with nodes and relationships
    """
    if clear_db:
        db.clear_database()
    
    # Execute Cypher statements
    db.execute_query(cypher)
    
    # Fetch all nodes
    graph = Graph()
    
    nodes_query = "MATCH (n) RETURN n"
    results = db.execute_query(nodes_query)
    
    for record in results:
        node = record['n']
        labels = set(node.labels)
        properties = dict(node)
        
        graph_node = GraphNode(
            id=str(node.element_id),
            labels=labels,
            properties=properties
        )
        graph.add_node(graph_node)
    
    # Fetch all relationships
    rels_query = "MATCH ()-[r]->() RETURN r"
    results = db.execute_query(rels_query)
    
    for record in results:
        rel = record['r']
        
        graph_rel = GraphRelationship(
            id=str(rel.element_id),
            start_node_id=str(rel.start_node.element_id),
            end_node_id=str(rel.end_node.element_id),
            relationship_type=rel.type,
            properties=dict(rel)
        )
        graph.add_relationship(graph_rel)
    
    return graph


def load_graph_from_file(filepath: str, db: Optional[DatabaseManager] = None) -> Graph:
    """
    Load a graph from a .macm file.
    
    Args:
        filepath: Path to .macm file
        db: Optional DatabaseManager (creates default if None)
        
    Returns:
        Graph object
    """
    with open(filepath, 'r') as f:
        cypher = f.read()
    
    if db is None:
        db = DatabaseManager('bolt://localhost:7687', 'neo4j', 'password')
        db.connect()
        graph = load_graph_from_cypher_string(db, cypher)
        db.disconnect()
    else:
        graph = load_graph_from_cypher_string(db, cypher)
    
    return graph
