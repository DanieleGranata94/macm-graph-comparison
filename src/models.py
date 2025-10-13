"""
Data models for graph comparison.

This module defines the data structures used to represent graphs,
nodes, relationships, and comparison results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict


@dataclass
class GraphNode:
    """Represents a node in the graph."""
    
    id: str
    labels: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    element_id: Optional[str] = None
    
    def __hash__(self) -> int:
        """Make node hashable for use in sets."""
        return hash((self.id, tuple(sorted(self.labels))))
    
    def __eq__(self, other) -> bool:
        """Compare nodes based on id and labels."""
        if not isinstance(other, GraphNode):
            return False
        return self.id == other.id and self.labels == other.labels
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value with default."""
        return self.properties.get(key, default)
    
    def has_label(self, label: str) -> bool:
        """Check if node has a specific label."""
        return label in self.labels


@dataclass
class GraphRelationship:
    """Represents a relationship in the graph."""
    
    id: str
    start_node_id: str
    end_node_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    element_id: Optional[str] = None
    
    def __hash__(self) -> int:
        """Make relationship hashable for use in sets."""
        return hash((self.id, self.start_node_id, self.end_node_id, self.relationship_type))
    
    def __eq__(self, other) -> bool:
        """Compare relationships based on key attributes."""
        if not isinstance(other, GraphRelationship):
            return False
        return (self.start_node_id == other.start_node_id and
                self.end_node_id == other.end_node_id and
                self.relationship_type == other.relationship_type)


@dataclass
class Graph:
    """Represents a complete graph structure."""
    
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    relationships: Dict[str, GraphRelationship] = field(default_factory=dict)
    adjacency_list: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        # Attach back-reference so other utilities can access graph context from node
        try:
            node._graph = self
        except Exception:
            pass
    
    def add_relationship(self, relationship: GraphRelationship) -> None:
        """Add a relationship to the graph."""
        self.relationships[relationship.id] = relationship
        # Attach back-reference for relationship
        try:
            relationship._graph = self
        except Exception:
            pass
        
        # Update adjacency list
        self.adjacency_list[relationship.start_node_id].add(relationship.end_node_id)
        self.adjacency_list[relationship.end_node_id].add(relationship.start_node_id)
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str) -> Set[str]:
        """Get all neighbor node IDs for a given node."""
        return self.adjacency_list.get(node_id, set())
    
    def get_nodes_by_label(self, label: str) -> List[GraphNode]:
        """Get all nodes with a specific label."""
        return [node for node in self.nodes.values() if node.has_label(label)]
    
    def get_relationships_by_type(self, rel_type: str) -> List[GraphRelationship]:
        """Get all relationships of a specific type."""
        return [rel for rel in self.relationships.values() if rel.relationship_type == rel_type]
    
    def get_node_count(self) -> int:
        """Get the total number of nodes."""
        return len(self.nodes)
    
    def get_relationship_count(self) -> int:
        """Get the total number of relationships."""
        return len(self.relationships)
    
    def get_degree_distribution(self) -> Dict[int, int]:
        """Get the degree distribution of the graph."""
        degree_count = defaultdict(int)
        for node_id in self.nodes:
            degree = len(self.adjacency_list[node_id])
            degree_count[degree] += 1
        return dict(degree_count)
    
    def get_label_statistics(self) -> Dict[str, int]:
        """Get statistics about node labels."""
        label_count = defaultdict(int)
        for node in self.nodes.values():
            for label in node.labels:
                label_count[label] += 1
        return dict(label_count)


@dataclass
class GraphComparisonResult:
    """Results of comparing two graphs."""
    
    graph1_stats: Dict[str, Any] = field(default_factory=dict)
    graph2_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Edit distance metrics
    edit_distance: float = 0.0
    normalized_edit_distance: float = 0.0
    
    # Common subgraph metrics
    maximum_common_subgraph_size: int = 0
    mcs_ratio_graph1: float = 0.0
    mcs_ratio_graph2: float = 0.0
    
    # Supergraph metrics
    minimum_common_supergraph_size: int = 0
    supergraph_ratio_graph1: float = 0.0
    supergraph_ratio_graph2: float = 0.0
    # Structural Jaccard based on node equivalence (asset type + local relationships)
    structural_jaccard: float = 0.0
    
    # Isomorphism invariants
    node_count_difference: int = 0
    relationship_count_difference: int = 0
    degree_distribution_similarity: float = 0.0
    label_distribution_similarity: float = 0.0
    
    # Additional metrics
    common_nodes: Set[str] = field(default_factory=set)
    common_relationships: Set[str] = field(default_factory=set)
    unique_to_graph1: Set[str] = field(default_factory=set)
    unique_to_graph2: Set[str] = field(default_factory=set)
    
    def get_similarity_score(self) -> float:
        """
        Calculate an overall similarity score between 0 and 1.
        
        Returns:
            Similarity score where 1 means identical graphs
        """
        # Weighted combination of different metrics
        weights = {
            'edit_distance': 0.3,
            'mcs_ratio': 0.3,
            'degree_similarity': 0.2,
            'label_similarity': 0.2
        }
        
        # Normalize metrics to 0-1 scale
        edit_similarity = 1.0 - self.normalized_edit_distance
        mcs_similarity = (self.mcs_ratio_graph1 + self.mcs_ratio_graph2) / 2.0
        
        score = (
            weights['edit_distance'] * edit_similarity +
            weights['mcs_ratio'] * mcs_similarity +
            weights['degree_similarity'] * self.degree_distribution_similarity +
            weights['label_similarity'] * self.label_distribution_similarity
        )
        
        return max(0.0, min(1.0, score))


def neo4j_node_to_graph_node(neo4j_node) -> GraphNode:
    """
    Convert a Neo4j node object to a GraphNode.
    
    Args:
        neo4j_node: Neo4j node object
        
    Returns:
        GraphNode object
    """
    return GraphNode(
        id=neo4j_node.get('component_id', str(neo4j_node.element_id)),
        labels=set(neo4j_node.labels),
        properties=dict(neo4j_node),
        element_id=neo4j_node.element_id
    )


def neo4j_relationship_to_graph_relationship(neo4j_rel) -> GraphRelationship:
    """
    Convert a Neo4j relationship object to a GraphRelationship.
    
    Args:
        neo4j_rel: Neo4j relationship object
        
    Returns:
        GraphRelationship object
    """
    start_node = neo4j_rel.start_node
    end_node = neo4j_rel.end_node
    
    return GraphRelationship(
        id=f"{start_node.element_id}_{end_node.element_id}_{neo4j_rel.type}",
        start_node_id=start_node.get('component_id', str(start_node.element_id)),
        end_node_id=end_node.get('component_id', str(end_node.element_id)),
        relationship_type=neo4j_rel.type,
        properties=dict(neo4j_rel),
        element_id=neo4j_rel.element_id
    )
