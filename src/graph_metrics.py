"""
Graph Metrics Calculator Module

This module provides functions to calculate similarity metrics between two MACM graphs
represented as Cypher files. It can be used as a standalone library in larger applications.

Main functions:
- calculate_edit_distance: Computes the edit distance between two graphs
- calculate_maximum_common_subgraph: Finds the maximum common subgraph
- calculate_all_metrics: Computes all available metrics

Author: Daniele Granata
Date: 6 November 2025
"""

import logging
from typing import Dict, Tuple, Set, Any, Optional
from collections import Counter
from dataclasses import dataclass

from .models import Graph, GraphNode, GraphRelationship


@dataclass
class GraphMetrics:
    """Container for graph comparison metrics"""
    edit_distance: float
    normalized_edit_distance: float
    mcs_size: int
    mcs_ratio_graph1: float
    mcs_ratio_graph2: float
    supergraph_size: int
    supergraph_ratio_graph1: float
    supergraph_ratio_graph2: float
    common_nodes_count: int
    common_relationships_count: int
    node_insertions: int
    node_deletions: int
    node_type_modifications: int
    edge_insertions: int
    edge_deletions: int
    type_modifications_detail: list


class GraphMetricsCalculator:
    """
    Calculator for graph similarity metrics.
    
    This class provides methods to compute various metrics between two MACM graphs,
    including edit distance, maximum common subgraph, and structural similarity.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the calculator.
        
        Args:
            logger: Optional logger instance. If not provided, creates a default one.
        """
        self.logger = logger or logging.getLogger(__name__)
        self._last_edit_operations = {}
    
    def calculate_edit_distance(self, graph1: Graph, graph2: Graph) -> Tuple[float, Dict]:
        """
        Calculate the edit distance between two graphs.
        
        The edit distance is the minimum number of operations (insert, delete, modify)
        needed to transform one graph into another. This implementation includes
        special handling for node type modifications.
        
        Operations counted:
        - Node type modification: 1 (when structure is identical but type changes)
        - Node insertion/deletion: 1 each
        - Edge insertion/deletion: 1 each
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            Tuple of (edit_distance, operations_detail)
            where operations_detail contains breakdown of operations
        """
        # Build canonical node signature: (type, sorted(multiset of (rel_type,direction)))
        def node_signature(node: GraphNode, graph: Graph):
            t = node.get_property('type', None)
            sig = Counter()
            for rel in graph.relationships.values():
                if rel.start_node_id == node.id:
                    sig[(rel.relationship_type, 'out')] += 1
                if rel.end_node_id == node.id:
                    sig[(rel.relationship_type, 'in')] += 1
            return (t, tuple(sorted(sig.items())))

        # Build structural signature (without type) for detecting type changes
        def structural_signature(node: GraphNode, graph: Graph):
            sig = Counter()
            for rel in graph.relationships.values():
                if rel.start_node_id == node.id:
                    sig[(rel.relationship_type, 'out')] += 1
                if rel.end_node_id == node.id:
                    sig[(rel.relationship_type, 'in')] += 1
            return tuple(sorted(sig.items()))

        nodes_sig_g1 = [node_signature(n, graph1) for n in graph1.nodes.values()]
        nodes_sig_g2 = [node_signature(n, graph2) for n in graph2.nodes.values()]

        # Build maps for detecting type modifications
        struct_to_type_g1 = {}
        struct_to_type_g2 = {}
        
        for n in graph1.nodes.values():
            struct_sig = structural_signature(n, graph1)
            node_type = n.get_property('type', None)
            if struct_sig not in struct_to_type_g1:
                struct_to_type_g1[struct_sig] = []
            struct_to_type_g1[struct_sig].append(node_type)
        
        for n in graph2.nodes.values():
            struct_sig = structural_signature(n, graph2)
            node_type = n.get_property('type', None)
            if struct_sig not in struct_to_type_g2:
                struct_to_type_g2[struct_sig] = []
            struct_to_type_g2[struct_sig].append(node_type)

        # Detect type modifications
        type_modifications = []
        for struct_sig in struct_to_type_g1:
            if struct_sig in struct_to_type_g2:
                types_g1 = Counter(struct_to_type_g1[struct_sig])
                types_g2 = Counter(struct_to_type_g2[struct_sig])
                
                common_count = sum((types_g1 & types_g2).values())
                total_g1 = sum(types_g1.values())
                total_g2 = sum(types_g2.values())
                
                if total_g1 == total_g2:
                    modifications = total_g1 - common_count
                    if modifications > 0:
                        types_only_g1 = list((types_g1 - types_g2).elements())
                        types_only_g2 = list((types_g2 - types_g1).elements())
                        for i in range(modifications):
                            type_modifications.append({
                                'struct_sig': struct_sig,
                                'old_type': types_only_g1[i] if i < len(types_only_g1) else None,
                                'new_type': types_only_g2[i] if i < len(types_only_g2) else None
                            })

        c1 = Counter(nodes_sig_g1)
        c2 = Counter(nodes_sig_g2)

        raw_node_deletions = sum((c1 - c2).values())
        raw_node_insertions = sum((c2 - c1).values())
        node_type_modifications = len(type_modifications)
        
        # Adjust deletions/insertions to account for type modifications
        node_deletions = raw_node_deletions - node_type_modifications
        node_insertions = raw_node_insertions - node_type_modifications

        # Build edge signatures
        node_sig_map_g1 = {n.id: node_signature(n, graph1) for n in graph1.nodes.values()}
        node_sig_map_g2 = {n.id: node_signature(n, graph2) for n in graph2.nodes.values()}

        def edge_sig_from_map(rel: GraphRelationship, node_sig_map: Dict[str, Any]):
            start_sig = node_sig_map.get(rel.start_node_id)
            end_sig = node_sig_map.get(rel.end_node_id)
            proto = rel.properties.get('protocol') if getattr(rel, 'properties', None) is not None else None
            if proto:
                return (start_sig, rel.relationship_type, proto, end_sig)
            return (start_sig, rel.relationship_type, end_sig)

        edges_sig_g1 = [edge_sig_from_map(r, node_sig_map_g1) for r in graph1.relationships.values()]
        edges_sig_g2 = [edge_sig_from_map(r, node_sig_map_g2) for r in graph2.relationships.values()]

        ce1 = Counter(edges_sig_g1)
        ce2 = Counter(edges_sig_g2)

        raw_edge_deletions = sum((ce1 - ce2).values())
        raw_edge_insertions = sum((ce2 - ce1).values())
        
        # Adjust edge operations for type modifications
        edges_affected_by_type_mod = min(raw_edge_deletions, raw_edge_insertions, 
                                         node_type_modifications * 2)
        
        edge_deletions = max(0, raw_edge_deletions - edges_affected_by_type_mod)
        edge_insertions = max(0, raw_edge_insertions - edges_affected_by_type_mod)

        edit_distance = node_insertions + node_deletions + node_type_modifications + edge_insertions + edge_deletions

        self.logger.debug(f"Edit distance: {edit_distance} (nodes: {node_insertions + node_deletions + node_type_modifications}, edges: {edge_insertions + edge_deletions})")
        
        operations_detail = {
            "node_insertions": node_insertions,
            "node_deletions": node_deletions,
            "node_type_modifications": node_type_modifications,
            "type_modifications_detail": type_modifications,
            "edge_insertions": edge_insertions,
            "edge_deletions": edge_deletions,
            "node_signatures_g1_only": list((c1 - c2).elements()),
            "node_signatures_g2_only": list((c2 - c1).elements()),
            "edge_signatures_g1_only": list((ce1 - ce2).elements()),
            "edge_signatures_g2_only": list((ce2 - ce1).elements())
        }
        
        self._last_edit_operations = operations_detail
        
        return edit_distance, operations_detail
    
    def calculate_maximum_common_subgraph(self, graph1: Graph, graph2: Graph) -> Tuple[int, Set[str], Set[str]]:
        """
        Calculate the maximum common subgraph between two graphs.
        
        The MCS is the largest subgraph that appears in both input graphs,
        considering structural equivalence of nodes (same type and connectivity).
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            Tuple of (size, common_nodes, common_edges)
            where size is the total number of elements (nodes + edges),
            common_nodes is the set of node IDs in the MCS,
            and common_edges is the set of edge IDs in the MCS
        """
        # Find common nodes (nodes with same type and structural properties)
        common_nodes = set()
        for node1 in graph1.nodes.values():
            for node2 in graph2.nodes.values():
                if self._nodes_equivalent(node1, node2):
                    common_nodes.add(node1.id)
                    break
        
        # Find common edges (relationships between common nodes)
        common_edges = set()
        for rel_id, relationship in graph1.relationships.items():
            if (relationship.start_node_id in common_nodes and 
                relationship.end_node_id in common_nodes):
                
                # Check if equivalent relationship exists in graph2
                for rel_id2, relationship2 in graph2.relationships.items():
                    if (relationship2.start_node_id == relationship.start_node_id and
                        relationship2.end_node_id == relationship.end_node_id and
                        relationship.relationship_type == relationship2.relationship_type and
                        self._relationships_equivalent(relationship, relationship2)):
                        common_edges.add(rel_id)
                        break
        
        mcs_size = len(common_nodes) + len(common_edges)
        
        self.logger.debug(f"MCS: {len(common_nodes)} nodes, {len(common_edges)} edges, total size: {mcs_size}")
        
        return mcs_size, common_nodes, common_edges
    
    def calculate_all_metrics(self, graph1: Graph, graph2: Graph) -> GraphMetrics:
        """
        Calculate all available metrics for two graphs.
        
        This is a convenience method that computes edit distance, MCS,
        and derived metrics in a single call.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            GraphMetrics object containing all computed metrics
        """
        self.logger.info("Calculating comprehensive graph metrics")
        
        # Calculate edit distance
        edit_distance, edit_ops = self.calculate_edit_distance(graph1, graph2)
        
        # Calculate MCS
        mcs_size, common_nodes, common_edges = self.calculate_maximum_common_subgraph(graph1, graph2)
        
        # Calculate graph sizes
        size_g1 = len(graph1.nodes) + len(graph1.relationships)
        size_g2 = len(graph2.nodes) + len(graph2.relationships)
        
        # Calculate MCS ratios
        mcs_ratio_g1 = mcs_size / size_g1 if size_g1 > 0 else 0.0
        mcs_ratio_g2 = mcs_size / size_g2 if size_g2 > 0 else 0.0
        
        # Calculate supergraph size and ratios
        supergraph_size = size_g1 + size_g2 - mcs_size
        supergraph_ratio_g1 = size_g1 / supergraph_size if supergraph_size > 0 else 0.0
        supergraph_ratio_g2 = size_g2 / supergraph_size if supergraph_size > 0 else 0.0
        
        # Calculate normalized edit distance
        max_edit = size_g1 + size_g2
        normalized_edit_distance = edit_distance / max_edit if max_edit > 0 else 0.0
        
        self.logger.info(f"Metrics calculated: Edit Distance={edit_distance:.3f}, MCS={mcs_size}, MCS Ratio={mcs_ratio_g1:.3f}")
        
        return GraphMetrics(
            edit_distance=edit_distance,
            normalized_edit_distance=normalized_edit_distance,
            mcs_size=mcs_size,
            mcs_ratio_graph1=mcs_ratio_g1,
            mcs_ratio_graph2=mcs_ratio_g2,
            supergraph_size=supergraph_size,
            supergraph_ratio_graph1=supergraph_ratio_g1,
            supergraph_ratio_graph2=supergraph_ratio_g2,
            common_nodes_count=len(common_nodes),
            common_relationships_count=len(common_edges),
            node_insertions=edit_ops["node_insertions"],
            node_deletions=edit_ops["node_deletions"],
            node_type_modifications=edit_ops["node_type_modifications"],
            edge_insertions=edit_ops["edge_insertions"],
            edge_deletions=edit_ops["edge_deletions"],
            type_modifications_detail=edit_ops["type_modifications_detail"]
        )
    
    def _nodes_equivalent(self, node1: GraphNode, node2: GraphNode) -> bool:
        """
        Check if two nodes are structurally equivalent.
        
        Nodes are considered equivalent if they have the same type property.
        This is used for MCS calculation.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            True if nodes are equivalent, False otherwise
        """
        type1 = node1.get_property('type', None)
        type2 = node2.get_property('type', None)
        return type1 == type2 and type1 is not None
    
    def _relationships_equivalent(self, rel1: GraphRelationship, rel2: GraphRelationship) -> bool:
        """
        Check if two relationships are equivalent.
        
        Relationships are considered equivalent if they have the same type
        and protocol properties (if present).
        
        Args:
            rel1: First relationship
            rel2: Second relationship
            
        Returns:
            True if relationships are equivalent, False otherwise
        """
        if rel1.relationship_type != rel2.relationship_type:
            return False
        
        # Check protocol properties if they exist
        proto1 = rel1.properties.get('protocol') if hasattr(rel1, 'properties') and rel1.properties else None
        proto2 = rel2.properties.get('protocol') if hasattr(rel2, 'properties') and rel2.properties else None
        
        return proto1 == proto2


# Standalone convenience functions

def calculate_edit_distance_from_cypher(
    cypher_file1: str, 
    cypher_file2: str,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None
) -> Tuple[float, Dict]:
    """
    Calculate edit distance directly from two Cypher files.
    
    This is a convenience function that handles all the setup automatically.
    Uses .env file for configuration if parameters are not provided.
    
    Args:
        cypher_file1: Path to first Cypher file
        cypher_file2: Path to second Cypher file
        neo4j_uri: Neo4j connection URI (optional, from .env if not provided)
        neo4j_user: Neo4j username (optional, from .env if not provided)
        neo4j_password: Neo4j password (optional, from .env if not provided)
        
    Returns:
        Tuple of (edit_distance, operations_detail)
    """
    from .database_manager import DatabaseManager
    from .utils import load_graph_from_cypher
    
    # Use defaults if not provided
    neo4j_uri = neo4j_uri or "bolt://localhost:7687"
    neo4j_user = neo4j_user or "neo4j"
    neo4j_password = neo4j_password or "password"
    
    # Setup
    db_manager = DatabaseManager(neo4j_uri, neo4j_user, neo4j_password)
    calculator = GraphMetricsCalculator()
    
    try:
        db_manager.connect()
        # Load graphs (note: load_graph_from_cypher signature is (manager, file_path))
        graph1 = load_graph_from_cypher(db_manager, cypher_file1)
        graph2 = load_graph_from_cypher(db_manager, cypher_file2)
        
        # Calculate metric
        return calculator.calculate_edit_distance(graph1, graph2)
    finally:
        db_manager.disconnect()


def calculate_maximum_common_subgraph_from_cypher(
    cypher_file1: str, 
    cypher_file2: str,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None
) -> Tuple[int, Set[str], Set[str]]:
    """
    Calculate maximum common subgraph directly from two Cypher files.
    
    This is a convenience function that handles all the setup automatically.
    Uses .env file for configuration if parameters are not provided.
    
    Args:
        cypher_file1: Path to first Cypher file
        cypher_file2: Path to second Cypher file
        neo4j_uri: Neo4j connection URI (optional, from .env if not provided)
        neo4j_user: Neo4j username (optional, from .env if not provided)
        neo4j_password: Neo4j password (optional, from .env if not provided)
        
    Returns:
        Tuple of (size, common_nodes, common_edges)
    """
    from .database_manager import DatabaseManager
    from .utils import load_graph_from_cypher
    
    # Use defaults if not provided
    neo4j_uri = neo4j_uri or "bolt://localhost:7687"
    neo4j_user = neo4j_user or "neo4j"
    neo4j_password = neo4j_password or "password"
    
    # Setup
    db_manager = DatabaseManager(neo4j_uri, neo4j_user, neo4j_password)
    calculator = GraphMetricsCalculator()
    
    try:
        db_manager.connect()
        # Load graphs (note: load_graph_from_cypher signature is (manager, file_path))
        graph1 = load_graph_from_cypher(db_manager, cypher_file1)
        graph2 = load_graph_from_cypher(db_manager, cypher_file2)
        
        # Calculate metric
        return calculator.calculate_maximum_common_subgraph(graph1, graph2)
    finally:
        db_manager.disconnect()


def calculate_all_metrics_from_cypher(
    cypher_file1: str, 
    cypher_file2: str,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None
) -> GraphMetrics:
    """
    Calculate all metrics directly from two Cypher files.
    
    This is a convenience function that handles all the setup automatically.
    Uses .env file for configuration if parameters are not provided.
    
    Args:
        cypher_file1: Path to first Cypher file
        cypher_file2: Path to second Cypher file
        neo4j_uri: Neo4j connection URI (optional, from .env if not provided)
        neo4j_user: Neo4j username (optional, from .env if not provided)
        neo4j_password: Neo4j password (optional, from .env if not provided)
        
    Returns:
        GraphMetrics object with all computed metrics
    """
    from .database_manager import DatabaseManager
    from .utils import load_graph_from_cypher
    
    # Use defaults if not provided
    neo4j_uri = neo4j_uri or "bolt://localhost:7687"
    neo4j_user = neo4j_user or "neo4j"
    neo4j_password = neo4j_password or "password"
    
    # Setup
    db_manager = DatabaseManager(neo4j_uri, neo4j_user, neo4j_password)
    calculator = GraphMetricsCalculator()
    
    try:
        db_manager.connect()
        # Load graphs (note: load_graph_from_cypher signature is (manager, file_path))
        graph1 = load_graph_from_cypher(db_manager, cypher_file1)
        graph2 = load_graph_from_cypher(db_manager, cypher_file2)
        
        # Calculate all metrics
        return calculator.calculate_all_metrics(graph1, graph2)
    finally:
        db_manager.disconnect()
