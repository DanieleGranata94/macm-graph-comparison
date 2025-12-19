"""
Core metrics for comparing MACM graphs.

Implements Graph Edit Distance (GED) and Maximum Common Subgraph (MCS) metrics.
"""
from typing import Tuple, Set
from collections import Counter
from .models import Graph, GraphNode, GraphRelationship


def node_signature(node: GraphNode, graph: Graph) -> Tuple:
    """
    Calculate structural signature for a node.
    
    Signature = (type, sorted_relations)
    where sorted_relations = sorted list of (relation_type, direction, count)
    
    Example: ('Service.API', (('connects', 'in', 1), ('uses', 'out', 2)))
    
    IMPORTANT: Name is intentionally NOT included to support LLM use case
    where semantically equivalent components may have different names.
    """
    node_type = node.get_property('type', None)
    sig = Counter()
    
    for rel in graph.relationships.values():
        if rel.start_node_id == node.id:
            sig[(rel.relationship_type, 'out')] += 1
        if rel.end_node_id == node.id:
            sig[(rel.relationship_type, 'in')] += 1
    
    return (node_type, tuple(sorted(sig.items())))


def calculate_edit_distance(graph1: Graph, graph2: Graph) -> int:
    """
    Calculate Graph Edit Distance using signature-based matching.
    
    Returns the minimum number of operations (add/delete/modify) needed
    to transform graph1 into graph2.
    
    KNOWN LIMITATION: Uses Counter-based matching which loses node identity.
    This means graphs with same node signatures but different topology
    may incorrectly show ED=0. See ALGORITHM_ISSUES.md for details.
    """
    nodes_sig_g1 = [node_signature(n, graph1) for n in graph1.nodes.values()]
    nodes_sig_g2 = [node_signature(n, graph2) for n in graph2.nodes.values()]
    
    c1 = Counter(nodes_sig_g1)
    c2 = Counter(nodes_sig_g2)
    
    node_deletions = sum((c1 - c2).values())
    node_insertions = sum((c2 - c1).values())
    
    # Create node mapping for edge comparison
    node_mapping = {}
    used_nodes_g2 = set()
    
    for node1 in graph1.nodes.values():
        sig1 = node_signature(node1, graph1)
        for node2 in graph2.nodes.values():
            if node2.id not in used_nodes_g2:
                sig2 = node_signature(node2, graph2)
                if sig1 == sig2:
                    node_mapping[node1.id] = node2.id
                    used_nodes_g2.add(node2.id)
                    break
    
    # Calculate edge differences
    edges_g1 = set()
    for rel in graph1.relationships.values():
        if rel.start_node_id in node_mapping and rel.end_node_id in node_mapping:
            mapped_edge = (
                node_mapping[rel.start_node_id],
                rel.relationship_type,
                node_mapping[rel.end_node_id]
            )
            edges_g1.add(mapped_edge)
    
    edges_g2 = set()
    for rel in graph2.relationships.values():
        edge = (rel.start_node_id, rel.relationship_type, rel.end_node_id)
        edges_g2.add(edge)
    
    edge_deletions = len(edges_g1 - edges_g2)
    edge_insertions = len(edges_g2 - edges_g1)
    
    edit_distance = node_deletions + node_insertions + edge_deletions + edge_insertions
    
    return edit_distance


def nodes_equivalent(node1: GraphNode, node2: GraphNode) -> bool:
    """Check if two nodes are equivalent (same type, ignoring name)."""
    type1 = node1.get_property('type', None)
    type2 = node2.get_property('type', None)
    return type1 == type2


def calculate_mcs(graph1: Graph, graph2: Graph) -> Tuple[int, Set[str], Set[str]]:
    """
    Calculate Maximum Common Subgraph.
    
    Returns:
        (mcs_size, common_nodes, common_edges)
        where mcs_size = len(common_nodes) + len(common_edges)
    """
    common_nodes = set()
    node_mapping = {}
    
    for node1 in graph1.nodes.values():
        for node2 in graph2.nodes.values():
            if node2.id not in node_mapping.values() and nodes_equivalent(node1, node2):
                common_nodes.add(node1.id)
                node_mapping[node1.id] = node2.id
                break
    
    common_edges = set()
    for rel_id, relationship in graph1.relationships.items():
        if (relationship.start_node_id in common_nodes and 
            relationship.end_node_id in common_nodes):
            
            start_node_id_g2 = node_mapping.get(relationship.start_node_id)
            end_node_id_g2 = node_mapping.get(relationship.end_node_id)
            
            if start_node_id_g2 and end_node_id_g2:
                for relationship2 in graph2.relationships.values():
                    if (relationship2.start_node_id == start_node_id_g2 and
                        relationship2.end_node_id == end_node_id_g2 and
                        relationship.relationship_type == relationship2.relationship_type):
                        common_edges.add(rel_id)
                        break
    
    mcs_size = len(common_nodes) + len(common_edges)
    
    return mcs_size, common_nodes, common_edges


def calculate_mcs_ratio(graph1: Graph, graph2: Graph) -> Tuple[float, float]:
    """
    Calculate MCS ratio relative to each graph.
    
    Returns:
        (ratio_g1, ratio_g2) as decimals in range [0.0, 1.0]
        
        ratio_g1 = mcs_size / (nodes_g1 + edges_g1)
        ratio_g2 = mcs_size / (nodes_g2 + edges_g2)
    """
    mcs_size, _, _ = calculate_mcs(graph1, graph2)
    
    size_g1 = len(graph1.nodes) + len(graph1.relationships)
    size_g2 = len(graph2.nodes) + len(graph2.relationships)
    
    ratio_g1 = mcs_size / size_g1 if size_g1 > 0 else 0.0
    ratio_g2 = mcs_size / size_g2 if size_g2 > 0 else 0.0
    
    return ratio_g1, ratio_g2
