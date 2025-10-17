def plot_jetracer_all_metrics(result_correct, result_incorrect, output_dir="output"):
    """
    Genera un unico grafico a barre con metriche normalizzate tra 0 e 1 per la coppia JetRacer.
    Args:
        result_correct: GraphComparisonResult per JetRacerMACM_correct.macm
        result_incorrect: GraphComparisonResult per JetRacerMACM_incorrect.macm
        output_dir: directory dove salvare il grafico
    """
    import matplotlib.pyplot as plt
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calcolo il valore di riferimento (dimensione massima tra le due architetture)
    size_graph1 = sum(result_correct.graph1_stats.values()) if result_correct.graph1_stats else 0
    size_graph2 = sum(result_correct.graph2_stats.values()) if result_correct.graph2_stats else 0
    reference_value = max(size_graph1, size_graph2)
    # Metriche normalizzate rispetto al valore di riferimento
    metrics = [
        result_correct.mcs_ratio_graph1
    ]
    metric_labels = ["Maximum Common Subgraph"]
    colors = ["#1976d2"]


def plot_all_metrics_raw(result, output_dir="output"):
    """Genera un grafico con tutte le metriche raw (senza normalizzazione).
    Mostra i valori così come sono nel `GraphComparisonResult`.
    """
    import matplotlib.pyplot as plt
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Raccogli le metriche raw
    metrics = [
        result.mcs_ratio_graph1,
        result.supergraph_ratio_graph1,
        result.edit_distance
    ]
    labels = ["Maximum Common Subgraph", "Minimum Common Supergraph", "Edit Distance"]
    colors = ["#1976d2", "#ffa000", "#F44336"]

    plt.figure(figsize=(9,6))
    plt.style.use('seaborn-v0_8-darkgrid')
    bars = plt.bar(labels, metrics, color=colors, edgecolor='black', width=0.6)
    plt.ylabel("Metric Value (raw)", fontsize=13)
    plt.title("All Metrics (Raw Values)", fontsize=15, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(0.01, abs(height)*0.02), f"{height:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout(pad=3)
    path = os.path.join(output_dir, "all_metrics_raw.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Chart saved as {path}")


def plot_all_metrics_normalized(result, output_dir="output"):
    """Genera un grafico con le 4 metriche normalizzate nello stesso intervallo [0,1].
    - Similarity Score: già 0-1
    - MCS Ratio: clipped a [0,1]
    - Supergraph Ratio: normalizzato/clipped a [0,1]
    - Edit Distance: convertita in similarità come 1 - (edit/max_possible)
    """
    import matplotlib.pyplot as plt
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # stima max edit distance
    max_edit_distance = None
    try:
        max_edit_distance = (sum(result.graph1_stats.values()) + sum(result.graph2_stats.values())) if result.graph1_stats and result.graph2_stats else None
    except Exception:
        max_edit_distance = None
    if not max_edit_distance or max_edit_distance <= 0:
        max_edit_distance = 100.0

    mcs = float(getattr(result, 'mcs_ratio_graph1', 0.0))
    super_r = float(getattr(result, 'supergraph_ratio_graph1', 0.0))

    # Normalize / clip
    mcs_n = max(0.0, min(1.0, mcs))
    super_n = max(0.0, min(1.0, super_r))

    metrics = [mcs_n, super_n]
    labels = ["Maximum Common Subgraph", "Minimum Common Supergraph"]
    colors = ["#43a047", "#1976d2"]

    plt.figure(figsize=(9,6))
    plt.style.use('seaborn-v0_8-darkgrid')
    bars = plt.bar(labels, metrics, color=colors, edgecolor='black', width=0.6)
    plt.ylim(0, 1.05)
    plt.ylabel("Normalized Metric Value (0-1)", fontsize=13)
    plt.title("All Metrics Normalized to [0,1]", fontsize=15, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout(pad=3)
    path = os.path.join(output_dir, "all_metrics_normalized_0_1.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Chart saved as {path}")

def plot_similarity_metrics_normalized(result, output_dir="output"):
    """
    Genera un grafico a barre solo con le metriche normalizzate tra 0 e 1, senza Supergraph Ratio.
    Args:
        result: GraphComparisonResult
        output_dir: directory dove salvare il grafico
    """
    import matplotlib.pyplot as plt
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metrics = [
        result.mcs_ratio_graph1
    ]
    metric_labels = ["Maximum Common Subgraph"]
    colors = ["#1976d2"]

    plt.figure(figsize=(7,6))
    plt.style.use('seaborn-v0_8-darkgrid')
    bars = plt.bar(metric_labels, metrics, color=colors, edgecolor='black', width=0.6)

    plt.ylim(0, 1.05)
    plt.ylabel("Normalized Metric Value (0-1)", fontsize=13)
    plt.title("Normalized Similarity Metrics", fontsize=15, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.04, f"{height:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(output_dir, "basic_metrics_range.png"), dpi=120)
    plt.close()
    print(f"Chart saved as {output_dir}/basic_metrics_range.png")

def plot_normalized_to_identity(result, output_dir="output"):
    """
    Genera un grafico a barre con tutte le metriche normalizzate rispetto al loro valore ideale (grafi identici).
    - Similarity Score: normalizzato rispetto a 1.0 (massima similarità)
    - MCS Ratio: normalizzato rispetto a 1.0 (massima sovrapposizione)
    - Edit Distance: invertita e normalizzata (1 - edit_distance/max_possible) per tendere a 1.0
    
    Args:
        result: GraphComparisonResult
        output_dir: directory dove salvare il grafico
    """
    import matplotlib.pyplot as plt
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calcola il massimo valore possibile di edit distance (somma dei nodi dei due grafi)
    max_edit_distance = sum(result.graph1_stats.values()) + sum(result.graph2_stats.values()) if result.graph1_stats and result.graph2_stats else 100

    # Normalizza edit distance invertendola (così tende a 1.0 quando i grafi sono identici)
    normalized_edit = max(0, 1 - (result.edit_distance / max_edit_distance))

    # Debug: stampa i valori per capire il problema
    print(f"Debug - Edit Distance: {result.edit_distance}")
    print(f"Debug - Graph1 stats: {result.graph1_stats}")
    print(f"Debug - Graph2 stats: {result.graph2_stats}")
    print(f"Debug - Max edit distance: {max_edit_distance}")
    print(f"Debug - Normalized edit: {normalized_edit}")

    metrics = [
        result.mcs_ratio_graph1,        # già normalizzato tra 0 e 1
        result.supergraph_ratio_graph1, # già normalizzato tra 0 e 1
        normalized_edit                 # normalizzato per tendere a 1 quando edit_distance = 0
    ]
    
    metric_labels = ["Maximum Common Subgraph", "Minimum Common Supergraph", "Normalized Edit Distance"]
    colors = ["#1976d2", "#ffa000", "#F44336"]

    plt.figure(figsize=(8,6))
    plt.style.use('seaborn-v0_8-darkgrid')
    bars = plt.bar(metric_labels, metrics, color=colors, edgecolor='black', width=0.6)

    plt.ylim(0, 1.05)
    plt.ylabel("Distance from Identity (1.0 = identical graphs)", fontsize=13)
    plt.title("Metrics Normalized to Identity", fontsize=15, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.04, f"{height:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(output_dir, "identity_normalized_metrics.png"), dpi=120)
    plt.close()
    print(f"Chart saved as {output_dir}/identity_normalized_metrics.png")

import logging
from typing import Dict, Set, List, Tuple, Any
from collections import defaultdict, Counter

import logging
from typing import Dict, Set, List, Tuple, Any
from collections import defaultdict, Counter
from typing import List, Dict, Any
import math

from models import Graph, GraphNode, GraphRelationship, GraphComparisonResult

# Nuova funzione per generare grafici comparativi tra i due JetRacer
import matplotlib.pyplot as plt
import os

def plot_jetracer_comparison(result_correct, result_incorrect, output_dir="output"):
    """
    Genera grafici comparativi tra JetRacerMACM_correct.macm e JetRacerMACM_incorrect.macm.
    Args:
        result_correct: GraphComparisonResult per JetRacerMACM_correct.macm
        result_incorrect: GraphComparisonResult per JetRacerMACM_incorrect.macm
        output_dir: directory dove salvare i grafici
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


class GraphMetricsCalculator:
    """Calculates various metrics for graph comparison."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_edit_distance(self, graph1: Graph, graph2: Graph) -> float:
        """
        Calculate the edit distance between two graphs.
        
        The edit distance is the minimum number of operations (add, delete, substitute)
        needed to transform one graph into another.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            Edit distance value
        """
        # Use canonical signatures for nodes (asset type + local relation multiset)
        # and compute edit distance as differences in multisets of node- and edge-signatures.
        # This respects the requested equivalence: nodes can have different names but the
        # same 'type' and same local relations are considered identical.

        # Build canonical node signature for each node: (type, sorted(multiset of (rel_type,direction)))
        def node_signature(node: GraphNode, graph: Graph):
            t = node.get_property('type', None)
            sig = Counter()
            for rel in graph.relationships.values():
                if rel.start_node_id == node.id:
                    sig[(rel.relationship_type, 'out')] += 1
                if rel.end_node_id == node.id:
                    sig[(rel.relationship_type, 'in')] += 1
            return (t, tuple(sorted(sig.items())))

        nodes_sig_g1 = [node_signature(n, graph1) for n in graph1.nodes.values()]
        nodes_sig_g2 = [node_signature(n, graph2) for n in graph2.nodes.values()]

        c1 = Counter(nodes_sig_g1)
        c2 = Counter(nodes_sig_g2)

        node_deletions = sum((c1 - c2).values())
        node_insertions = sum((c2 - c1).values())
        node_substitutions = 0  # canonical signatures capture substitution as deletion+insertion

        # Build edge signatures using node canonical signatures instead of raw node ids
        # Map node id -> canonical signature for each graph
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

        edge_deletions = sum((ce1 - ce2).values())
        edge_insertions = sum((ce2 - ce1).values())
        edge_substitutions = 0

        edit_distance = node_insertions + node_deletions + node_substitutions + edge_insertions + edge_deletions + edge_substitutions

        self.logger.debug(f"Edit distance (structural) node_ops={node_insertions + node_deletions}, edge_ops={edge_insertions + edge_deletions}")
        
        # Store detailed operations for JSON export
        self._last_edit_operations = {
            "node_insertions": node_insertions,
            "node_deletions": node_deletions, 
            "edge_insertions": edge_insertions,
            "edge_deletions": edge_deletions,
            "node_signatures_g1_only": list((c1 - c2).elements()),
            "node_signatures_g2_only": list((c2 - c1).elements()),
            "edge_signatures_g1_only": list((ce1 - ce2).elements()),
            "edge_signatures_g2_only": list((ce2 - ce1).elements())
        }
        
        return edit_distance
    
    def calculate_maximum_common_subgraph(self, graph1: Graph, graph2: Graph) -> Tuple[int, Set[str], Set[str]]:
        """
        Calculate the maximum common subgraph between two graphs.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            Tuple of (size, common_nodes, common_edges)
        """
        # Find common nodes (nodes with same type, x, y properties)
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
        
        self.logger.debug(f"MCS calculation: {len(common_nodes)} nodes, {len(common_edges)} edges")
        
        return mcs_size, common_nodes, common_edges
    
    def calculate_minimum_common_supergraph(self, graph1: Graph, graph2: Graph) -> int:
        """
        Calculate the size of the minimum common supergraph.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            Size of the minimum common supergraph
        """
        # The minimum common supergraph contains all nodes and edges from both graphs
        # minus the common parts (to avoid double counting)
        
        mcs_size, common_nodes, common_edges = self.calculate_maximum_common_subgraph(graph1, graph2)
        
        # Total nodes and edges in both graphs
        total_nodes = len(graph1.nodes) + len(graph2.nodes) - len(common_nodes)
        total_edges = len(graph1.relationships) + len(graph2.relationships) - len(common_edges)
        
        supergraph_size = total_nodes + total_edges
        
        self.logger.debug(f"Supergraph calculation: {total_nodes} nodes, {total_edges} edges")
        
        return supergraph_size

    def calculate_structural_jaccard(self, graph1: Graph, graph2: Graph) -> float:
        """
        Calculate a structural Jaccard similarity between two graphs where nodes are
        considered equal if they have the same asset 'type' and matching local
        relationship signatures (as defined by _nodes_equivalent).

        Returns a float in [0,1].
        """
        # Build sets of canonical node signatures for each graph based on equivalence
        def canonical_nodes(graph: Graph) -> List[Tuple[str, Counter]]:
            nodes_canon = []
            for node in graph.nodes.values():
                # type
                t = node.get_property('type', None)
                # local relation signature (multiset)
                sig = Counter()
                for rel in graph.relationships.values():
                    if rel.start_node_id == node.id:
                        sig[(rel.relationship_type, 'out')] += 1
                    if rel.end_node_id == node.id:
                        sig[(rel.relationship_type, 'in')] += 1
                nodes_canon.append((t, tuple(sorted(sig.items()))))
            return nodes_canon

        nodes1 = canonical_nodes(graph1)
        nodes2 = canonical_nodes(graph2)

        # Use multisets (Counter) to account for multiplicities of identical signatures
        c1 = Counter(nodes1)
        c2 = Counter(nodes2)

        inter = sum((c1 & c2).values())
        union = sum((c1 | c2).values())
        if union == 0:
            return 0.0
        return inter / union
    
    def calculate_isomorphism_invariants(self, graph1: Graph, graph2: Graph) -> Dict[str, float]:
        """
        Calculate isomorphism invariants to compare graphs.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            Dictionary of invariant metrics
        """
        # Node count difference
        node_count_diff = abs(len(graph1.nodes) - len(graph2.nodes))
        
        # Relationship count difference
        rel_count_diff = abs(len(graph1.relationships) - len(graph2.relationships))
        
        # Degree distribution similarity
        degree_dist1 = graph1.get_degree_distribution()
        degree_dist2 = graph2.get_degree_distribution()
        degree_similarity = self._calculate_distribution_similarity(degree_dist1, degree_dist2)
        
        # Label distribution similarity
        label_dist1 = graph1.get_label_statistics()
        label_dist2 = graph2.get_label_statistics()
        label_similarity = self._calculate_distribution_similarity(label_dist1, label_dist2)
        
        # Relationship type distribution
        rel_type_dist1 = self._get_relationship_type_distribution(graph1)
        rel_type_dist2 = self._get_relationship_type_distribution(graph2)
        rel_type_similarity = self._calculate_distribution_similarity(rel_type_dist1, rel_type_dist2)
        
        return {
            'node_count_difference': node_count_diff,
            'relationship_count_difference': rel_count_diff,
            'degree_distribution_similarity': degree_similarity,
            'label_distribution_similarity': label_similarity,
            'relationship_type_similarity': rel_type_similarity
        }
    
    def compare_graphs(self, graph1: Graph, graph2: Graph) -> GraphComparisonResult:
        """
        Perform comprehensive comparison between two graphs.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            GraphComparisonResult with all metrics
        """
        self.logger.info("Starting comprehensive graph comparison")
        
        # Calculate edit distance
        edit_distance = self.calculate_edit_distance(graph1, graph2)
        
        # Calculate maximum common subgraph
        mcs_size, common_nodes, common_edges = self.calculate_maximum_common_subgraph(graph1, graph2)
        
        # Calculate minimum common supergraph
        supergraph_size = self.calculate_minimum_common_supergraph(graph1, graph2)
        
        # Calculate isomorphism invariants
        invariants = self.calculate_isomorphism_invariants(graph1, graph2)
        
        # Calculate ratios
        # Use total elements (nodes + relationships) as reference for edit operations
        total_elements_g1 = len(graph1.nodes) + len(graph1.relationships)
        total_elements_g2 = len(graph2.nodes) + len(graph2.relationships)
        max_graph_size = max(total_elements_g1, total_elements_g2)
        if max_graph_size > 0:
            normalized_edit_distance = edit_distance / max_graph_size
            # Clamp to [0,1] so similarity measures remain interpretable
            normalized_edit_distance = max(0.0, min(1.0, normalized_edit_distance))
        else:
            normalized_edit_distance = 0
        
        # MCS ratio should be calculated as: mcs_size / (nodes + relationships) of each graph
        graph1_size = len(graph1.nodes) + len(graph1.relationships)
        graph2_size = len(graph2.nodes) + len(graph2.relationships)
        mcs_ratio_graph1 = mcs_size / graph1_size if graph1_size > 0 else 0
        mcs_ratio_graph2 = mcs_size / graph2_size if graph2_size > 0 else 0

        # Compute structural Jaccard (based on asset type + local relations) and
        # use it as the interpretable similarity measure in place of raw supergraph ratio
        structural_jacc = self.calculate_structural_jaccard(graph1, graph2)
        supergraph_ratio_graph1 = structural_jacc
        supergraph_ratio_graph2 = structural_jacc
        
        # Find unique elements based on names
        nodes1_names = set()
        nodes2_names = set()
        
        for node in graph1.nodes.values():
            name = node.get_property('name', '')
            if name:  # Only add non-empty names
                nodes1_names.add(name)
        
        for node in graph2.nodes.values():
            name = node.get_property('name', '')
            if name:  # Only add non-empty names
                nodes2_names.add(name)
        
        edges1 = set(self._get_edge_signatures(graph1))
        edges2 = set(self._get_edge_signatures(graph2))
        
        unique_to_graph1 = (nodes1_names - nodes2_names) | (edges1 - edges2)
        unique_to_graph2 = (nodes2_names - nodes1_names) | (edges2 - edges1)
        
        result = GraphComparisonResult(
            graph1_stats=graph1.get_label_statistics(),
            graph2_stats=graph2.get_label_statistics(),
            edit_distance=edit_distance,
            normalized_edit_distance=normalized_edit_distance,
            maximum_common_subgraph_size=mcs_size,
            mcs_ratio_graph1=mcs_ratio_graph1,
            mcs_ratio_graph2=mcs_ratio_graph2,
            minimum_common_supergraph_size=supergraph_size,
            supergraph_ratio_graph1=supergraph_ratio_graph1,
            supergraph_ratio_graph2=supergraph_ratio_graph2,
            structural_jaccard=structural_jacc,
            node_count_difference=invariants['node_count_difference'],
            relationship_count_difference=invariants['relationship_count_difference'],
            degree_distribution_similarity=invariants['degree_distribution_similarity'],
            label_distribution_similarity=invariants['label_distribution_similarity'],
            common_nodes=common_nodes,
            common_relationships=common_edges,
            unique_to_graph1=unique_to_graph1,
            unique_to_graph2=unique_to_graph2
        )
        
        # Store edit operations details in result for JSON export
        if hasattr(self, '_last_edit_operations'):
            result.edit_operations_detail = self._last_edit_operations
        
        self.logger.info(f"Graph comparison completed. Edit distance: {result.edit_distance:.3f}")
        
        return result
    
    def _nodes_equivalent(self, node1: GraphNode, node2: GraphNode) -> bool:
        """Check if two nodes are equivalent.

        New rules (requested):
        - Node names may differ.
        - Asset type (property 'type') must coincide.
        - The set of relationship types and connectivity around the node should match
          (i.e., same multiset of (rel_type, direction) for edges incident to the node).

        This function is intentionally conservative: returns True only if asset type
        matches and the local relationship signatures are equivalent.
        """
        # Asset type must match
        type1 = node1.get_property('type', None)
        type2 = node2.get_property('type', None)
        if type1 is None or type2 is None or type1 != type2:
            return False

        # Compare local relationships: build a multiset of (rel_type, direction)
        # direction: 'out' if node is start_node, 'in' if node is end_node
        def local_rel_signatures(node: GraphNode, graph: Graph) -> Counter:
            sigs = Counter()
            for rel in graph.relationships.values():
                if rel.start_node_id == node.id:
                    sigs[(rel.relationship_type, 'out')] += 1
                if rel.end_node_id == node.id:
                    sigs[(rel.relationship_type, 'in')] += 1
            return sigs

        # We need access to the graphs to compute local relations. As a fallback
        # if the Graph objects are not globally available here, we conservatively
        # compare only the existence of the same key properties (primary/secondary).
        # However, within this codebase _nodes_equivalent is always called from
        # methods that have graph context; to avoid changing all call sites we
        # check for an attached attribute 'graph' on nodes (not guaranteed).
        graph1 = getattr(node1, '_graph', None)
        graph2 = getattr(node2, '_graph', None)

        if graph1 is not None and graph2 is not None:
            sigs1 = local_rel_signatures(node1, graph1)
            sigs2 = local_rel_signatures(node2, graph2)
            if sigs1 != sigs2:
                return False
        else:
            # Fallback: compare a small set of key properties if relation context is unavailable
            key_properties = ['primary', 'secondary']
            for prop in key_properties:
                if node1.get_property(prop) != node2.get_property(prop):
                    return False

        return True
    
    def _relationships_equivalent(self, rel1: GraphRelationship, rel2: GraphRelationship) -> bool:
        """Check if two relationships are equivalent."""
        # Compare type and endpoints
        if not (rel1.relationship_type == rel2.relationship_type and
                rel1.start_node_id == rel2.start_node_id and
                rel1.end_node_id == rel2.end_node_id):
            return False

        # If a 'protocol' property is present on either relationship, require equality
        proto1 = getattr(rel1, 'properties', {}).get('protocol') if getattr(rel1, 'properties', None) is not None else None
        proto2 = getattr(rel2, 'properties', {}).get('protocol') if getattr(rel2, 'properties', None) is not None else None
        if proto1 is not None or proto2 is not None:
            return proto1 == proto2

        return True
    
    def _get_edge_signatures(self, graph: Graph) -> List[str]:
        """Get unique signatures for all edges in the graph."""
        signatures = []
        for relationship in graph.relationships.values():
            # include protocol in signature if present -> makes signature stricter
            proto = relationship.properties.get('protocol') if getattr(relationship, 'properties', None) is not None else None
            if proto:
                signature = f"{relationship.start_node_id}-{relationship.relationship_type}:{proto}-{relationship.end_node_id}"
            else:
                signature = f"{relationship.start_node_id}-{relationship.relationship_type}-{relationship.end_node_id}"
            signatures.append(signature)
        return signatures
    
    def _find_edge_by_signature(self, graph: Graph, signature: str) -> GraphRelationship:
        """Find a relationship by its signature."""
        parts = signature.split('-')
        if len(parts) >= 3:
            start_node, rel_type, end_node = parts[0], parts[1], '-'.join(parts[2:])
            for relationship in graph.relationships.values():
                if (relationship.start_node_id == start_node and
                    relationship.end_node_id == end_node and
                    relationship.relationship_type == rel_type):
                    return relationship
        return None
    
    def _calculate_distribution_similarity(self, dist1: Dict, dist2: Dict) -> float:
        """Calculate similarity between two distributions using cosine similarity."""
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        if not all_keys:
            return 1.0
        
        vector1 = [dist1.get(key, 0) for key in all_keys]
        vector2 = [dist2.get(key, 0) for key in all_keys]
        
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(a * a for a in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _get_relationship_type_distribution(self, graph: Graph) -> Dict[str, int]:
        """Get distribution of relationship types in the graph."""
        type_counts = Counter()
        for relationship in graph.relationships.values():
            type_counts[relationship.relationship_type] += 1
        return dict(type_counts)





def generate_graph_visualizations(graph1: Graph, graph2: Graph, output_dir="output/macm_output", file1_path="", file2_path="", comparison_result=None):
    """
    Genera visualizzazioni PNG dei grafi di input utilizzando NetworkX e Matplotlib con layout ELK.
    Evidenzia i nodi e gli archi unici di ciascun grafo.
    
    Args:
        graph1: Il primo grafo da visualizzare
        graph2: Il secondo grafo da visualizzare
        output_dir: Directory dove salvare le immagini PNG (default: "output/macm_output")
        file1_path: Path del primo file per determinare se è corretto/scorretto
        file2_path: Path del secondo file per determinare se è corretto/scorretto
        comparison_result: Risultato del confronto per evidenziare le differenze
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import os
        from pathlib import Path
        from collections import Counter
        
        # Crea la cartella di output se non esiste
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Generando visualizzazioni dei grafi in {output_dir}...")
        
        # Calcola le differenze se non fornite
        if comparison_result is None:
            metrics = GraphMetricsCalculator()
            comparison_result = metrics.compare_graphs(graph1, graph2)
        
        # Estrae nodi e relazioni uniche per ciascun grafo
        unique_nodes_g1 = set()
        unique_nodes_g2 = set()
        unique_relations_g1 = set()
        unique_relations_g2 = set()
        
        if hasattr(comparison_result, 'unique_to_graph1') and hasattr(comparison_result, 'unique_to_graph2'):
            # Separa nodi e relazioni dagli elementi unici
            for item in comparison_result.unique_to_graph1:
                if '-' in str(item) and len(str(item).split('-')) >= 3:  # Potenziale relazione
                    # Prova a parsare come relazione (es: "UE-hosts-eWeLink_APP")
                    parts = str(item).split('-')
                    if len(parts) >= 3:
                        start_node = parts[0]
                        rel_type = parts[1]
                        end_node = '-'.join(parts[2:])  # Nel caso ci siano trattini nel nome finale
                        unique_relations_g1.add((start_node, end_node, rel_type))
                else:
                    # È un nodo
                    unique_nodes_g1.add(str(item))
            
            for item in comparison_result.unique_to_graph2:
                if '-' in str(item) and len(str(item).split('-')) >= 3:  # Potenziale relazione
                    parts = str(item).split('-')
                    if len(parts) >= 3:
                        start_node = parts[0]
                        rel_type = parts[1]
                        end_node = '-'.join(parts[2:])
                        unique_relations_g2.add((start_node, end_node, rel_type))
                else:
                    # È un nodo
                    unique_nodes_g2.add(str(item))
        
        def create_networkx_graph(graph: Graph, name: str):
            """Converte un Graph in un NetworkX DiGraph per la visualizzazione."""
            G = nx.DiGraph()
            
            # Aggiungi nodi con attributi
            for node in graph.nodes.values():
                node_type = node.get_property('type', 'unknown')
                # Usa l'ID del nodo (variabile Cypher) come etichetta nel PNG
                node_display_name = node.id  # Questo è ora la variabile Cypher (es. "CSP", "WAN", "CoolKit_API")
                
                G.add_node(node.id, 
                          label=f"{node_display_name}\n({node_type})",
                          type=node_type,
                          name=node_display_name)
            
            # Aggiungi archi
            for rel in graph.relationships.values():
                edge_label = rel.relationship_type
                protocol = rel.properties.get('protocol') if hasattr(rel, 'properties') and rel.properties else None
                if protocol:
                    edge_label += f"\n({protocol})"
                
                G.add_edge(rel.start_node_id, rel.end_node_id, 
                          label=edge_label,
                          type=rel.relationship_type)
            
            return G
        
        def elk_like_layout(G, scale=4):
            """Layout gerarchico simile a ELK usando NetworkX."""
            # 1. Prova prima con layout gerarchico multipartite
            try:
                # Assegna subset ai nodi in base al loro livello gerarchico
                levels = {}
                # Trova i nodi radice (senza predecessori)
                roots = [n for n in G.nodes() if G.in_degree(n) == 0]
                if not roots:
                    roots = [list(G.nodes())[0]]  # Se non ci sono radici, prendi il primo nodo
                
                # Assegna livelli usando BFS
                from collections import deque
                queue = deque([(root, 0) for root in roots])
                visited = set()
                
                while queue:
                    node, level = queue.popleft()
                    if node in visited:
                        continue
                    visited.add(node)
                    levels[node] = level
                    
                    for successor in G.successors(node):
                        if successor not in visited:
                            queue.append((successor, level + 1))
                
                # Assegna subset per multipartite_layout
                for node in G.nodes():
                    if node not in levels:
                        levels[node] = 0
                    G.nodes[node]['subset'] = levels[node]
                
                # Usa multipartite layout con orientamento verticale (top-down come ELK)
                return nx.multipartite_layout(G, subset_key='subset', align='vertical', scale=scale)
                
            except Exception:
                # Fallback 1: Kamada-Kawai (buono per grafi piccoli-medi)
                try:
                    return nx.kamada_kawai_layout(G, scale=scale)
                except Exception:
                    # Fallback 2: Spring layout ottimizzato per chiarezza
                    return nx.spring_layout(G, k=5, iterations=200, seed=42, scale=scale)

        def visualize_graph(G: nx.DiGraph, title: str, filename: str, unique_nodes=None, unique_relations=None):
            """Crea una visualizzazione pulita del grafo NetworkX con layout ELK.
            
            Args:
                G: Il grafo NetworkX
                title: Titolo della visualizzazione
                filename: Nome del file di output
                unique_nodes: Set di nodi unici da evidenziare
                unique_relations: Set di relazioni uniche da evidenziare (start, end, type)
            """
            plt.figure(figsize=(16, 12))
            
            pos = elk_like_layout(G, scale=4)
            
            # Carica colori dal file AssetTypes.xlsm
            type_colors = get_asset_type_colors()
            # Aggiungi colore di default per tipi non trovati
            type_colors['unknown'] = '#D3D3D3'
            
            # Disegna gli archi prima (così vanno sotto i nodi)
            for edge in G.edges():
                start_pos = pos[edge[0]]
                end_pos = pos[edge[1]]
                edge_type = G.edges[edge].get('type', '')
                
                # Controlla se questo arco è unico
                is_unique = False
                if unique_relations:
                    for unique_rel in unique_relations:
                        if (unique_rel[0] == edge[0] and unique_rel[1] == edge[1] and unique_rel[2] == edge_type):
                            is_unique = True
                            break
                
                # Stile avanzato per archi unici
                if is_unique:
                    # Disegna un arco ombra per archi unici
                    plt.annotate('', xy=(end_pos[0] + 0.01, end_pos[1] - 0.01), 
                               xytext=(start_pos[0] + 0.01, start_pos[1] - 0.01),
                               arrowprops=dict(arrowstyle='->', lw=4, color='black', alpha=0.3))
                    
                    # Arco principale rosso spesso
                    plt.annotate('', xy=end_pos, xytext=start_pos,
                               arrowprops=dict(arrowstyle='->', lw=4, color='#FF0000', alpha=1.0))
                    
                    # Arco secondario per effetto "doppio"
                    plt.annotate('', xy=end_pos, xytext=start_pos,
                               arrowprops=dict(arrowstyle='->', lw=2, color='#FF6666', alpha=0.8))
                else:
                    # Arco normale
                    plt.annotate('', xy=end_pos, xytext=start_pos,
                               arrowprops=dict(arrowstyle='->', lw=2, color='#666666', alpha=0.7))
                
                # Aggiungi etichetta del tipo di relazione
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                
                if edge_type:  # Solo se c'è un tipo di relazione
                    if is_unique:
                        # Etichetta speciale per relazioni uniche
                        plt.annotate(f"★ {edge_type} ★", xy=(mid_x, mid_y), ha='center', va='center',
                                   fontsize=10, fontweight='bold', color='#FFFFFF',
                                   bbox=dict(boxstyle="round,pad=0.4", facecolor='#FF0000', 
                                            edgecolor='#800000', linewidth=2, alpha=0.95))
                        
                        # Etichetta "UNIQUE RELATION" sotto
                        plt.annotate("UNIQUE", xy=(mid_x, mid_y - 0.08), ha='center', va='center',
                                   fontsize=7, fontweight='bold', color='#FF0000',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='#FFEEEE', 
                                            edgecolor='#FF0000', alpha=0.9))
                    else:
                        # Etichetta normale
                        plt.annotate(edge_type, xy=(mid_x, mid_y), ha='center', va='center',
                                   fontsize=9, fontweight='bold', color='#333333',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                            edgecolor='gray', alpha=0.8))
            
            # Disegna i nodi sopra gli archi
            for node, (x, y) in pos.items():
                node_data = G.nodes[node]
                node_type = node_data.get('type', 'unknown')
                node_name = node_data.get('name', node)
                color = type_colors.get(node_type, '#D3D3D3')
                
                # Controlla se questo nodo è unico
                is_unique_node = unique_nodes and node in unique_nodes
                
                # Stile diverso per nodi unici
                edge_color = '#FF0000' if is_unique_node else 'black'
                edge_width = 3 if is_unique_node else 2
                radius = 0.09 if is_unique_node else 0.08
                
                # Effetti visivi avanzati per nodi unici
                if is_unique_node:
                    # Ombra per nodi unici
                    shadow = plt.Circle((x + 0.015, y - 0.015), radius, facecolor='black', 
                                      alpha=0.4, zorder=1)
                    plt.gca().add_patch(shadow)
                    
                    # Anello esterno pulsante per nodi unici
                    outer_ring = plt.Circle((x, y), radius + 0.03, facecolor='none', 
                                          edgecolor='#FF0000', linewidth=3, alpha=0.8, zorder=2)
                    plt.gca().add_patch(outer_ring)
                    
                    # Secondo anello per effetto "pulsante"
                    pulse_ring = plt.Circle((x, y), radius + 0.05, facecolor='none', 
                                          edgecolor='#FF6666', linewidth=1, alpha=0.5, zorder=2)
                    plt.gca().add_patch(pulse_ring)
                
                # Disegna il nodo principale
                alpha_val = 1.0 if is_unique_node else 0.8
                circle = plt.Circle((x, y), radius, facecolor=color, alpha=alpha_val, 
                                  edgecolor=edge_color, linewidth=edge_width, zorder=3)
                plt.gca().add_patch(circle)
                
                # Testo del nodo con stile diverso per unici
                text_color = '#FFFFFF' if is_unique_node else 'black'
                text_size = 9 if is_unique_node else 8
                
                # Sfondo del testo per nodi unici per migliore leggibilità
                if is_unique_node:
                    plt.annotate(node_name, xy=(x, y), ha='center', va='center',
                               fontsize=text_size, fontweight='bold', color=text_color, zorder=4,
                               bbox=dict(boxstyle="round,pad=0.1", facecolor='#800000', alpha=0.8))
                else:
                    plt.annotate(node_name, xy=(x, y), ha='center', va='center',
                               fontsize=text_size, fontweight='bold', color=text_color, zorder=4)
                
                # Etichetta "UNIQUE" sopra i nodi unici
                if is_unique_node:
                    plt.annotate('★ UNIQUE ★', xy=(x, y + radius + 0.08), ha='center', va='center',
                               fontsize=8, fontweight='bold', color='#FF0000',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFEEEE', 
                                        edgecolor='#FF0000', linewidth=2, alpha=0.9), zorder=5)
            
            # Titolo migliorato con informazioni sulle differenze
            unique_count_info = ""
            if unique_nodes or unique_relations:
                unique_node_count = len(unique_nodes) if unique_nodes else 0
                unique_rel_count = len(unique_relations) if unique_relations else 0
                unique_count_info = f"\n(★ {unique_node_count} Unique Nodes, {unique_rel_count} Unique Relations ★)"
            
            plt.title(title + unique_count_info, fontsize=16, fontweight='bold', pad=40, color='darkblue')
            plt.axis('off')
            plt.axis('equal')  # Mantiene i cerchi rotondi
            
            # Legenda migliorata con descrizioni chiare
            legend_elements = []
            
            # Tipi di nodi
            used_types = set(G.nodes[node].get('type', 'unknown') for node in G.nodes())
            for node_type in sorted(used_types):
                if node_type in type_colors:
                    legend_elements.append(patches.Patch(color=type_colors[node_type], label=f"{node_type} (Node Type)"))
            
            # Separatore e elementi unici
            if unique_nodes or unique_relations:
                legend_elements.append(patches.Patch(color='white', label='──── DIFFERENCES ────'))
                
                # Conta elementi unici
                unique_node_count = len(unique_nodes) if unique_nodes else 0
                unique_rel_count = len(unique_relations) if unique_relations else 0
                
                if unique_node_count > 0:
                    legend_elements.append(patches.Patch(color='#FF0000', 
                                                        label=f'★ Unique Nodes ({unique_node_count})'))
                if unique_rel_count > 0:
                    legend_elements.append(patches.Patch(color='#FF4444', 
                                                        label=f'★ Unique Relations ({unique_rel_count})'))
                
                # Istruzioni visive
                legend_elements.append(patches.Patch(color='lightgray', 
                                                    label='• Red borders = Unique elements'))
                legend_elements.append(patches.Patch(color='lightgray', 
                                                    label='• Thick arrows = Unique relations'))
            
            if legend_elements:
                legend = plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                                   fontsize=9, frameon=True, fancybox=True, shadow=True,
                                   title="GRAPH LEGEND", title_fontsize=11)
                legend.get_title().set_fontweight('bold')
            
            # Salva il grafico
            filepath = os.path.join(output_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return filepath
        
        # Genera le visualizzazioni
        G1 = create_networkx_graph(graph1, "Graph 1")
        G2 = create_networkx_graph(graph2, "Graph 2")
        
        # Determina i nomi dei grafi basandosi sui file path e contenuto
        def get_graph_info(graph, file_path):
            # Determina se è corretto o scorretto dal nome del file
            correctness = ""
            if file_path:
                if 'incorrect' in file_path.lower() or 'wrong' in file_path.lower():
                    correctness = " (INCORRECT)"
                elif 'correct' in file_path.lower() or 'right' in file_path.lower():
                    correctness = " (CORRECT)"
            
            # Prova a dedurre il nome dal contenuto del grafo
            base_name = ""
            for node in graph.nodes.values():
                name = node.get_property('name', '')
                if 'jetracer' in name.lower():
                    base_name = 'JetRacer'
                    break
                elif 'ewelink' in name.lower() or 'sonoff' in name.lower():
                    base_name = 'Ewelink'
                    break
            
            # Se non trova nomi specifici dal contenuto, prova dal file path
            if not base_name and file_path:
                if 'jetracer' in file_path.lower():
                    base_name = 'JetRacer'
                elif 'ewelink' in file_path.lower():
                    base_name = 'Ewelink'
                else:
                    base_name = f'Graph_{len(graph.nodes)}nodes'
            
            return base_name + correctness, correctness
        
        name1, correctness1 = get_graph_info(graph1, file1_path)
        name2, correctness2 = get_graph_info(graph2, file2_path)
        
        # Crea nomi file basati sulla correttezza
        base_name1 = name1.replace(" (INCORRECT)", "").replace(" (CORRECT)", "").lower()
        base_name2 = name2.replace(" (INCORRECT)", "").replace(" (CORRECT)", "").lower()
        
        filename1 = base_name1 + ("_incorrect" if "INCORRECT" in name1 else "_correct")
        filename2 = base_name2 + ("_incorrect" if "INCORRECT" in name2 else "_correct")
        
        # Visualizza i grafi senza evidenziazione (quella sarà solo nella comparison)
        path1 = visualize_graph(G1, f"{name1} - Graph Structure", f"{filename1}_graph.png")
        path2 = visualize_graph(G2, f"{name2} - Graph Structure", f"{filename2}_graph.png")
        
        print(f"Grafo 1 salvato in: {path1}")
        print(f"Grafo 2 salvato in: {path2}")
        
        # Crea anche una visualizzazione combinata per confronto
        plt.figure(figsize=(20, 10))
        
        # Usa la stessa mappatura colori dal file Excel
        type_colors = get_asset_type_colors()
        type_colors['unknown'] = '#D3D3D3'
        
        # Identifica elementi comuni e mancanti usando la stessa logica delle metriche (type + signature)
        def get_node_signature(node_id, graph, networkx_graph):
            """Calcola la signature di un nodo: (type, sorted(multiset of (rel_type, direction)))"""
            node_type = networkx_graph.nodes[node_id].get('type', 'unknown')
            sig = Counter()
            
            # Trova il nodo originale nel grafo
            original_node = None
            for node in graph.nodes.values():
                if node.id == node_id:
                    original_node = node
                    break
            
            if original_node:
                # Usa le relazioni dal grafo originale per la signature
                for rel in graph.relationships.values():
                    if rel.start_node_id == node_id:
                        sig[(rel.relationship_type, 'out')] += 1
                    if rel.end_node_id == node_id:
                        sig[(rel.relationship_type, 'in')] += 1
            
            return (node_type, tuple(sorted(sig.items())))
        
        # Crea signature per tutti i nodi
        signatures_g1 = {node_id: get_node_signature(node_id, graph1, G1) for node_id in G1.nodes()}
        signatures_g2 = {node_id: get_node_signature(node_id, graph2, G2) for node_id in G2.nodes()}
        
        # Trova nodi equivalenti basandosi su signature
        equivalent_nodes_g1_to_g2 = {}  # node_id_g1 -> node_id_g2 se equivalenti
        equivalent_nodes_g2_to_g1 = {}  # node_id_g2 -> node_id_g1 se equivalenti
        
        for node1_id, sig1 in signatures_g1.items():
            for node2_id, sig2 in signatures_g2.items():
                if sig1 == sig2 and node2_id not in equivalent_nodes_g2_to_g1:
                    equivalent_nodes_g1_to_g2[node1_id] = node2_id
                    equivalent_nodes_g2_to_g1[node2_id] = node1_id
                    break
        
        # Nodi comuni e mancanti basati su equivalenza
        common_nodes_g1 = set(equivalent_nodes_g1_to_g2.keys())
        common_nodes_g2 = set(equivalent_nodes_g2_to_g1.keys())
        missing_in_g2 = set(G1.nodes()) - common_nodes_g1
        missing_in_g1 = set(G2.nodes()) - common_nodes_g2
        
        # Relazioni comuni e mancanti (considera equivalenza dei nodi)
        def normalize_edge_with_equivalence(edge, equiv_map):
            """Normalizza un arco usando la mappatura di equivalenza dei nodi"""
            start, end, edge_type = edge
            norm_start = equiv_map.get(start, start)
            norm_end = equiv_map.get(end, end)
            return (norm_start, norm_end, edge_type)
        
        edges_g1_normalized = {normalize_edge_with_equivalence((e[0], e[1], G1.edges[e].get('type', '')), equivalent_nodes_g1_to_g2) 
                              for e in G1.edges()}
        edges_g2_normalized = {normalize_edge_with_equivalence((e[0], e[1], G2.edges[e].get('type', '')), equivalent_nodes_g2_to_g1) 
                              for e in G2.edges()}
        
        common_edges_normalized = edges_g1_normalized & edges_g2_normalized
        
        # Archi mancanti (basati su signature originali)
        edges_g1_orig = {(e[0], e[1], G1.edges[e].get('type', '')) for e in G1.edges()}
        edges_g2_orig = {(e[0], e[1], G2.edges[e].get('type', '')) for e in G2.edges()}
        
        missing_edges_in_g2 = set()
        missing_edges_in_g1 = set()
        
        # Un arco di G1 manca in G2 se la sua versione normalizzata non è in common_edges_normalized
        for edge in edges_g1_orig:
            edge_normalized = normalize_edge_with_equivalence(edge, equivalent_nodes_g1_to_g2)
            if edge_normalized not in common_edges_normalized:
                missing_edges_in_g2.add(edge)
        
        # Un arco di G2 manca in G1 se la sua versione normalizzata non è in common_edges_normalized  
        for edge in edges_g2_orig:
            edge_normalized = normalize_edge_with_equivalence(edge, equivalent_nodes_g2_to_g1)
            if edge_normalized not in common_edges_normalized:
                missing_edges_in_g1.add(edge)
        
        # Subplot per il primo grafo
        plt.subplot(1, 2, 1)
        pos1 = elk_like_layout(G1, scale=2)
        
        # Disegna archi con evidenziazione di quelli mancanti in G2
        for edge in G1.edges():
            start_pos = pos1[edge[0]]
            end_pos = pos1[edge[1]]
            edge_type = G1.edges[edge].get('type', '')
            edge_tuple = (edge[0], edge[1], edge_type)
            
            # Evidenzia gli archi che mancano in G2
            is_missing = edge_tuple in missing_edges_in_g2
            arrow_color = '#FF4444' if is_missing else '#666666'
            arrow_width = 3 if is_missing else 1.5
            arrow_alpha = 1.0 if is_missing else 0.7
            
            plt.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', lw=arrow_width, color=arrow_color, alpha=arrow_alpha))
            
            # Etichetta del tipo di relazione
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            if edge_type:
                label_color = '#FF0000' if is_missing else '#333333'
                bbox_color = '#FFEEEE' if is_missing else 'white'
                edge_color = 'red' if is_missing else 'gray'
                plt.annotate(edge_type, xy=(mid_x, mid_y), ha='center', va='center',
                           fontsize=7, fontweight='bold', color=label_color,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=bbox_color, 
                                    edgecolor=edge_color, alpha=0.8))
        
        # Disegna nodi con evidenziazione per "unknown" e mancanti in G2
        for node in G1.nodes():
            node_type = G1.nodes[node].get('type', 'unknown')
            is_unknown = node_type == 'unknown'
            is_missing = node in missing_in_g2
            
            # Colore di base
            color = type_colors.get(node_type, '#D3D3D3')
            
            # Modifica colore per evidenziare
            if is_unknown:
                color = '#FFAA00'  # Arancione per unknown
                edge_color = '#FF6600'
                edge_width = 3
            elif is_missing:
                # Aggiungi un bordo rosso per i nodi mancanti
                edge_color = '#FF4444'
                edge_width = 4
            else:
                edge_color = 'black'
                edge_width = 1.5
            
            # Disegna il nodo
            nx.draw_networkx_nodes(G1, pos1, nodelist=[node], node_color=[color], 
                                 node_size=600, alpha=0.8, edgecolors=edge_color, linewidths=edge_width)
        
        # Etichette dei nomi dei nodi
        node_labels1 = {n: G1.nodes[n].get('name', n) for n in G1.nodes()}
        nx.draw_networkx_labels(G1, pos1, node_labels1, font_size=6, font_weight='bold')
        
        # Colore del titolo basato sulla correttezza
        title_color1 = 'green' if 'CORRECT' in name1 else 'red' if 'INCORRECT' in name1 else 'black'
        plt.title(f"{name1}", fontsize=14, fontweight='bold', color=title_color1)
        plt.axis('off')
        
        # Subplot per il secondo grafo
        plt.subplot(1, 2, 2)
        pos2 = elk_like_layout(G2, scale=2)
        
        # Disegna archi con evidenziazione di quelli mancanti in G1
        for edge in G2.edges():
            start_pos = pos2[edge[0]]
            end_pos = pos2[edge[1]]
            edge_type = G2.edges[edge].get('type', '')
            edge_tuple = (edge[0], edge[1], edge_type)
            
            # Evidenzia gli archi che mancano in G1
            is_missing = edge_tuple in missing_edges_in_g1
            arrow_color = '#FF4444' if is_missing else '#666666'
            arrow_width = 3 if is_missing else 1.5
            arrow_alpha = 1.0 if is_missing else 0.7
            
            plt.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', lw=arrow_width, color=arrow_color, alpha=arrow_alpha))
            
            # Etichetta del tipo di relazione
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            if edge_type:
                label_color = '#FF0000' if is_missing else '#333333'
                bbox_color = '#FFEEEE' if is_missing else 'white'
                edge_color = 'red' if is_missing else 'gray'
                plt.annotate(edge_type, xy=(mid_x, mid_y), ha='center', va='center',
                           fontsize=7, fontweight='bold', color=label_color,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=bbox_color, 
                                    edgecolor=edge_color, alpha=0.8))
        
        # Disegna nodi con evidenziazione per "unknown" e mancanti in G1
        for node in G2.nodes():
            node_type = G2.nodes[node].get('type', 'unknown')
            is_unknown = node_type == 'unknown'
            is_missing = node in missing_in_g1
            
            # Colore di base
            color = type_colors.get(node_type, '#D3D3D3')
            
            # Modifica colore per evidenziare
            if is_unknown:
                color = '#FFAA00'  # Arancione per unknown
                edge_color = '#FF6600'
                edge_width = 3
            elif is_missing:
                # Aggiungi un bordo rosso per i nodi mancanti
                edge_color = '#FF4444'
                edge_width = 4
            else:
                edge_color = 'black'
                edge_width = 1.5
            
            # Disegna il nodo
            nx.draw_networkx_nodes(G2, pos2, nodelist=[node], node_color=[color], 
                                 node_size=600, alpha=0.8, edgecolors=edge_color, linewidths=edge_width)
        
        # Etichette dei nomi dei nodi
        node_labels2 = {n: G2.nodes[n].get('name', n) for n in G2.nodes()}
        nx.draw_networkx_labels(G2, pos2, node_labels2, font_size=6, font_weight='bold')
        
        # Colore del titolo basato sulla correttezza
        title_color2 = 'green' if 'CORRECT' in name2 else 'red' if 'INCORRECT' in name2 else 'black'
        plt.title(f"{name2}", fontsize=14, fontweight='bold', color=title_color2)
        plt.axis('off')
        
        # Aggiungi legenda per le differenze evidenziate
        legend_elements = [
            patches.Patch(color='#FFAA00', label='Nodi Unknown'),
            patches.Patch(color='#FF4444', label='Elementi Mancanti'),
            patches.Patch(color='#666666', label='Elementi Comuni')
        ]
        plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                     ncol=3, fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle("Graph Comparison - Differenze Evidenziate", fontsize=16, fontweight='bold')
        combined_path = os.path.join(output_dir, "graph_comparison.png")
        plt.tight_layout()
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confronto combinato salvato in: {combined_path}")
        print(f"Tutte le visualizzazioni dei grafi sono state generate con successo in {output_dir}/")
        
    except ImportError as e:
        print(f"Errore: NetworkX non è disponibile per la visualizzazione dei grafi. Installa con: pip install networkx")
        print(f"Dettaglio errore: {e}")
    except Exception as e:
        print(f"Errore durante la generazione delle visualizzazioni dei grafi: {e}")
        import traceback
        traceback.print_exc()

import pandas as pd

def get_asset_type_colors(xlsm_path="AssetTypes.xlsm"):
    """
    Legge la mappatura asset type -> colore dal file Excel.
    Restituisce un dizionario {type: colore_hex}
    """
    try:
        df = pd.read_excel(xlsm_path)
        color_map = {}
        for _, row in df.iterrows():
            asset_name = str(row['Name']).strip()
            color = str(row['Color']).strip()
            if asset_name and color and color != 'nan':
                # Usa sia Primary Label che Name come chiavi per la mappatura
                primary_label = str(row['Primary Label']).strip()
                if primary_label and primary_label != 'nan':
                    color_map[primary_label] = color
                color_map[asset_name] = color
        
        print(f"Caricati {len(color_map)} colori dal file {xlsm_path}")
        return color_map
    except Exception as e:
        print(f"Errore lettura colori da {xlsm_path}: {e}")
        # Fallback con colori predefiniti
        return {
            'Asset': '#FF6B6B', 'Device': '#4ECDC4', 'Software': '#45B7D1',
            'Network': '#96CEB4', 'Protocol': '#FFEAA7', 'Service': '#DDA0DD',
            'HW': '#FF8C42', 'Browser': '#8E44AD', 'OS': '#2ECC71'
        }

# Sostituisci la definizione di type_colors nella visualizzazione con:
type_colors = get_asset_type_colors()
# (Usa type_colors come prima, ma ora viene dal file Excel)