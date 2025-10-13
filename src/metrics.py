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
        result_correct.get_similarity_score(),
        result_correct.mcs_ratio_graph1
    ]
    metric_labels = ["Similarity Score", "MCS Ratio"]
    colors = ["#43a047", "#1976d2"]


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
        result.get_similarity_score(),
        result.mcs_ratio_graph1,
        result.supergraph_ratio_graph1,
        result.edit_distance
    ]
    labels = ["Similarity Score", "MCS Ratio", "Supergraph Ratio", "Edit Distance"]
    colors = ["#43a047", "#1976d2", "#ffa000", "#F44336"]

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


def plot_metrics_in_0_1_range(result, output_dir="output"):
    """Genera un grafico con le sole metriche il cui valore raw è già nel range [0,1]."""
    import matplotlib.pyplot as plt
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Only plot the two metrics that are guaranteed to be in [0,1]
    similarity = float(result.get_similarity_score())
    mcs_ratio = float(getattr(result, 'mcs_ratio_graph1', 0.0))

    labels = ["Similarity Score", "MCS Ratio"]
    metrics = [similarity, mcs_ratio]
    colors = ["#43a047", "#1976d2"]

    plt.figure(figsize=(7,5))
    plt.style.use('seaborn-v0_8-darkgrid')
    bars = plt.bar(labels, metrics, color=colors, edgecolor='black', width=0.6)
    plt.ylim(0, 1.05)
    plt.ylabel("Metric Value (0-1)", fontsize=13)
    plt.title("Metrics in Range [0,1]", fontsize=14, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout(pad=3)
    path = os.path.join(output_dir, "metrics_in_0_1_range.png")
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

    sim = float(result.get_similarity_score())
    mcs = float(getattr(result, 'mcs_ratio_graph1', 0.0))
    super_r = float(getattr(result, 'supergraph_ratio_graph1', 0.0))
    edit = float(getattr(result, 'edit_distance', 0.0))

    # Normalize / clip
    mcs_n = max(0.0, min(1.0, mcs))
    super_n = max(0.0, min(1.0, super_r))
    edit_n = max(0.0, min(1.0, 1.0 - (edit / max_edit_distance)))

    metrics = [sim, mcs_n, super_n, edit_n]
    labels = ["Similarity Score", "MCS Ratio", "Supergraph Ratio", "Edit Distance Similarity"]
    colors = ["#43a047", "#1976d2", "#ffa000", "#F44336"]

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

def plot_similarity_metrics_all(result, output_dir="output"):
    """
    Genera un grafico a barre con i valori di similarity per una coppia di grafi, normalizzati, senza Supergraph Ratio.
    Args:
        result: GraphComparisonResult
        output_dir: directory dove salvare il grafico
    """
    import matplotlib.pyplot as plt
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calcolo dei valori normalizzati
    similarity_score = result.get_similarity_score()
    mcs_ratio = result.mcs_ratio_graph1
    edit_distance = result.edit_distance

    max_edit_distance = 100  # Supponendo un valore massimo teorico per Edit Distance

    normalized_metrics = [
        similarity_score,  # Supponendo che sia già normalizzato tra 0 e 1
        mcs_ratio,         # Supponendo che sia già normalizzato tra 0 e 1
        1 - (edit_distance / max_edit_distance)  # Normalizzazione inversa per Edit Distance
    ]

    metric_labels = ["Similarity Score", "MCS Ratio", "Edit Distance"]
    colors = ["#43a047", "#1976d2", "#F44336"]

    plt.figure(figsize=(8,6))
    plt.style.use('seaborn-v0_8-darkgrid')
    bars = plt.bar(metric_labels, normalized_metrics, color=colors, edgecolor='black', width=0.6)

    plt.ylabel("Normalized Metric Value", fontsize=13)
    plt.title("All Similarity Metrics (Normalized)", fontsize=15, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.04, f"{height:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(output_dir, "overall_similarity_normalized.png"), dpi=120)
    plt.close()
    print(f"Chart saved as {output_dir}/overall_similarity_normalized.png")

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
        result.get_similarity_score(),
        result.mcs_ratio_graph1
    ]
    metric_labels = ["Similarity Score", "MCS Ratio"]
    colors = ["#43a047", "#1976d2"]

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
        result.get_similarity_score(),  # già normalizzato tra 0 e 1
        result.mcs_ratio_graph1,        # già normalizzato tra 0 e 1
        result.supergraph_ratio_graph1, # già normalizzato tra 0 e 1
        normalized_edit                 # normalizzato per tendere a 1 quando edit_distance = 0
    ]
    
    metric_labels = ["Similarity Score", "MCS Ratio", "Supergraph Ratio", "Normalized Edit Distance"]
    colors = ["#43a047", "#1976d2", "#ffa000", "#F44336"]

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


def plot_similarity_metrics_all(result, output_dir="output"):
    """
    Genera un grafico a barre con i valori di similarity per una coppia di grafi, normalizzati, senza Supergraph Ratio.
    Args:
        result: GraphComparisonResult
        output_dir: directory dove salvare il grafico
    """
    import matplotlib.pyplot as plt
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calcolo dei valori normalizzati
    similarity_score = result.get_similarity_score()
    mcs_ratio = result.mcs_ratio_graph1
    edit_distance = result.edit_distance

    max_edit_distance = 100  # Supponendo un valore massimo teorico per Edit Distance

    normalized_metrics = [
        similarity_score,  # Supponendo che sia già normalizzato tra 0 e 1
        mcs_ratio,         # Supponendo che sia già normalizzato tra 0 e 1
        1 - (edit_distance / max_edit_distance)  # Normalizzazione inversa per Edit Distance
    ]

    metric_labels = ["Similarity Score", "MCS Ratio", "Edit Distance"]
    colors = ["#43a047", "#1976d2", "#F44336"]

    plt.figure(figsize=(8,6))
    plt.style.use('seaborn-v0_8-darkgrid')
    bars = plt.bar(metric_labels, normalized_metrics, color=colors, edgecolor='black', width=0.6)

    plt.ylabel("Normalized Metric Value", fontsize=13)
    plt.title("All Similarity Metrics (Normalized)", fontsize=15, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.04, f"{height:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(output_dir, "all_similarity_metrics_normalized.png"), dpi=120)
    plt.close()
    print(f"Chart saved as {output_dir}/all_similarity_metrics_normalized.png")


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
        
        mcs_ratio_graph1 = mcs_size / len(graph1.nodes) if len(graph1.nodes) > 0 else 0
        mcs_ratio_graph2 = mcs_size / len(graph2.nodes) if len(graph2.nodes) > 0 else 0

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
        
        self.logger.info(f"Graph comparison completed. Similarity score: {result.get_similarity_score():.3f}")
        
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