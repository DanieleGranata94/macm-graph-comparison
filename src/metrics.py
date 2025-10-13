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

def plot_similarity_metrics_all(result, output_dir="output"):
    """
    Genera un grafico a barre con i 4 valori di similarity per una coppia di grafi.
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
        result.mcs_ratio_graph1,
        result.supergraph_ratio_graph1,
        result.edit_distance
    ]
    metric_labels = ["Similarity Score", "MCS Ratio", "Supergraph Ratio", "Edit Distance"]
    colors = ["#43a047", "#1976d2", "#ffa000", "#F44336"]

    plt.figure(figsize=(8,6))
    plt.style.use('seaborn-v0_8-darkgrid')
    bars = plt.bar(metric_labels, metrics, color=colors, edgecolor='black', width=0.6)

    plt.ylabel("Metric Value", fontsize=13)
    plt.title("All Similarity Metrics", fontsize=15, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.04, f"{height:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(output_dir, "all_similarity_metrics.png"), dpi=120)
    plt.close()
    print(f"Chart saved as {output_dir}/all_similarity_metrics.png")

def plot_similarity_metrics_normalized(result, output_dir="output"):
    """
    Genera un grafico a barre solo con le metriche normalizzate tra 0 e 1.
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
        result.mcs_ratio_graph1,
        result.supergraph_ratio_graph1
    ]
    metric_labels = ["Similarity Score", "MCS Ratio", "Supergraph Ratio"]
    colors = ["#43a047", "#1976d2", "#ffa000"]

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
    plt.savefig(os.path.join(output_dir, "normalized_similarity_metrics.png"), dpi=120)
    plt.close()
    print(f"Chart saved as {output_dir}/normalized_similarity_metrics.png")

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
    plt.savefig(os.path.join(output_dir, "metrics_normalized_to_identity.png"), dpi=120)
    plt.close()
    print(f"Chart saved as {output_dir}/metrics_normalized_to_identity.png")

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
    Genera un grafico a barre con i 4 valori di similarity per una coppia di grafi.
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
        result.mcs_ratio_graph1,
        result.supergraph_ratio_graph1,
        result.edit_distance
    ]
    metric_labels = ["Similarity Score", "MCS Ratio", "Supergraph Ratio", "Edit Distance"]
    colors = ["#43a047", "#1976d2", "#ffa000", "#F44336"]

    plt.figure(figsize=(8,6))
    plt.style.use('seaborn-v0_8-darkgrid')
    bars = plt.bar(metric_labels, metrics, color=colors, edgecolor='black', width=0.6)

    plt.ylabel("Metric Value", fontsize=13)
    plt.title("All Similarity Metrics", fontsize=15, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.04, f"{height:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(output_dir, "all_similarity_metrics.png"), dpi=120)
    plt.close()
    print(f"Chart saved as {output_dir}/all_similarity_metrics.png")


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
        # For simplicity, we'll use a simplified version based on node and edge differences
        # A full edit distance algorithm would be more complex and computationally expensive
        
        # Node differences based on name, then check properties
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
        
        # Calculate insertions and deletions (nodes with names not in the other graph)
        node_insertions = len(nodes2_names - nodes1_names)
        node_deletions = len(nodes1_names - nodes2_names)
        
        # Calculate substitutions (nodes with same name but different properties)
        node_substitutions = 0
        common_names = nodes1_names & nodes2_names
        for name in common_names:
            # Find nodes with this name in both graphs
            node1 = None
            node2 = None
            
            for node in graph1.nodes.values():
                if node.get_property('name', '') == name:
                    node1 = node
                    break
            
            for node in graph2.nodes.values():
                if node.get_property('name', '') == name:
                    node2 = node
                    break
            
            # If nodes exist but are not equivalent, it's a substitution
            if node1 and node2 and not self._nodes_equivalent(node1, node2):
                node_substitutions += 1
        
        # Relationship differences
        edges1 = set(self._get_edge_signatures(graph1))
        edges2 = set(self._get_edge_signatures(graph2))
        
        edge_insertions = len(edges2 - edges1)
        edge_deletions = len(edges1 - edges2)
        edge_substitutions = 0
        
        # Calculate edge substitutions (edges that exist in both but have different properties)
        common_edges = edges1 & edges2
        for edge_signature in common_edges:
            edge1 = self._find_edge_by_signature(graph1, edge_signature)
            edge2 = self._find_edge_by_signature(graph2, edge_signature)
            if edge1 and edge2 and not self._relationships_equivalent(edge1, edge2):
                edge_substitutions += 1
        
        # Total edit distance
        edit_distance = (node_insertions + node_deletions + node_substitutions + 
                        edge_insertions + edge_deletions + edge_substitutions)
        
        self.logger.debug(f"Edit distance calculation: "
                         f"node_ops={node_insertions + node_deletions + node_substitutions}, "
                         f"edge_ops={edge_insertions + edge_deletions + edge_substitutions}")
        
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
        max_graph_size = max(len(graph1.nodes), len(graph2.nodes))
        normalized_edit_distance = edit_distance / max_graph_size if max_graph_size > 0 else 0
        
        mcs_ratio_graph1 = mcs_size / len(graph1.nodes) if len(graph1.nodes) > 0 else 0
        mcs_ratio_graph2 = mcs_size / len(graph2.nodes) if len(graph2.nodes) > 0 else 0
        
        supergraph_ratio_graph1 = supergraph_size / len(graph1.nodes) if len(graph1.nodes) > 0 else 0
        supergraph_ratio_graph2 = supergraph_size / len(graph2.nodes) if len(graph2.nodes) > 0 else 0
        
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
            node_count_difference=invariants['node_count_difference'],
            relationship_count_difference=invariants['relationship_count_difference'],
            degree_distribution_similarity=invariants['degree_distribution_similarity'],
            label_distribution_similarity=invariants['label_distribution_similarity'],
            common_nodes=common_nodes,
            common_relationships=common_edges,
            unique_to_graph1=unique_to_graph1,
            unique_to_graph2=unique_to_graph2
        )
        
        self.logger.info(f"Graph comparison completed. Similarity score: {result.get_similarity_score():.3f}")
        
        return result
    
    def _nodes_equivalent(self, node1: GraphNode, node2: GraphNode) -> bool:
        """Check if two nodes are equivalent based on name and their properties (type, primary, secondary)."""
        # First check if nodes have the same name
        name1 = node1.get_property('name', '')
        name2 = node2.get_property('name', '')
        
        if name1 != name2 or name1 == '':
            return False
        
        # If names are the same, check if properties are also the same
        key_properties = ['type', 'primary', 'secondary']
        for prop in key_properties:
            if node1.get_property(prop) != node2.get_property(prop):
                return False
        
        return True
    
    def _relationships_equivalent(self, rel1: GraphRelationship, rel2: GraphRelationship) -> bool:
        """Check if two relationships are equivalent."""
        return (rel1.relationship_type == rel2.relationship_type and
                rel1.start_node_id == rel2.start_node_id and
                rel2.end_node_id == rel2.end_node_id)
    
    def _get_edge_signatures(self, graph: Graph) -> List[str]:
        """Get unique signatures for all edges in the graph."""
        signatures = []
        for relationship in graph.relationships.values():
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