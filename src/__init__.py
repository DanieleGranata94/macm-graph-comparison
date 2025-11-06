"""
MACM Graph Metrics Module

A clean, reusable module for calculating similarity metrics between MACM graphs.

Main exports:
- calculate_edit_distance_from_cypher: Calculate edit distance from Cypher files
- calculate_maximum_common_subgraph_from_cypher: Calculate MCS from Cypher files  
- calculate_all_metrics_from_cypher: Calculate all metrics from Cypher files
- GraphMetricsCalculator: Main calculator class
- GraphMetrics: Data class for metrics results
"""

from .graph_metrics import (
    calculate_edit_distance_from_cypher,
    calculate_maximum_common_subgraph_from_cypher,
    calculate_all_metrics_from_cypher,
    GraphMetricsCalculator,
    GraphMetrics
)

from .models import Graph, GraphNode, GraphRelationship
from .database_manager import DatabaseManager
from .utils import load_graph_from_cypher

__version__ = "1.0.0"
__author__ = "Daniele Granata"

__all__ = [
    # Main functions
    "calculate_edit_distance_from_cypher",
    "calculate_maximum_common_subgraph_from_cypher", 
    "calculate_all_metrics_from_cypher",
    
    # Classes
    "GraphMetricsCalculator",
    "GraphMetrics",
    "Graph",
    "GraphNode", 
    "GraphRelationship",
    "DatabaseManager",
    
    # Utilities
    "load_graph_from_cypher",
]
