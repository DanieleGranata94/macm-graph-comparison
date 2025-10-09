"""
Utility functions for the Graph Comparison Tool.

This module provides logging setup, data conversion utilities,
and other helper functions.
"""

import logging
import sys
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from config import AppConfig
from models import Graph, GraphNode, GraphRelationship, GraphComparisonResult
from database_manager import Neo4jManager


def setup_logging(config: AppConfig) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: Application configuration
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def load_graph_from_cypher_file(neo4j_manager: Neo4jManager, file_path: str) -> Graph:
    """
    Load a graph from a Cypher file.
    
    Args:
        neo4j_manager: Neo4j database manager
        file_path: Path to the Cypher file
        
    Returns:
        Graph object
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Clear database first
        neo4j_manager.clear_database()
        
        # Execute Cypher file
        neo4j_manager.execute_cypher_file(file_path)
        
        # Load nodes
        neo4j_nodes = neo4j_manager.get_all_nodes()
        graph = Graph()
        
        for neo4j_node in neo4j_nodes:
            graph_node = GraphNode(
                id=neo4j_node.get('component_id', str(neo4j_node.element_id)),
                labels=set(neo4j_node.labels),
                properties=dict(neo4j_node),
                element_id=neo4j_node.element_id
            )
            graph.add_node(graph_node)
        
        # Load relationships
        neo4j_relationships = neo4j_manager.get_all_relationships()
        
        for neo4j_rel in neo4j_relationships:
            start_node = neo4j_rel.start_node
            end_node = neo4j_rel.end_node
            
            graph_rel = GraphRelationship(
                id=f"{start_node.element_id}_{end_node.element_id}_{neo4j_rel.type}",
                start_node_id=start_node.get('component_id', str(start_node.element_id)),
                end_node_id=end_node.get('component_id', str(end_node.element_id)),
                relationship_type=neo4j_rel.type,
                properties=dict(neo4j_rel),
                element_id=neo4j_rel.element_id
            )
            graph.add_relationship(graph_rel)
        
        logger.info(f"Loaded graph from {file_path}: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")
        return graph
        
    except Exception as e:
        logger.error(f"Failed to load graph from {file_path}: {e}")
        raise


def format_comparison_results(result: GraphComparisonResult, output_format: str = "text") -> str:
    """
    Format comparison results for output.
    
    Args:
        result: Graph comparison result
        output_format: Output format (text, json, csv)
        
    Returns:
        Formatted string
    """
    if output_format == "json":
        return _format_as_json(result)
    elif output_format == "csv":
        return _format_as_csv(result)
    else:
        return _format_as_text(result)


def _format_as_text(result: GraphComparisonResult) -> str:
    """Format results as human-readable text."""
    output = []
    output.append("=" * 60)
    output.append("GRAPH COMPARISON RESULTS")
    output.append("=" * 60)
    
    # Basic statistics
    output.append(f"\nGraph 1 Statistics:")
    output.append(f"  Nodes: {result.graph1_stats}")
    
    output.append(f"\nGraph 2 Statistics:")
    output.append(f"  Nodes: {result.graph2_stats}")
    
    # Edit distance
    output.append(f"\nEdit Distance Metrics:")
    output.append(f"  Edit Distance: {result.edit_distance:.2f}")
    output.append(f"  Normalized Edit Distance: {result.normalized_edit_distance:.3f}")
    
    # Common subgraph
    output.append(f"\nMaximum Common Subgraph:")
    output.append(f"  Size: {result.maximum_common_subgraph_size}")
    output.append(f"  Ratio (Graph 1): {result.mcs_ratio_graph1:.3f}")
    output.append(f"  Ratio (Graph 2): {result.mcs_ratio_graph2:.3f}")
    
    # Supergraph
    output.append(f"\nMinimum Common Supergraph:")
    output.append(f"  Size: {result.minimum_common_supergraph_size}")
    output.append(f"  Ratio (Graph 1): {result.supergraph_ratio_graph1:.3f}")
    output.append(f"  Ratio (Graph 2): {result.supergraph_ratio_graph2:.3f}")
    
    # Isomorphism invariants
    output.append(f"\nIsomorphism Invariants:")
    output.append(f"  Node Count Difference: {result.node_count_difference}")
    output.append(f"  Relationship Count Difference: {result.relationship_count_difference}")
    output.append(f"  Degree Distribution Similarity: {result.degree_distribution_similarity:.3f}")
    output.append(f"  Label Distribution Similarity: {result.label_distribution_similarity:.3f}")
    
    # Similarity score
    similarity_score = result.get_similarity_score()
    output.append(f"\nOverall Similarity Score: {similarity_score:.3f}")
    
    if similarity_score >= 0.8:
        output.append("  Interpretation: Very similar graphs")
    elif similarity_score >= 0.6:
        output.append("  Interpretation: Moderately similar graphs")
    elif similarity_score >= 0.4:
        output.append("  Interpretation: Somewhat similar graphs")
    else:
        output.append("  Interpretation: Very different graphs")
    
    # Detailed differences
    output.append(f"\nDetailed Differences:")
    output.append(f"  Common Nodes: {len(result.common_nodes)}")
    output.append(f"  Common Relationships: {len(result.common_relationships)}")
    output.append(f"  Unique to Graph 1: {len(result.unique_to_graph1)}")
    output.append(f"  Unique to Graph 2: {len(result.unique_to_graph2)}")
    
    output.append("=" * 60)
    
    return "\n".join(output)


def _format_as_json(result: GraphComparisonResult) -> str:
    """Format results as JSON."""
    data = {
        "similarity_score": result.get_similarity_score(),
        "edit_distance": {
            "value": result.edit_distance,
            "normalized": result.normalized_edit_distance
        },
        "maximum_common_subgraph": {
            "size": result.maximum_common_subgraph_size,
            "ratio_graph1": result.mcs_ratio_graph1,
            "ratio_graph2": result.mcs_ratio_graph2
        },
        "minimum_common_supergraph": {
            "size": result.minimum_common_supergraph_size,
            "ratio_graph1": result.supergraph_ratio_graph1,
            "ratio_graph2": result.supergraph_ratio_graph2
        },
        "isomorphism_invariants": {
            "node_count_difference": result.node_count_difference,
            "relationship_count_difference": result.relationship_count_difference,
            "degree_distribution_similarity": result.degree_distribution_similarity,
            "label_distribution_similarity": result.label_distribution_similarity
        },
        "graph_statistics": {
            "graph1": result.graph1_stats,
            "graph2": result.graph2_stats
        },
        "differences": {
            "common_nodes_count": len(result.common_nodes),
            "common_relationships_count": len(result.common_relationships),
            "unique_to_graph1_count": len(result.unique_to_graph1),
            "unique_to_graph2_count": len(result.unique_to_graph2)
        }
    }
    
    return json.dumps(data, indent=2, default=str)


def _format_as_csv(result: GraphComparisonResult) -> str:
    """Format results as CSV."""
    lines = []
    lines.append("Metric,Value")
    lines.append(f"Similarity Score,{result.get_similarity_score():.3f}")
    lines.append(f"Edit Distance,{result.edit_distance:.2f}")
    lines.append(f"Normalized Edit Distance,{result.normalized_edit_distance:.3f}")
    lines.append(f"MCS Size,{result.maximum_common_subgraph_size}")
    lines.append(f"MCS Ratio Graph1,{result.mcs_ratio_graph1:.3f}")
    lines.append(f"MCS Ratio Graph2,{result.mcs_ratio_graph2:.3f}")
    lines.append(f"Supergraph Size,{result.minimum_common_supergraph_size}")
    lines.append(f"Supergraph Ratio Graph1,{result.supergraph_ratio_graph1:.3f}")
    lines.append(f"Supergraph Ratio Graph2,{result.supergraph_ratio_graph2:.3f}")
    lines.append(f"Node Count Difference,{result.node_count_difference}")
    lines.append(f"Relationship Count Difference,{result.relationship_count_difference}")
    lines.append(f"Degree Distribution Similarity,{result.degree_distribution_similarity:.3f}")
    lines.append(f"Label Distribution Similarity,{result.label_distribution_similarity:.3f}")
    
    return "\n".join(lines)


def validate_cypher_files(file1_path: str, file2_path: str) -> bool:
    """
    Validate that Cypher files exist and are readable.
    
    Args:
        file1_path: Path to first Cypher file
        file2_path: Path to second Cypher file
        
    Returns:
        True if both files are valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    files = [file1_path, file2_path]
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Cypher file not found: {file_path}")
            return False
        
        if not path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False
        
        try:
            with open(path, 'r') as f:
                content = f.read().strip()
                if not content:
                    logger.error(f"Cypher file is empty: {file_path}")
                    return False
        except Exception as e:
            logger.error(f"Error reading Cypher file {file_path}: {e}")
            return False
    
    return True
