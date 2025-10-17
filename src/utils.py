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
        
        # Prima di eseguire il file, estrai la mappatura delle variabili Cypher
        cypher_var_mapping = _extract_cypher_variable_mapping(file_path)
        logger.info(f"Cypher variable mapping: {cypher_var_mapping}")
        
        # Execute Cypher file
        neo4j_manager.execute_cypher_file(file_path)
        
        # Load nodes
        neo4j_nodes = neo4j_manager.get_all_nodes()
        graph = Graph()
        
        # Crea mappatura: component_id -> variabile_cypher per trovare l'ID corretto
        component_to_cypher_var = {comp_id: var for var, comp_id in cypher_var_mapping.items()}
        
        for neo4j_node in neo4j_nodes:
            component_id = neo4j_node.get('component_id', str(neo4j_node.element_id))
            # Usa la variabile Cypher come ID del nodo (es. "CSP", "WAN", "CoolKit_API")
            cypher_var_id = component_to_cypher_var.get(component_id, component_id)
            
            graph_node = GraphNode(
                id=cypher_var_id,
                labels=set(neo4j_node.labels),
                properties=dict(neo4j_node),
                element_id=neo4j_node.element_id
            )
            graph.add_node(graph_node)
        
        # Crea una mappatura: element_id -> cypher_var usando i nodi caricati
        element_to_cypher_var = {}
        for node in graph.nodes.values():
            element_to_cypher_var[node.element_id] = node.id
        
        # Load relationships usando la mappatura delle variabili Cypher
        neo4j_relationships = neo4j_manager.get_all_relationships()
        
        for neo4j_rel in neo4j_relationships:
            start_node = neo4j_rel.start_node
            end_node = neo4j_rel.end_node
            
            # Usa element_id per trovare le variabili Cypher corrispondenti
            start_cypher_var = element_to_cypher_var.get(start_node.element_id)
            end_cypher_var = element_to_cypher_var.get(end_node.element_id)
            
            if start_cypher_var and end_cypher_var:
                graph_rel = GraphRelationship(
                    id=f"{start_cypher_var}_{end_cypher_var}_{neo4j_rel.type}",
                    start_node_id=start_cypher_var,
                    end_node_id=end_cypher_var,
                    relationship_type=neo4j_rel.type,
                    properties=dict(neo4j_rel),
                    element_id=neo4j_rel.element_id
                )
                graph.add_relationship(graph_rel)
            else:
                logger.warning(f"Impossibile mappare relazione: start_element_id={start_node.element_id}, end_element_id={end_node.element_id}")
        
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
    
    # Edit distance interpretation
    if result.normalized_edit_distance <= 0.2:
        output.append("\nInterpretation: Very similar graphs")
    elif result.normalized_edit_distance <= 0.4:
        output.append("\nInterpretation: Moderately similar graphs")
    elif result.normalized_edit_distance <= 0.6:
        output.append("\nInterpretation: Somewhat similar graphs")
    else:
        output.append("\nInterpretation: Very different graphs")
    
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


def _extract_cypher_variable_mapping(file_path: str) -> Dict[str, str]:
    """
    Estrae la mappatura tra variabili Cypher e component_id dal file.
    
    Args:
        file_path: Path al file Cypher
        
    Returns:
        Dizionario {variabile_cypher: component_id}
    """
    import re
    
    mapping = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern per trovare le definizioni dei nodi nel formato:
        # (variabile:Label1:Label2 {component_id: 'id', ...})
        pattern = r'\((\w+):[^{]*\{[^}]*component_id:\s*[\'"]([^\'"]+)[\'"][^}]*\}'
        
        matches = re.findall(pattern, content)
        for var_name, component_id in matches:
            mapping[var_name] = component_id
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Error extracting Cypher variable mapping: {e}")
    
    return mapping


def validate_cypher_files(*file_paths: str) -> bool:
    """
    Validate that the provided Cypher files exist and are readable.
    
    Args:
        *file_paths: Variable number of file paths to validate
        
    Returns:
        True if all files are valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    for file_path in file_paths:
        if not file_path:
            logger.error("Empty file path provided")
            return False
            
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
            
        if not path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False
            
        if not path.suffix.lower() in ['.cypher', '.cql', '.macm']:
            logger.warning(f"File does not have a recognized Cypher extension: {file_path}")
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.error(f"File is empty: {file_path}")
                    return False
        except Exception as e:
            logger.error(f"Cannot read file {file_path}: {e}")
            return False
    
    return True
