#!/usr/bin/env python3
"""
Script per caricare i risultati del confronto grafi su Neo4j.
Crea un nuovo database chiamato 'macm' e carica i risultati.
"""

import json
import logging
from typing import Dict, Any

from database_manager import Neo4jManager
from config import DatabaseConfig, get_config
from utils import setup_logging, load_graph_from_cypher_file, GraphComparisonResult
from metrics import GraphMetricsCalculator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_comparison_database():
    """Create the test_comparison database in Neo4j."""
    db_config, app_config = get_config()
    
    with Neo4jManager(db_config) as neo4j_manager:
        # Create database (this might require admin privileges)
        try:
            with neo4j_manager.get_session() as session:
                # Create the test_comparison database
                session.run("CREATE DATABASE test_comparison IF NOT EXISTS")
                logger.info("Created test_comparison database")
        except Exception as e:
            logger.warning(f"Could not create database (might need admin privileges): {e}")
            logger.info("Will use default database instead")

def load_graphs_only():
    """Load only the two graphs into Neo4j without comparison results."""
    db_config, app_config = get_config()
    
    with Neo4jManager(db_config) as neo4j_manager:
        # Clear existing data
        neo4j_manager.clear_database()
        
        # Load first graph (wrong architecture)
        logger.info("Loading wrong_architecture.macm...")
        graph1 = load_graph_from_cypher_file(neo4j_manager, "wrong_architecture.macm")
        
        # Load second graph (correct architecture)
        logger.info("Loading correct_architecture.macm...")
        graph2 = load_graph_from_cypher_file(neo4j_manager, "correct_architecture.macm")
        
        logger.info("Successfully loaded both graphs to Neo4j!")
        
        # Print summary
        print("\n" + "="*60)
        print("TEST COMPARISON DATABASE - GRAPHS LOADED")
        print("="*60)
        print(f"Graph 1 (wrong_architecture): {len(graph1.nodes)} nodes, {len(graph1.relationships)} relationships")
        print(f"Graph 2 (correct_architecture): {len(graph2.nodes)} nodes, {len(graph2.relationships)} relationships")
        print(f"Total nodes in database: {len(graph1.nodes) + len(graph2.nodes)}")
        print(f"Total relationships in database: {len(graph1.relationships) + len(graph2.relationships)}")
        print("\nDatabase contains:")
        print("- All nodes and relationships from wrong_architecture.macm")
        print("- All nodes and relationships from correct_architecture.macm")
        print("- No comparison metadata (graphs only)")
        print("="*60)

def main():
    """Main function to load graphs to Neo4j."""
    try:
        logger.info("Starting test_comparison database loading to Neo4j...")
        
        # Create database
        create_test_comparison_database()
        
        # Load graphs only
        load_graphs_only()
        
        logger.info("Graphs successfully loaded to test_comparison database!")
        
    except Exception as e:
        logger.error(f"Error loading data to Neo4j: {e}")
        raise

if __name__ == "__main__":
    main()
