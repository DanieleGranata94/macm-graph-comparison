#!/usr/bin/env python3
"""
Script per interrogare i risultati del progetto MACM su Neo4j.
Fornisce query predefinite per analizzare i dati caricati.
"""

import logging
from database_manager import Neo4jManager
from config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_graph_statistics():
    """Query basic graph statistics."""
    db_config, app_config = get_config()
    
    with Neo4jManager(db_config) as neo4j_manager:
        with neo4j_manager.get_session() as session:
            print("\n" + "="*60)
            print("MACM PROJECT - GRAPH STATISTICS")
            print("="*60)
            
            # Get graph metadata
            result = session.run("""
                MATCH (g:Graph)
                RETURN g.name as name, g.description as description, 
                       g.node_count as node_count, g.relationship_count as rel_count,
                       g.type as type
                ORDER BY g.name
            """)
            
            for record in result:
                print(f"\nGraph: {record['name']}")
                print(f"  Description: {record['description']}")
                print(f"  Type: {record['type']}")
                print(f"  Nodes: {record['node_count']}")
                print(f"  Relationships: {record['rel_count']}")

def query_comparison_results():
    """Query comparison results."""
    db_config, app_config = get_config()
    
    with Neo4jManager(db_config) as neo4j_manager:
        with neo4j_manager.get_session() as session:
            print("\n" + "="*60)
            print("MACM PROJECT - COMPARISON RESULTS")
            print("="*60)
            
            # Get comparison results
            result = session.run("""
                MATCH (cr:ComparisonResult)
                RETURN cr.similarity_score as similarity,
                       cr.edit_distance as edit_distance,
                       cr.normalized_edit_distance as normalized_edit_distance,
                       cr.mcs_size as mcs_size,
                       cr.common_nodes_count as common_nodes,
                       cr.unique_to_graph1_count as unique_graph1,
                       cr.unique_to_graph2_count as unique_graph2,
                       cr.interpretation as interpretation
            """)
            
            for record in result:
                print(f"\nSimilarity Score: {record['similarity']:.3f}")
                print(f"Interpretation: {record['interpretation']}")
                print(f"\nEdit Distance Metrics:")
                print(f"  Edit Distance: {record['edit_distance']}")
                print(f"  Normalized Edit Distance: {record['normalized_edit_distance']:.3f}")
                print(f"\nCommon Subgraph:")
                print(f"  MCS Size: {record['mcs_size']}")
                print(f"\nDifferences:")
                print(f"  Common Nodes: {record['common_nodes']}")
                print(f"  Unique to Graph 1: {record['unique_graph1']}")
                print(f"  Unique to Graph 2: {record['unique_graph2']}")

def query_differences():
    """Query detailed differences."""
    db_config, app_config = get_config()
    
    with Neo4jManager(db_config) as neo4j_manager:
        with neo4j_manager.get_session() as session:
            print("\n" + "="*60)
            print("MACM PROJECT - DETAILED DIFFERENCES")
            print("="*60)
            
            # Get unique items for graph 1
            print("\nUnique to Graph 1 (cypher_sbagliato):")
            result = session.run("""
                MATCH (d:Difference {type: 'unique_to_graph1'})
                RETURN d.item as item, d.description as description
                ORDER BY d.item
            """)
            
            for record in result:
                print(f"  - {record['item']}")
            
            # Get unique items for graph 2
            print("\nUnique to Graph 2 (architecture_diagram):")
            result = session.run("""
                MATCH (d:Difference {type: 'unique_to_graph2'})
                RETURN d.item as item, d.description as description
                ORDER BY d.item
            """)
            
            for record in result:
                print(f"  - {record['item']}")

def query_node_types():
    """Query node type distribution."""
    db_config, app_config = get_config()
    
    with Neo4jManager(db_config) as neo4j_manager:
        with neo4j_manager.get_session() as session:
            print("\n" + "="*60)
            print("MACM PROJECT - NODE TYPE DISTRIBUTION")
            print("="*60)
            
            # Get node types from cypher_sbagliato
            print("\nNode Types in cypher_sbagliato:")
            result = session.run("""
                MATCH (n)
                WHERE NOT n:Graph AND NOT n:ComparisonResult AND NOT n:Difference
                WITH labels(n) as node_labels
                UNWIND node_labels as label
                WITH label, count(*) as count
                ORDER BY count DESC
                RETURN label, count
                LIMIT 10
            """)
            
            for record in result:
                print(f"  {record['label']}: {record['count']}")

def query_sample_nodes():
    """Query sample nodes from both graphs."""
    db_config, app_config = get_config()
    
    with Neo4jManager(db_config) as neo4j_manager:
        with neo4j_manager.get_session() as session:
            print("\n" + "="*60)
            print("MACM PROJECT - SAMPLE NODES")
            print("="*60)
            
            # Get sample nodes
            result = session.run("""
                MATCH (n)
                WHERE NOT n:Graph AND NOT n:ComparisonResult AND NOT n:Difference
                RETURN n.name as name, n.type as type, n.primary as primary, n.secondary as secondary
                ORDER BY n.name
                LIMIT 15
            """)
            
            print("\nSample Nodes (showing type, primary, secondary):")
            for record in result:
                name = record['name'] or 'Unnamed'
                node_type = record['type'] or 'No type'
                primary = record['primary'] or 'No primary'
                secondary = record['secondary'] or 'No secondary'
                print(f"  {name}: {node_type} | {primary} | {secondary}")

def main():
    """Main function to run all queries."""
    try:
        logger.info("Querying MACM project data from Neo4j...")
        
        query_graph_statistics()
        query_comparison_results()
        query_differences()
        query_node_types()
        query_sample_nodes()
        
        print("\n" + "="*60)
        print("MACM PROJECT QUERIES COMPLETED")
        print("="*60)
        print("\nTo explore more, connect to Neo4j Browser and run:")
        print("MATCH (n) RETURN n LIMIT 25")
        print("\nOr use these sample queries:")
        print("- MATCH (g:Graph) RETURN g")
        print("- MATCH (cr:ComparisonResult) RETURN cr")
        print("- MATCH (d:Difference) RETURN d")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error querying data from Neo4j: {e}")
        raise

if __name__ == "__main__":
    main()
