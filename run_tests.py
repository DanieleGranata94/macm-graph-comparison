#!/usr/bin/env python3
"""
Script per testare le metriche con input semplici e verificare i risultati attesi
"""

import sys
import os
sys.path.append('src')

from database_manager import Neo4jManager
from config import DatabaseConfig
from utils import load_graph_from_cypher_file
from metrics import GraphMetricsCalculator

def test_graphs(file1, file2, test_name, expected_results=None):
    """
    Testa due grafi e stampa i risultati
    
    Args:
        file1: Path al primo file MACM
        file2: Path al secondo file MACM
        test_name: Nome del test
        expected_results: Dizionario con risultati attesi (opzionale)
    """
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    
    db_config = DatabaseConfig()
    
    with Neo4jManager(db_config) as neo4j_manager:
        # Carica i grafi
        graph1 = load_graph_from_cypher_file(neo4j_manager, file1)
        graph2 = load_graph_from_cypher_file(neo4j_manager, file2)
    
    print(f"\nGraph 1: {len(graph1.nodes)} nodi, {len(graph1.relationships)} relazioni")
    print(f"Graph 2: {len(graph2.nodes)} nodi, {len(graph2.relationships)} relazioni")
    
    # Calcola le metriche
    calculator = GraphMetricsCalculator()
    result = calculator.compare_graphs(graph1, graph2)
    
    print(f"\n{'='*50}")
    print("RISULTATI:")
    print(f"{'='*50}")
    print(f"Edit Distance:              {result.edit_distance:.2f}")
    print(f"Normalized Edit Distance:   {result.normalized_edit_distance:.3f}")
    print(f"MCS Size:                   {result.maximum_common_subgraph_size}")
    print(f"MCS Ratio Graph1:           {result.mcs_ratio_graph1:.3f} ({result.mcs_ratio_graph1*100:.1f}%)")
    print(f"MCS Ratio Graph2:           {result.mcs_ratio_graph2:.3f} ({result.mcs_ratio_graph2*100:.1f}%)")
    print(f"Supergraph Size:            {result.minimum_common_supergraph_size}")
    print(f"Supergraph Ratio Graph1:    {result.supergraph_ratio_graph1:.3f} ({result.supergraph_ratio_graph1*100:.1f}%)")
    print(f"Supergraph Ratio Graph2:    {result.supergraph_ratio_graph2:.3f} ({result.supergraph_ratio_graph2*100:.1f}%)")
    print(f"Common Nodes:               {len(result.common_nodes)}")
    print(f"Common Relationships:       {len(result.common_relationships)}")
    
    # Confronta con risultati attesi se forniti
    if expected_results:
        print(f"\n{'='*50}")
        print("VERIFICA RISULTATI ATTESI:")
        print(f"{'='*50}")
        
        all_passed = True
        for key, expected_value in expected_results.items():
            if key == 'edit_distance':
                actual_value = result.edit_distance
            elif key == 'mcs_size':
                actual_value = result.maximum_common_subgraph_size
            elif key == 'mcs_ratio':
                actual_value = result.mcs_ratio_graph1
            elif key == 'common_nodes':
                actual_value = len(result.common_nodes)
            else:
                continue
            
            passed = abs(actual_value - expected_value) < 0.01
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {key}: expected={expected_value}, actual={actual_value:.2f}")
            
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n✅ Tutti i test sono passati!")
        else:
            print("\n❌ Alcuni test sono falliti!")
    
    return result

def main():
    """Esegue tutti i test"""
    
    print("="*80)
    print("TEST SUITE PER METRICHE DI CONFRONTO GRAFI")
    print("="*80)
    
    base_path = "data/test_macm"
    
    # Test 1: Grafi Identici
    test_graphs(
        f"{base_path}/test_identical_A.macm",
        f"{base_path}/test_identical_B.macm",
        "Test 1: Grafi Identici",
        expected_results={
            'edit_distance': 0.0,
            'mcs_ratio': 1.0,
            'common_nodes': 4
        }
    )
    
    # Test 2: Un nodo di differenza
    test_graphs(
        f"{base_path}/test_one_node_diff_A.macm",
        f"{base_path}/test_one_node_diff_B.macm",
        "Test 2: Un Nodo Aggiuntivo",
        expected_results={
            'mcs_size': 3,
            'common_nodes': 3
        }
    )
    
    # Test 3: Grafi completamente diversi
    test_graphs(
        f"{base_path}/test_completely_different_A.macm",
        f"{base_path}/test_completely_different_B.macm",
        "Test 3: Grafi Completamente Diversi",
        expected_results={
            'mcs_size': 0,
            'common_nodes': 0
        }
    )
    
    # Test 4: Stessi nodi, relazioni diverse
    test_graphs(
        f"{base_path}/test_same_nodes_diff_edges_A.macm",
        f"{base_path}/test_same_nodes_diff_edges_B.macm",
        "Test 4: Stessi Nodi, Relazioni Diverse",
        expected_results={
            'mcs_size': 0,  # Signature diverse
            'common_nodes': 0
        }
    )
    
    # Test 5: Sottoinsieme
    test_graphs(
        f"{base_path}/test_subset_small.macm",
        f"{base_path}/test_subset_large.macm",
        "Test 5: Grafo Piccolo vs Grande (Sottoinsieme)",
        expected_results={
            'mcs_size': 2,  # I 2 nodi comuni
            'common_nodes': 2
        }
    )
    
    # Test 6: Grafi minimali
    test_graphs(
        f"{base_path}/test_minimal_single_node.macm",
        f"{base_path}/test_minimal_two_nodes.macm",
        "Test 6: Grafi Minimali (1 nodo vs 2 nodi)",
        expected_results={
            'edit_distance': 2.0,  # 1 nodo + 1 relazione
            'mcs_size': 1
        }
    )
    
    print(f"\n{'='*80}")
    print("TEST SUITE COMPLETATA")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
