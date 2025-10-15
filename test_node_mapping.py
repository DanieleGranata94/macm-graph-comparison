#!/usr/bin/env python3
"""
Test per analizzare le differenze tra i due file MACM Ewelink
e spiegare perché certi nodi non sono mappati.
"""

import sys
import os
sys.path.append('src')

from collections import Counter
from utils import load_graph_from_cypher_file
from models import Graph, GraphNode
from database_manager import Neo4jManager
from config import DatabaseConfig

def get_node_signature(node: GraphNode, graph: Graph):
    """Calcola la signature di un nodo: (type, sorted(multiset of (rel_type, direction)))"""
    node_type = node.get_property('type', 'unknown')
    sig = Counter()
    
    # Conta le relazioni in uscita e in entrata
    for rel in graph.relationships.values():
        if rel.start_node_id == node.id:
            sig[(rel.relationship_type, 'out')] += 1
        if rel.end_node_id == node.id:
            sig[(rel.relationship_type, 'in')] += 1
    
    return (node_type, tuple(sorted(sig.items())))

def analyze_node_differences(file1, file2):
    """Analizza le differenze tra due file MACM"""
    
    print("="*80)
    print("ANALISI DETTAGLIATA DELLE DIFFERENZE TRA GRAFI MACM")
    print("="*80)
    
    # Carica i grafi
    db_config = DatabaseConfig()
    
    print(f"Caricando {file1}...")
    with Neo4jManager(db_config) as neo4j_manager:
        graph1 = load_graph_from_cypher_file(neo4j_manager, file1)
        print(f"Caricando {file2}...")
        graph2 = load_graph_from_cypher_file(neo4j_manager, file2)
    
    print(f"\nGrafo 1: {len(graph1.nodes)} nodi, {len(graph1.relationships)} relazioni")
    print(f"Grafo 2: {len(graph2.nodes)} nodi, {len(graph2.relationships)} relazioni")
    
    # Calcola le signature per tutti i nodi
    signatures_g1 = {}
    signatures_g2 = {}
    
    print(f"\n{'='*50}")
    print("SIGNATURE DEI NODI - GRAFO 1")
    print(f"{'='*50}")
    for node in graph1.nodes.values():
        sig = get_node_signature(node, graph1)
        signatures_g1[node.id] = sig
        name = node.get_property('name', node.id)
        print(f"{node.id:20} | {name:25} | Type: {sig[0]:20} | Relations: {sig[1]}")
    
    print(f"\n{'='*50}")
    print("SIGNATURE DEI NODI - GRAFO 2")
    print(f"{'='*50}")
    for node in graph2.nodes.values():
        sig = get_node_signature(node, graph2)
        signatures_g2[node.id] = sig
        name = node.get_property('name', node.id)
        print(f"{node.id:20} | {name:25} | Type: {sig[0]:20} | Relations: {sig[1]}")
    
    # Trova equivalenze
    print(f"\n{'='*50}")
    print("ANALISI EQUIVALENZE")
    print(f"{'='*50}")
    
    equivalent_nodes_g1_to_g2 = {}
    equivalent_nodes_g2_to_g1 = {}
    
    for node1_id, sig1 in signatures_g1.items():
        for node2_id, sig2 in signatures_g2.items():
            if sig1 == sig2 and node2_id not in equivalent_nodes_g2_to_g1:
                equivalent_nodes_g1_to_g2[node1_id] = node2_id
                equivalent_nodes_g2_to_g1[node2_id] = node1_id
                name1 = graph1.nodes[node1_id].get_property('name', node1_id)
                name2 = graph2.nodes[node2_id].get_property('name', node2_id)
                print(f"✓ EQUIVALENTI: {node1_id:15} ({name1:20}) ↔ {node2_id:15} ({name2:20})")
                print(f"    Signature: Type={sig1[0]}, Relations={sig1[1]}")
                break
    
    print(f"\nTotale equivalenze trovate: {len(equivalent_nodes_g1_to_g2)}")
    
    # Analizza i nodi non mappati
    unmapped_g1 = set(signatures_g1.keys()) - set(equivalent_nodes_g1_to_g2.keys())
    unmapped_g2 = set(signatures_g2.keys()) - set(equivalent_nodes_g2_to_g1.keys())
    
    print(f"\n{'='*50}")
    print("NODI NON MAPPATI - GRAFO 1")
    print(f"{'='*50}")
    
    for node_id in unmapped_g1:
        node = graph1.nodes[node_id]
        sig = signatures_g1[node_id]
        name = node.get_property('name', node_id)
        
        print(f"\n✗ {node_id} ({name})")
        print(f"  Type: {sig[0]}")
        print(f"  Relations: {sig[1]}")
        
        # Cerca nodi con stesso type in G2
        same_type_in_g2 = []
        for node2_id, sig2 in signatures_g2.items():
            if sig2[0] == sig[0]:  # Stesso type
                name2 = graph2.nodes[node2_id].get_property('name', node2_id)
                same_type_in_g2.append((node2_id, name2, sig2[1]))
        
        if same_type_in_g2:
            print(f"  Nodi con stesso type in G2:")
            for node2_id, name2, relations in same_type_in_g2:
                print(f"    - {node2_id} ({name2}): Relations={relations}")
                # Analizza differenza nelle relazioni
                if relations != sig[1]:
                    diff_out = set([r for r, d in sig[1] if d == 'out']) - set([r for r, d in relations if d == 'out'])
                    diff_in = set([r for r, d in sig[1] if d == 'in']) - set([r for r, d in relations if d == 'in'])
                    if diff_out:
                        print(f"      Relazioni OUT diverse: {diff_out}")
                    if diff_in:
                        print(f"      Relazioni IN diverse: {diff_in}")
        else:
            print(f"  ⚠ Nessun nodo con type '{sig[0]}' trovato in G2!")
    
    print(f"\n{'='*50}")
    print("NODI NON MAPPATI - GRAFO 2")
    print(f"{'='*50}")
    
    for node_id in unmapped_g2:
        node = graph2.nodes[node_id]
        sig = signatures_g2[node_id]
        name = node.get_property('name', node_id)
        
        print(f"\n✗ {node_id} ({name})")
        print(f"  Type: {sig[0]}")
        print(f"  Relations: {sig[1]}")
        
        # Cerca nodi con stesso type in G1
        same_type_in_g1 = []
        for node1_id, sig1 in signatures_g1.items():
            if sig1[0] == sig[0]:  # Stesso type
                name1 = graph1.nodes[node1_id].get_property('name', node1_id)
                same_type_in_g1.append((node1_id, name1, sig1[1]))
        
        if same_type_in_g1:
            print(f"  Nodi con stesso type in G1:")
            for node1_id, name1, relations in same_type_in_g1:
                print(f"    - {node1_id} ({name1}): Relations={relations}")
                # Analizza differenza nelle relazioni
                if relations != sig[1]:
                    diff_out = set([r for r, d in sig[1] if d == 'out']) - set([r for r, d in relations if d == 'out'])
                    diff_in = set([r for r, d in sig[1] if d == 'in']) - set([r for r, d in relations if d == 'in'])
                    if diff_out:
                        print(f"      Relazioni OUT diverse: {diff_out}")
                    if diff_in:
                        print(f"      Relazioni IN diverse: {diff_in}")
        else:
            print(f"  ⚠ Nessun nodo con type '{sig[0]}' trovato in G1!")
    
    # Riassunto finale
    print(f"\n{'='*50}")
    print("RIASSUNTO")
    print(f"{'='*50}")
    print(f"Nodi equivalenti: {len(equivalent_nodes_g1_to_g2)}")
    print(f"Nodi non mappati G1: {len(unmapped_g1)}")
    print(f"Nodi non mappati G2: {len(unmapped_g2)}")
    print(f"Percentuale similarità nodi: {len(equivalent_nodes_g1_to_g2) / max(len(signatures_g1), len(signatures_g2)) * 100:.1f}%")
    
    # Analizza i type più comuni
    types_g1 = Counter([sig[0] for sig in signatures_g1.values()])
    types_g2 = Counter([sig[0] for sig in signatures_g2.values()])
    
    print(f"\nDistribuzione types G1: {dict(types_g1)}")
    print(f"Distribuzione types G2: {dict(types_g2)}")
    
    common_types = set(types_g1.keys()) & set(types_g2.keys())
    print(f"Types in comune: {common_types}")
    
    only_g1_types = set(types_g1.keys()) - set(types_g2.keys())
    only_g2_types = set(types_g2.keys()) - set(types_g1.keys())
    
    if only_g1_types:
        print(f"Types solo in G1: {only_g1_types}")
    if only_g2_types:
        print(f"Types solo in G2: {only_g2_types}")

if __name__ == "__main__":
    # Analizza i file Ewelink
    analyze_node_differences(
        "data/macm_files/Ewelink_incorrect.macm",
        "data/macm_files/Ewelink_correct.macm"
    )