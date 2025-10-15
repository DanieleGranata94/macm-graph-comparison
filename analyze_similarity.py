#!/usr/bin/env python3
"""
Test per analizzare nel dettaglio il calcolo del Similarity Score
"""

import sys
import os
sys.path.append('src')

from config import DatabaseConfig
from database_manager import Neo4jManager
from utils import load_graph_from_cypher_file
from metrics import GraphMetricsCalculator

def analyze_similarity_score(file1, file2):
    """Analizza nel dettaglio il calcolo del Similarity Score"""
    
    print("="*80)
    print("ANALISI DETTAGLIATA DEL SIMILARITY SCORE")
    print("="*80)
    
    # Carica i grafi
    db_config = DatabaseConfig()
    
    print(f"Caricando {file1} e {file2}...")
    with Neo4jManager(db_config) as neo4j_manager:
        graph1 = load_graph_from_cypher_file(neo4j_manager, file1)
        graph2 = load_graph_from_cypher_file(neo4j_manager, file2)
    
    # Calcola le metriche
    calculator = GraphMetricsCalculator()
    result = calculator.compare_graphs(graph1, graph2)
    
    print(f"\nGrafo 1: {len(graph1.nodes)} nodi, {len(graph1.relationships)} relazioni")
    print(f"Grafo 2: {len(graph2.nodes)} nodi, {len(graph2.relationships)} relazioni")
    
    print(f"\n{'='*50}")
    print("METRICHE GREZZE")
    print(f"{'='*50}")
    
    print(f"Edit Distance: {result.edit_distance}")
    print(f"Normalized Edit Distance: {result.normalized_edit_distance:.3f}")
    print(f"MCS Size: {result.maximum_common_subgraph_size}")
    print(f"MCS Ratio Graph1: {result.mcs_ratio_graph1:.3f}")
    print(f"MCS Ratio Graph2: {result.mcs_ratio_graph2:.3f}")
    print(f"Degree Distribution Similarity: {result.degree_distribution_similarity:.3f}")
    print(f"Label Distribution Similarity: {result.label_distribution_similarity:.3f}")
    
    print(f"\n{'='*50}")
    print("CALCOLO SIMILARITY SCORE")
    print(f"{'='*50}")
    
    # Pesi usati nel calcolo
    weights = {
        'edit_distance': 0.3,
        'mcs_ratio': 0.3,
        'degree_similarity': 0.2,
        'label_similarity': 0.2
    }
    
    print("Pesi utilizzati:")
    for metric, weight in weights.items():
        print(f"  {metric}: {weight}")
    
    # Normalizzazione delle metriche
    edit_similarity = 1.0 - result.normalized_edit_distance
    mcs_similarity = (result.mcs_ratio_graph1 + result.mcs_ratio_graph2) / 2.0
    degree_similarity = result.degree_distribution_similarity
    label_similarity = result.label_distribution_similarity
    
    print(f"\nMetriche normalizzate (0-1):")
    print(f"  Edit Similarity: 1.0 - {result.normalized_edit_distance:.3f} = {edit_similarity:.3f}")
    print(f"  MCS Similarity: ({result.mcs_ratio_graph1:.3f} + {result.mcs_ratio_graph2:.3f}) / 2 = {mcs_similarity:.3f}")
    print(f"  Degree Similarity: {degree_similarity:.3f}")
    print(f"  Label Similarity: {label_similarity:.3f}")
    
    # Contributi ponderati
    edit_contrib = weights['edit_distance'] * edit_similarity
    mcs_contrib = weights['mcs_ratio'] * mcs_similarity
    degree_contrib = weights['degree_similarity'] * degree_similarity
    label_contrib = weights['label_similarity'] * label_similarity
    
    print(f"\nContributi al punteggio finale:")
    print(f"  Edit Distance:    {weights['edit_distance']:.1f} × {edit_similarity:.3f} = {edit_contrib:.3f}")
    print(f"  MCS Ratio:        {weights['mcs_ratio']:.1f} × {mcs_similarity:.3f} = {mcs_contrib:.3f}")
    print(f"  Degree Distrib.:  {weights['degree_similarity']:.1f} × {degree_similarity:.3f} = {degree_contrib:.3f}")
    print(f"  Label Distrib.:   {weights['label_similarity']:.1f} × {label_similarity:.3f} = {label_contrib:.3f}")
    
    total_score = edit_contrib + mcs_contrib + degree_contrib + label_contrib
    
    print(f"\nSimilarity Score = {edit_contrib:.3f} + {mcs_contrib:.3f} + {degree_contrib:.3f} + {label_contrib:.3f} = {total_score:.3f}")
    print(f"Similarity Score finale: {result.get_similarity_score():.3f}")
    
    print(f"\n{'='*50}")
    print("ANALISI DEI CONTRIBUTI")
    print(f"{'='*50}")
    
    contributions = [
        ("Edit Distance", edit_contrib),
        ("MCS Ratio", mcs_contrib),
        ("Degree Distribution", degree_contrib),
        ("Label Distribution", label_contrib)
    ]
    
    # Ordina per contributo decrescente
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    print("Contributi ordinati per importanza:")
    for i, (name, contrib) in enumerate(contributions, 1):
        percentage = (contrib / total_score) * 100
        print(f"  {i}. {name:18}: {contrib:.3f} ({percentage:.1f}%)")
    
    print(f"\n{'='*50}")
    print("SPIEGAZIONE DEL PUNTEGGIO ALTO")
    print(f"{'='*50}")
    
    print(f"Il Similarity Score di {result.get_similarity_score():.3f} può sembrare alto nonostante solo 3 nodi equivalenti,")
    print("ma deriva da diversi fattori:")
    print()
    
    if degree_similarity > 0.9:
        print(f"1. DEGREE DISTRIBUTION molto simile ({degree_similarity:.3f}):")
        print("   - I grafi hanno pattern di connettività simili")
        print("   - Nodi con numero di connessioni simile")
        print()
    
    if label_similarity > 0.8:
        print(f"2. LABEL DISTRIBUTION alta ({label_similarity:.3f}):")
        print("   - Tipi di nodi simili in entrambi i grafi")
        print("   - Stessa 'vocabolario' di componenti")
        print()
    
    if edit_similarity < 0.5:
        print(f"3. EDIT DISTANCE penalizza ({edit_similarity:.3f}):")
        print("   - Molte operazioni necessarie per trasformare un grafo nell'altro")
        print()
    else:
        print(f"3. EDIT DISTANCE contribuisce positivamente ({edit_similarity:.3f}):")
        print("   - Relativamente poche operazioni per trasformare i grafi")
        print()
    
    if mcs_similarity < 0.5:
        print(f"4. MCS RATIO basso ({mcs_similarity:.3f}):")
        print("   - Poca sovrapposizione strutturale diretta")
        print("   - Pochi nodi/relazioni identiche")
        print()
    
    print("CONCLUSIONE:")
    print("I grafi hanno STRUTTURA DIVERSA ma CARATTERISTICHE DISTRIBUZIONALI SIMILI!")
    print("Sono 'diversi nel dettaglio ma simili nel complesso'")

if __name__ == "__main__":
    analyze_similarity_score(
        "data/macm_files/Ewelink_incorrect.macm",
        "data/macm_files/Ewelink_correct.macm"
    )