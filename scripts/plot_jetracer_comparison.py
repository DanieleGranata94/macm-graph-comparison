#!/usr/bin/env python3
"""
Script per generare grafici comparativi tra JetRacerMACM_correct.macm e JetRacerMACM_incorrect.macm.
"""

from src.metrics import GraphMetricsCalculator, plot_jetracer_comparison
from src.utils import load_graph_from_cypher_file
from src.database_manager import Neo4jManager
from src.config import get_config

if __name__ == "__main__":
    db_config, app_config = get_config()
    with Neo4jManager(db_config) as neo4j_manager:
        # Carica i due grafi JetRacer
        graph_correct = load_graph_from_cypher_file(neo4j_manager, "MACMs/JetRacerMACM_correct.macm")
        graph_incorrect = load_graph_from_cypher_file(neo4j_manager, "MACMs/JetRacerMACM_incorrect.macm")

        # Calcola i risultati di confronto
        calculator = GraphMetricsCalculator()
        result_correct = calculator.compare_graphs(graph_correct, graph_incorrect)
        result_incorrect = calculator.compare_graphs(graph_incorrect, graph_correct)

        # Genera i grafici
        plot_jetracer_comparison(result_correct, result_incorrect)
