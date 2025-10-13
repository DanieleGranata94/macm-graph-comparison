#!/usr/bin/env python3
"""
Graph Comparison Tool - Main Application

This is the main entry point for the Graph Comparison Tool.
It compares two graphs loaded from Cypher files and provides detailed metrics.

Usage:
    python main.py [options]

Example:
    python main.py --file1 cypher_sbagliato.macm --file2 correct.macm --verbose
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import sys
import logging
from pathlib import Path

from config import get_config, DatabaseConfig, AppConfig
from database_manager import Neo4jManager
from metrics import GraphMetricsCalculator
from utils import setup_logging, load_graph_from_cypher_file, format_comparison_results, validate_cypher_files


class GraphComparisonApp:
    """Main application class for graph comparison."""
    
    def __init__(self, db_config: DatabaseConfig, app_config: AppConfig):
        """
        Initialize the application.
        
        Args:
            db_config: Database configuration
            app_config: Application configuration
        """
        self.db_config = db_config
        self.app_config = app_config
        self.logger = logging.getLogger(__name__)
        
    def run(self, file1: str = None, file2: str = None) -> None:
        """
        Run the graph comparison application.
        
        Args:
            file1: Path to first Cypher file (overrides config)
            file2: Path to second Cypher file (overrides config)
        """
        try:
            # Always use files defined in configuration (config.py or env vars)
            cypher_file1 = self.app_config.cypher_file_1
            cypher_file2 = self.app_config.cypher_file_2

            self.logger.info(f"Starting graph comparison between {cypher_file1} and {cypher_file2}")
            
            # Validate input files
            if not validate_cypher_files(cypher_file1, cypher_file2):
                self.logger.error("Invalid Cypher files provided")
                sys.exit(1)
            
            # Initialize database manager
            with Neo4jManager(self.db_config) as neo4j_manager:
                
                # Load first graph
                self.logger.info(f"Loading first graph from {cypher_file1}")
                graph1 = load_graph_from_cypher_file(neo4j_manager, cypher_file1)
                
                if self.app_config.verbose:
                    stats1 = graph1.get_label_statistics()
                    self.logger.info(f"Graph 1 loaded: {len(graph1.nodes)} nodes, {len(graph1.relationships)} relationships")
                    self.logger.info(f"Graph 1 labels: {stats1}")
                
                # Load second graph
                self.logger.info(f"Loading second graph from {cypher_file2}")
                graph2 = load_graph_from_cypher_file(neo4j_manager, cypher_file2)
                
                if self.app_config.verbose:
                    stats2 = graph2.get_label_statistics()
                    self.logger.info(f"Graph 2 loaded: {len(graph2.nodes)} nodes, {len(graph2.relationships)} relationships")
                    self.logger.info(f"Graph 2 labels: {stats2}")
                
                # Compare graphs
                self.logger.info("Starting graph comparison analysis")
                metrics_calculator = GraphMetricsCalculator()
                comparison_result = metrics_calculator.compare_graphs(graph1, graph2)
                
                # Display results
                self._display_results(comparison_result)
                
                # Save results to file if requested
                if self.app_config.output_format != "text":
                    self._save_results(comparison_result, cypher_file1, cypher_file2)

                # Generate a structured JSON diff describing additions/removals/changes
                try:
                    from diff import generate_graph_diff
                    out_diff = generate_graph_diff(graph1, graph2, comparison_result, output_path="output/graph_diff.json")
                    self.logger.info(f"Graph diff written to {out_diff}")
                except Exception as e:
                    self.logger.warning(f"Unable to generate graph diff: {e}")
                
                # Se i file sono i JetRacer, genera i grafici PNG
                from metrics import plot_jetracer_comparison
                if ("JetRacerMACM_correct.macm" in cypher_file1 and "JetRacerMACM_incorrect.macm" in cypher_file2) or ("JetRacerMACM_incorrect.macm" in cypher_file1 and "JetRacerMACM_correct.macm" in cypher_file2):
                    # Confronto bidirezionale: produci 3 grafici richiesti
                    result1 = metrics_calculator.compare_graphs(graph1, graph2)
                    result2 = metrics_calculator.compare_graphs(graph2, graph1)
                    from metrics import plot_all_metrics_raw, plot_metrics_in_0_1_range, plot_all_metrics_normalized
                    # Produci grafici per result1 (graph1 vs graph2)
                    plot_all_metrics_raw(result1)
                    plot_metrics_in_0_1_range(result1)
                    plot_all_metrics_normalized(result1)
                    self.logger.info("Grafici PNG generati nella cartella output/")
                self.logger.info("Graph comparison completed successfully")
                return comparison_result
                
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            if self.app_config.verbose:
                self.logger.exception("Full traceback:")
            sys.exit(1)
    
    def _display_results(self, result) -> None:
        """
        Display comparison results.
        
        Args:
            result: GraphComparisonResult object
        """
        formatted_output = format_comparison_results(result, self.app_config.output_format)
        
        if self.app_config.output_format == "json":
            print(formatted_output)
        else:
            print(formatted_output)
    
    def _save_results(self, result, file1: str, file2: str) -> None:
        """
        Save results to file.
        
        Args:
            result: GraphComparisonResult object
            file1: First file name
            file2: Second file name
        """
        try:
            # Generate output filename
            file1_name = Path(file1).stem
            file2_name = Path(file2).stem
            output_filename = f"comparison_{file1_name}_vs_{file2_name}.{self.app_config.output_format}"
            
            formatted_output = format_comparison_results(result, self.app_config.output_format)
            
            with open(output_filename, 'w') as f:
                f.write(formatted_output)
            
            self.logger.info(f"Results saved to {output_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare two graphs loaded from Cypher files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --file1 graph1.cypher --file2 graph2.cypher
  python main.py --verbose --output-format json
  python main.py --log-level DEBUG --log-file comparison.log

Environment Variables:
  NEO4J_URI          Neo4j connection URI (default: bolt://localhost:7687)
  NEO4J_USERNAME     Neo4j username (default: neo4j)
  NEO4J_PASSWORD     Neo4j password (default: neo4j)
  CYPHER_FILE_1      First Cypher file path
  CYPHER_FILE_2      Second Cypher file path
  LOG_LEVEL          Logging level (default: INFO)
  VERBOSE            Enable verbose output (default: false)
  OUTPUT_FORMAT      Output format: text, json, csv (default: text)
        """
    )
    
    parser.add_argument(
        "--file1", "-f1",
        help="Path to first Cypher file"
    )
    
    parser.add_argument(
        "--file2", "-f2",
        help="Path to second Cypher file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format"
    )
    
    parser.add_argument(
        "--neo4j-uri",
        help="Neo4j connection URI"
    )
    
    parser.add_argument(
        "--neo4j-username",
        help="Neo4j username"
    )
    
    parser.add_argument(
        "--neo4j-password",
        help="Neo4j password"
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Get configuration
    db_config, app_config = get_config()

    # Override config with command line arguments
    if args.neo4j_uri:
        db_config.uri = args.neo4j_uri
    if args.neo4j_username:
        db_config.username = args.neo4j_username
    if args.neo4j_password:
        db_config.password = args.neo4j_password

    # Setup logging
    setup_logging(app_config)

    # Create and run application
    app = GraphComparisonApp(db_config, app_config)
    result = app.run(args.file1, args.file2)
    
    output_dir = "output"
    if result is not None:
        from src.metrics import (
            plot_similarity_metrics_all,
            plot_all_metrics_raw,
            plot_metrics_in_0_1_range,
            plot_all_metrics_normalized,
        )
        # Generate the three requested charts
        plot_all_metrics_raw(result, output_dir=output_dir)
        plot_metrics_in_0_1_range(result, output_dir=output_dir)
        plot_all_metrics_normalized(result, output_dir=output_dir)
    else:
        print("Errore: Nessun risultato dalla comparazione dei grafi.")
    
    # Exit message
    print(f"Grafici PNG generati nella cartella {output_dir}/")
    print("Graph comparison completed successfully")


if __name__ == "__main__":
    main()
