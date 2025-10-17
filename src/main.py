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
            # Use command-line arguments if provided, otherwise use config defaults
            cypher_file1 = file1 if file1 else self.app_config.cypher_file_1
            cypher_file2 = file2 if file2 else self.app_config.cypher_file_2

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
                
                # Create output folder with timestamp and descriptive name
                import os
                import shutil
                from datetime import datetime
                from pathlib import Path
                
                # Extract file names without extensions
                file1_name = Path(cypher_file1).stem
                file2_name = Path(cypher_file2).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_folder = f"output/{file1_name}_vs_{file2_name}_{timestamp}"
                os.makedirs(output_folder, exist_ok=True)
                
                # Copy .macm files to output folder
                try:
                    shutil.copy2(cypher_file1, f"{output_folder}/{Path(cypher_file1).name}")
                    shutil.copy2(cypher_file2, f"{output_folder}/{Path(cypher_file2).name}")
                    self.logger.info(f"Copied input files to {output_folder}/")
                except Exception as e:
                    self.logger.warning(f"Could not copy input files: {e}")
                
                # Save results to file if requested
                if self.app_config.output_format != "text":
                    self._save_results(comparison_result, cypher_file1, cypher_file2)

                # Generate a structured JSON diff describing additions/removals/changes
                try:
                    from diff import generate_graph_diff
                    out_diff = generate_graph_diff(graph1, graph2, comparison_result, output_path=f"{output_folder}/graph_diff.json")
                    self.logger.info(f"Graph diff written to {out_diff}")
                except Exception as e:
                    self.logger.warning(f"Unable to generate graph diff: {e}")
                
                # Genera sempre le visualizzazioni dei grafi di input come PNG
                try:
                    from metrics import generate_graph_visualizations
                    generate_graph_visualizations(graph1, graph2, output_folder, cypher_file1, cypher_file2)
                    self.logger.info(f"Graph visualizations generated in {output_folder}/")
                except Exception as e:
                    self.logger.warning(f"Error generating graph visualizations: {e}")
                
                # Generate README with comparison summary
                try:
                    self._generate_comparison_readme(comparison_result, output_folder, cypher_file1, cypher_file2, graph1, graph2)
                    self.logger.info(f"Comparison README generated in {output_folder}/README.md")
                except Exception as e:
                    self.logger.warning(f"Error generating README: {e}")
                
                self.logger.info("Graph comparison completed successfully")
                return comparison_result, output_folder  # Return also output folder
                
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            if self.app_config.verbose:
                self.logger.exception("Full traceback:")
            sys.exit(1)
    
    def _generate_comparison_readme(self, result, output_folder: str, file1: str, file2: str, graph1, graph2) -> None:
        """
        Generate a README.md file describing the comparison results.
        
        Args:
            result: GraphComparisonResult object
            output_folder: Output folder path
            file1: First file path
            file2: Second file path
            graph1: First Graph object
            graph2: Second Graph object
        """
        from pathlib import Path
        
        readme_content = f"""# Graph Comparison Report

## Comparison Overview

**Date**: {Path(output_folder).name.split('_')[-2] + '_' + Path(output_folder).name.split('_')[-1]}  
**Graph 1**: `{Path(file1).name}`  
**Graph 2**: `{Path(file2).name}`

---

## Graph Statistics

### Graph 1
- **Nodes**: {len(graph1.nodes)}
- **Relationships**: {len(graph1.relationships)}
- **Node Labels**: {dict(graph1.get_label_statistics())}

### Graph 2
- **Nodes**: {len(graph2.nodes)}
- **Relationships**: {len(graph2.relationships)}
- **Node Labels**: {dict(graph2.get_label_statistics())}

---

## Comparison Metrics

### Edit Distance
- **Raw Edit Distance**: {result.edit_distance:.2f}
- **Normalized (0-1)**: {result.normalized_edit_distance:.3f}
- **Interpretation**: {"Identical" if result.edit_distance == 0 else "Very similar" if result.normalized_edit_distance < 0.3 else "Somewhat similar" if result.normalized_edit_distance < 0.7 else "Very different"}

### Maximum Common Subgraph (MCS)
- **MCS Size**: {result.maximum_common_subgraph_size} elements (nodes + edges)
- **MCS Ratio (Graph 1)**: {result.mcs_ratio_graph1:.3f} ({result.mcs_ratio_graph1*100:.1f}%)
- **MCS Ratio (Graph 2)**: {result.mcs_ratio_graph2:.3f} ({result.mcs_ratio_graph2*100:.1f}%)

### Minimum Common Supergraph
- **Supergraph Size**: {result.minimum_common_supergraph_size} elements
- **Supergraph Ratio (Graph 1)**: {result.supergraph_ratio_graph1:.3f}
- **Supergraph Ratio (Graph 2)**: {result.supergraph_ratio_graph2:.3f}

### Structural Similarity
- **Node Count Difference**: {result.node_count_difference}
- **Relationship Count Difference**: {result.relationship_count_difference}
- **Degree Distribution Similarity**: {result.degree_distribution_similarity:.3f}
- **Label Distribution Similarity**: {result.label_distribution_similarity:.3f}

---

## Detailed Differences

### Common Elements
- **Common Nodes**: {len(result.common_nodes)}
- **Common Relationships**: {len(result.common_relationships)}

### Unique Elements
- **Unique to Graph 1**: {len(result.unique_to_graph1)} elements
- **Unique to Graph 2**: {len(result.unique_to_graph2)} elements

---

## Files in This Directory

- `README.md` - This comparison report
- `{Path(file1).name}` - First input MACM file
- `{Path(file2).name}` - Second input MACM file  
- `graph_diff.json` - Detailed JSON diff of the two graphs
- `graph_*_graph.png` - Visualizations of individual graphs
- `graph_comparison.png` - Side-by-side comparison visualization
- `all_metrics_raw.png` - Raw metric values chart
- `all_metrics_normalized_0_1.png` - Normalized metrics chart (0-1 scale)

---

## Interpretation

**Overall Similarity**: {self._get_interpretation_label(result)}

{self._get_interpretation_details(result)}

---

*Generated by Graph Comparison Tool*
"""
        
        readme_path = Path(output_folder) / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def _get_interpretation_label(self, result) -> str:
        """Get interpretation label based on edit distance."""
        if result.edit_distance == 0:
            return "Identical graphs"
        elif result.normalized_edit_distance < 0.3:
            return "Very similar graphs"
        elif result.normalized_edit_distance < 0.6:
            return "Moderately similar graphs"
        elif result.normalized_edit_distance < 0.8:
            return "Somewhat similar graphs"
        else:
            return "Very different graphs"
    
    def _get_interpretation_details(self, result) -> str:
        """Generate detailed interpretation based on metrics."""
        details = []
        
        if result.edit_distance == 0:
            details.append("✅ The graphs are **identical** - no differences detected.")
        elif result.normalized_edit_distance < 0.3:
            details.append("✅ The graphs are **very similar** with minimal structural differences.")
        elif result.normalized_edit_distance < 0.7:
            details.append("⚠️ The graphs have **moderate differences** - some structural variations exist.")
        else:
            details.append("❌ The graphs are **very different** - significant structural differences detected.")
        
        if result.mcs_ratio_graph1 > 0.8:
            details.append(f"- High MCS ratio ({result.mcs_ratio_graph1*100:.0f}%) indicates strong structural overlap.")
        elif result.mcs_ratio_graph1 < 0.2:
            details.append(f"- Low MCS ratio ({result.mcs_ratio_graph1*100:.0f}%) indicates minimal structural overlap.")
        
        if result.node_count_difference == 0 and result.relationship_count_difference == 0:
            details.append("- Graphs have the same size (same number of nodes and relationships).")
        elif abs(result.node_count_difference) > 5 or abs(result.relationship_count_difference) > 5:
            details.append(f"- Significant size difference detected (Δnodes={result.node_count_difference}, Δedges={result.relationship_count_difference}).")
        
        return "\n".join(details)
    
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
    result_tuple = app.run(args.file1, args.file2)
    
    # Unpack result and output folder
    if result_tuple is not None and isinstance(result_tuple, tuple):
        result, output_folder = result_tuple
    else:
        result = result_tuple
        output_folder = "output"
    
    if result is not None:
        from src.metrics import (
            plot_all_metrics_raw,
            plot_all_metrics_normalized,
        )
        # Generate the two requested charts in the specific output folder
        plot_all_metrics_raw(result, output_dir=output_folder)
        plot_all_metrics_normalized(result, output_dir=output_folder)
        
        print(f"Grafici PNG generati nella cartella {output_folder}/")
    else:
        print("Errore: Nessun risultato dalla comparazione dei grafi.")
    
    # Exit message
    print("Graph comparison completed successfully")


if __name__ == "__main__":
    main()
