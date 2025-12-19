"""
Minimal example: Compare two MACM graphs.

This demonstrates the core functionality of the library.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import load_graph_from_file, calculate_edit_distance, calculate_mcs_ratio

# Load graphs from .macm files
script_dir = os.path.dirname(os.path.abspath(__file__))
graph1 = load_graph_from_file(os.path.join(script_dir, 'data/graph1.macm'))
graph2 = load_graph_from_file(os.path.join(script_dir, 'data/graph2.macm'))

# Calculate metrics
edit_distance = calculate_edit_distance(graph1, graph2)
mcs_ratio_g1, mcs_ratio_g2 = calculate_mcs_ratio(graph1, graph2)

# Print results
print(f"Edit Distance: {edit_distance}")
print(f"MCS Ratio (graph1): {mcs_ratio_g1:.2%}")
print(f"MCS Ratio (graph2): {mcs_ratio_g2:.2%}")
