# Examples

## Quick Start

Run the basic comparison example:

```bash
python examples/compare.py
```

Expected output:
```
Edit Distance: 0
MCS Ratio (graph1): 100.00%
MCS Ratio (graph2): 100.00%
```

This compares two architecturally identical graphs with different component names:
- `graph1.macm`: WebClient → APIServer → Database
- `graph2.macm`: FrontendApp → BackendService → DataStore

The algorithm correctly identifies them as **identical** (ED=0, MCS=100%) because it compares **structure and types**, not names.

## Understanding the Results

- **Edit Distance = 0**: Graphs are structurally identical
- **MCS Ratio = 100%**: The entire graph is common to both
- **Name independence**: "WebClient" and "FrontendApp" match because both are `Service.WebUI`

## Custom Comparison

Create your own `.macm` files and compare them:

```python
from src import load_graph_from_file, calculate_edit_distance

graph_a = load_graph_from_file('path/to/architecture_a.macm')
graph_b = load_graph_from_file('path/to/architecture_b.macm')

distance = calculate_edit_distance(graph_a, graph_b)
print(f"Graphs differ by {distance} operations")
```

## MACM File Format

Example `.macm` file (Cypher syntax):

```cypher
CREATE 
(client:Component {name: 'WebUI', type: 'Service.WebUI'}),
(server:Component {name: 'API', type: 'Service.API'}),
(client)-[:connects]->(server);
```

**Key points**:
- Use `Component` label for nodes
- Always include `type` property (used for matching)
- `name` property is optional (not used in comparison)
- Relationship types define connectivity patterns
