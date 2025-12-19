# MACM Similarity Metrics

Graph Edit Distance (GED) and Maximum Common Subgraph (MCS) for Multi-layer Architecture Component Models.

## What is MACM?

MACM represents software architectures as multi-layer graphs:
- **Nodes**: Components (Hardware, System, Service, Network, Actor)
- **Edges**: Relationships (uses, hosts, connects, interacts)

Reference: https://pennet.vseclab.it/home

## Metrics

### 1. Graph Edit Distance (GED)
Minimum operations to transform one graph into another (add/delete/modify nodes and edges).

### 2. Maximum Common Subgraph (MCS)
Largest isomorphic subgraph between two graphs, measured as overlap ratio (0-100%).

## Key Feature

**Name-independent comparison**: Matches nodes by `(type, incoming_relations, outgoing_relations)`, not by name.

Perfect for comparing LLM-generated vs expert-created architectures.

## Quick Start

```python
from src.models import Graph
from src.metrics import calculate_edit_distance, calculate_mcs_ratio

# Load MACM graphs from Cypher files
graph1 = Graph.from_cypher_file("examples/data/graph1.macm")
graph2 = Graph.from_cypher_file("examples/data/graph2.macm")

# Calculate metrics
ed = calculate_edit_distance(graph1, graph2)
mcs_ratio = calculate_mcs_ratio(graph1, graph2)

print(f"Edit Distance: {ed}")
print(f"MCS Ratio: {mcs_ratio:.1%}")
```

## Installation

```bash
pip install -r requirements.txt
```

## Example

```bash
python examples/compare.py
```

## Algorithm

Node signature:
```python
signature = (node_type, sorted_relations_in_out)
```

Example:
```python
('Service.API', (('uses', 'out', 2), ('connects', 'in', 1)))
```

## Known Limitations

1. **Signature-based matching loses node identity**: Counter-based approach treats `[A_out, B_in] ≡ [A_in, B_out]`
2. **Relation changes = DELETE+ADD**: Modifying relations counts as 2 operations (not 1 MODIFY)
3. **No semantic pattern recognition**: Topologically different = different, even if architecturally equivalent

## License

MIT


