# Graph Comparison Tool

A professional Python tool for comparing graphs loaded from Neo4j databases using various metrics including edit distance, maximum common subgraph, and isomorphism invariants.

## Features

- **Multiple Comparison Metrics**:
  - Edit Distance calculation
  - Maximum Common Subgraph (MCS) analysis
  - Minimum Common Supergraph computation
  - Isomorphism invariants comparison

- **Professional Architecture**:
  - Modular design with clear separation of concerns
  - Comprehensive error handling and logging
  - Configuration management
  - Command-line interface

- **Flexible Output Formats**:
  - Human-readable text format
  - JSON for programmatic use
  - CSV for data analysis

## Installation

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Neo4j is running**:
   - Install and start Neo4j database
   - Default connection: `bolt://localhost:7687`
   - Default credentials: `neo4j/neo4j`

## Usage

### Basic Usage

```bash
# Compare two Cypher files with default settings
python main.py

# Specify custom files
python main.py --file1 cypher_sbagliato.macm --file2 correct.macm

# Enable verbose output
python main.py --verbose

# Output in JSON format
python main.py --output-format json
```

### Advanced Usage

```bash
# Custom Neo4j connection
python main.py --neo4j-uri bolt://localhost:7687 --neo4j-username neo4j --neo4j-password mypassword

# Debug logging to file
python main.py --log-level DEBUG --log-file comparison.log

# CSV output for data analysis
python main.py --output-format csv
```

### Command Line Options

- `--file1`, `-f1`: Path to first Cypher file
- `--file2`, `-f2`: Path to second Cypher file
- `--verbose`, `-v`: Enable verbose output
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file`: Log file path
- `--output-format`: Output format (text, json, csv)
- `--neo4j-uri`: Neo4j connection URI
- `--neo4j-username`: Neo4j username
- `--neo4j-password`: Neo4j password

### Environment Variables

You can configure the application using environment variables:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="mypassword"
export CYPHER_FILE_1="cypher_sbagliato.macm"
export CYPHER_FILE_2="correct.macm"
export LOG_LEVEL="INFO"
export VERBOSE="true"
export OUTPUT_FORMAT="json"
```

## Project Structure

```
.
├── main.py                 # Main application entry point
├── config.py              # Configuration management
├── database_manager.py    # Neo4j database operations
├── models.py              # Data models and structures
├── metrics.py             # Graph comparison algorithms
├── utils.py               # Utility functions and helpers
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── cypher_sbagliato.macm # Example Cypher file 1
└── correct.macm          # Example Cypher file 2
```

## Comparison Metrics

### 1. Edit Distance
Measures the minimum number of operations (add, delete, substitute nodes/edges) required to transform one graph into another.

## 📐 Come Viene Calcolata l'Edit Distance

L'**Edit Distance** misura il numero minimo di operazioni necessarie per trasformare un grafo in un altro.

### Algoritmo utilizzato:

1. **Converti i grafi in signature canoniche**:
   - **Node Signature**: `(type, sorted_multiset_of_local_relations)`
   - **Edge Signature**: `(start_type, edge_type, end_type)`

2. **Calcola differenze nei multiset**:
   ```
   Edit Distance = |multiset_A - multiset_B| + |multiset_B - multiset_A|
   ```

### Esempio concreto: Test Case 2

#### Grafo A (test_one_node_diff_A.macm):
```cypher
CREATE 
(server:HW:Server {component_id: '1', name: 'Server', type: 'HW.Server'}),
(database:Service:Database {component_id: '2', name: 'Database', type: 'Service.Database'}),
(client:HW:UE {component_id: '3', name: 'Client', type: 'HW.UE'}),
(server)-[:hosts {}]->(database),
(client)-[:connects {}]->(server);
```

- **Nodi**: Server, Database, Client (3 nodi)
- **Relazioni**: Server→hosts→Database, Client→connects→Server (2 relazioni)

**Node Signatures A:**
```python
('HW.Server', (('hosts', 'out'), 1))
('Service.Database', (('hosts', 'in'), 1))
('HW.UE', (('connects', 'out'), 1))
```

**Edge Signatures A:**
```python
('HW.Server', 'hosts', 'Service.Database')
('HW.UE', 'connects', 'HW.Server')
```

#### Grafo B (test_one_node_diff_B.macm):
```cypher
CREATE 
(server:HW:Server {component_id: '1', name: 'Server', type: 'HW.Server'}),
(database:Service:Database {component_id: '2', name: 'Database', type: 'Service.Database'}),
(client:HW:UE {component_id: '3', name: 'Client', type: 'HW.UE'}),
(firewall:Security:Firewall {component_id: '4', name: 'Firewall', type: 'Security.Firewall'}),
(server)-[:hosts {}]->(database),
(client)-[:connects {}]->(firewall),
(firewall)-[:connects {}]->(server);
```

- **Nodi**: Server, Database, Client, **Firewall** (4 nodi)
- **Relazioni**: Server→hosts→Database, Client→connects→**Firewall**, **Firewall→connects→Server** (3 relazioni)

**Node Signatures B:**
```python
('HW.Server', (('connects', 'in'), 1), (('hosts', 'out'), 1))  # ← DIVERSA!
('Service.Database', (('hosts', 'in'), 1))
('HW.UE', (('connects', 'out'), 1))
('Security.Firewall', (('connects', 'in'), 1), (('connects', 'out'), 1))  # ← NUOVA!
```

**Edge Signatures B:**
```python
('HW.Server', 'hosts', 'Service.Database')
('HW.UE', 'connects', 'Security.Firewall')  # ← DIVERSA!
('Security.Firewall', 'connects', 'HW.Server')  # ← NUOVA!
```

### Calcolo Edit Distance:

**Operazioni necessarie:**

1. **Node Operations**:
   - Add: `('Security.Firewall', (('connects', 'in'), 1), (('connects', 'out'), 1))` → **1 operazione**

2. **Edge Operations**:
   - Remove: `('HW.UE', 'connects', 'HW.Server')` → **1 operazione**
   - Add: `('HW.UE', 'connects', 'Security.Firewall')` → **1 operazione**
   - Add: `('Security.Firewall', 'connects', 'HW.Server')` → **1 operazione**

**Risultato:**
- **Edit Distance totale: 4.0** (1 nodo + 3 edge operations)
- **Normalized: 4.0 / 7 = 0.571** (57% di differenza)

### Perché 4 operazioni e non 2?

Anche se logicamente stiamo "aggiungendo un nodo Firewall nel mezzo", l'algoritmo conta:
1. **Aggiunta del nodo Firewall** (1 op)
2. **Rimozione** della connessione diretta Client→Server (1 op)
3. **Aggiunta** di Client→Firewall (1 op)
4. **Aggiunta** di Firewall→Server (1 op)

Questo perché l'Edit Distance opera su **signature strutturali**: quando modifichi la topologia, devi rimuovere le vecchie connessioni e aggiungere le nuove. Non esiste un'operazione atomica "inserisci nodo nel mezzo di un percorso".

### 2. Maximum Common Subgraph (MCS)
Finds the largest subgraph that exists in both graphs, providing insight into shared structure.

### 3. Minimum Common Supergraph
Determines the smallest graph that contains both input graphs as subgraphs.

### 4. Isomorphism Invariants
Compares structural properties that remain unchanged under graph isomorphism:
- Node and edge counts
- Degree distributions
- Label distributions
- Relationship type distributions

### 5. Overall Similarity Score
A weighted combination of all metrics providing a single similarity score between 0 and 1.

## Output Interpretation

- **Similarity Score ≥ 0.8**: Very similar graphs
- **Similarity Score ≥ 0.6**: Moderately similar graphs  
- **Similarity Score ≥ 0.4**: Somewhat similar graphs
- **Similarity Score < 0.4**: Very different graphs

## Development

### Running Tests
```bash
# Install development dependencies
pip install pytest pytest-cov

# Run tests
pytest

# Run with coverage
pytest --cov=.
```

### Code Formatting
```bash
# Install formatting tools
pip install black flake8 mypy

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure Neo4j is running and accessible
2. **File Not Found**: Check that Cypher files exist and are readable
3. **Permission Error**: Verify database credentials and permissions

### Debug Mode
Enable debug logging for detailed information:
```bash
python main.py --log-level DEBUG --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please create an issue in the project repository.
