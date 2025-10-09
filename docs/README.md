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
