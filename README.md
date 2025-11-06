# MACM Graph Metrics Module# MACM Graph Metrics Module



Modulo Python minimalista per calcolare metriche di similarità tra grafi MACM.Modulo Python minimalista per calcolare metriche di similarità tra grafi MACM.



## 🚀 Quick Start│   ├── config.py                 # Configurazione



### Come modulo Python## 🚀 Quick Start│   ├── database_manager.py       # Gestione Neo4j



```python│   ├── models.py                 # Modelli dati

from src import calculate_all_metrics_from_cypher

### Installation│   ├── metrics.py                # Algoritmi di confronto

# Calcola tutte le metriche

metrics = calculate_all_metrics_from_cypher(│   ├── utils.py                  # Utility e helpers

    "graph1.macm",

    "graph2.macm",```bash│   ├── load_results_to_neo4j.py  # Script di caricamento Neo4j

    neo4j_uri="bolt://localhost:7687",

    neo4j_user="neo4j",# Clone repository│   └── query_macm_results.py     # Script di query Neo4j

    neo4j_password="password"

)git clone https://github.com/DanieleGranata94/macm-graph-comparison.git├── data/                         # Dati del progetto



print(f"Edit Distance: {metrics.edit_distance}")cd macm-graph-comparison│   └── macm_files/              # File MACM

print(f"MCS Size: {metrics.mcs_size}")

```│       ├── wrong_architecture.macm    # Architettura errata



### Come microservizio Docker# Install dependencies│       └── correct_architecture.macm  # Architettura corretta



```bashpip install -r requirements.txt├── docs/                         # Documentazione

# Avvia Neo4j + API

docker-compose up -d│   └── README.md                # Documentazione principale



# Testa l'API# Start Neo4j (required)├── scripts/                      # Script di utilità

curl -X POST http://localhost:5000/metrics \

  -F "file1=@graph1.macm" \neo4j start├── tests/                        # Test unitari

  -F "file2=@graph2.macm"

``````├── requirements.txt              # Dipendenze Python



Risposta:└── venv/                        # Ambiente virtuale Python

```json

{### Basic Usage```

  "edit_distance": 5.0,

  "normalized_edit_distance": 0.25,

  "mcs_size": 10,

  "mcs_ratio_graph1": 0.8,```python## 🚀 Installazione e Setup

  "mcs_ratio_graph2": 0.75

}from src.graph_metrics import calculate_all_metrics_from_cypher

```

### 1. Clona e Prepara l'Ambiente

## 📦 Installazione

# Compare two MACM graphs```bash

```bash

# Clone repositorymetrics = calculate_all_metrics_from_cypher("graph1.macm", "graph2.macm")cd "Metrics LLM MACM"

git clone <repo-url>

cd Metrics\ LLM\ MACMpython3 -m venv venv



# Crea virtual environmentprint(f"Edit Distance: {metrics.edit_distance}")source venv/bin/activate  # Su Windows: venv\Scripts\activate

python -m venv .venv

source .venv/bin/activateprint(f"MCS Ratio: {metrics.mcs_ratio_graph1:.3f}")pip install -r requirements.txt



# Installa dipendenzeprint(f"Type Modifications: {metrics.node_type_modifications}")```

pip install -r requirements.txt

``````



## 🐳 Docker### 2. Avvia Neo4j



### Build immagine## 📁 Project Structure- Installa e avvia Neo4j Database



```bash- URI di default: `bolt://localhost:7687`

docker build -t macm-metrics .

``````- Credenziali di default: `neo4j/neo4j`



### Run con docker-compose (consigliato)macm-graph-comparison/



```bash├── src/## 📊 Utilizzo

# Start

docker-compose up -d│   ├── graph_metrics.py      # Main module (use this!)



# Stop│   ├── models.py              # Graph data structures### Confronto Grafi

docker-compose down

│   ├── database_manager.py   # Neo4j connection```bash

# Logs

docker-compose logs -f metrics-api│   └── utils.py               # Helper functions# Confronto con configurazione di default

```

├── examples/cd src

### Run standalone

│   └── usage_examples.py      # Complete working examplespython main.py

```bash

# Start Neo4j├── docs/

docker run -d \

  --name neo4j \│   └── GRAPH_METRICS_MODULE.md  # Full API documentation# Confronto con file specifici

  -p 7474:7474 -p 7687:7687 \

  -e NEO4J_AUTH=neo4j/password \├── data/python main.py --file1 ../data/macm_files/wrong_architecture.macm --file2 ../data/macm_files/correct_architecture.macm

  neo4j:latest

│   └── test_macm/             # Test cases for validation

# Run API

docker run -d \├── AssetTypes.xlsm            # Asset type configuration# Output dettagliato

  --name metrics-api \

  -p 5000:5000 \├── requirements.txt           # Python dependenciespython main.py --verbose

  -e NEO4J_URI=bolt://neo4j:7687 \

  -e NEO4J_USER=neo4j \└── README.md                  # This file

  -e NEO4J_PASSWORD=password \

  --link neo4j \```# Output in formato JSON

  macm-metrics

```python main.py --output-format json



## 🔌 API Endpoints## 📚 Documentation```



### Health Check

```bash

GET /health**Full API Documentation**: See [`docs/GRAPH_METRICS_MODULE.md`](docs/GRAPH_METRICS_MODULE.md)### Caricamento su Neo4j

```

```bash

### Calculate Metrics

```bash### Key Functionscd src

POST /metrics

Content-Type: multipart/form-datapython load_results_to_neo4j.py



Parameters:#### 1. Calculate Edit Distance```

- file1: MACM file (primo grafo)

- file2: MACM file (secondo grafo)

```

```python### Query Neo4j

Esempio con curl:

```bashfrom src.graph_metrics import calculate_edit_distance_from_cypher```bash

curl -X POST http://localhost:5000/metrics \

  -F "file1=@data/macm_files/Ewelink_correct.macm" \cd src

  -F "file2=@data/macm_files/Ewelink_incorrect.macm"

```edit_distance, operations = calculate_edit_distance_from_cypher(python query_macm_results.py



Esempio con Python:    "graph1.macm", ```

```python

import requests    "graph2.macm"



files = {)## 🔍 Metriche di Confronto

    'file1': open('graph1.macm', 'rb'),

    'file2': open('graph2.macm', 'rb')

}

print(f"Distance: {edit_distance}")Il tool implementa diverse metriche per confrontare architetture:

response = requests.post('http://localhost:5000/metrics', files=files)

print(response.json())print(f"Node Type Modifications: {operations['node_type_modifications']}")

```

```### 1. **Edit Distance**

## 📊 Metriche Disponibili

Misura il numero minimo di operazioni (add, delete, substitute) per trasformare un grafo nell'altro.

- **edit_distance**: Numero di operazioni (add/remove node/edge, modify type)

- **normalized_edit_distance**: Edit distance normalizzata [0-1]#### 2. Calculate Maximum Common Subgraph

- **mcs_size**: Dimensione del Maximum Common Subgraph

- **mcs_ratio_graph1**: Rapporto MCS / size graph1### 2. **Maximum Common Subgraph (MCS)**

- **mcs_ratio_graph2**: Rapporto MCS / size graph2

```pythonTrova il più grande sottografo comune tra i due grafi.

## 🛠 Configurazione

from src.graph_metrics import calculate_maximum_common_subgraph_from_cypher

### Environment Variables

### 3. **Minimum Common Supergraph**

```bash

NEO4J_URI=bolt://localhost:7687mcs_size, common_nodes, common_edges = calculate_maximum_common_subgraph_from_cypher(Determina il più piccolo grafo che contiene entrambi i grafi.

NEO4J_USER=neo4j

NEO4J_PASSWORD=your_password    "graph1.macm", 

```

    "graph2.macm"### 4. **Isomorphism Invariants**

### File .env (locale)

)Confronta proprietà strutturali invarianti:

```bash

# Copia template- Conteggio nodi e relazioni

cp .env.example .env

print(f"MCS Size: {mcs_size} ({len(common_nodes)} nodes, {len(common_edges)} edges)")- Distribuzione dei gradi

# Edita con i tuoi valori

vim .env```- Distribuzione delle etichette

```



## 📁 Struttura

#### 3. Calculate All Metrics (Recommended)### 5. **Similarity Score**

```

.Score complessivo tra 0 e 1 che combina tutte le metriche.

├── src/                      # Modulo principale

│   ├── __init__.py          # Exports```python

│   ├── graph_metrics.py     # Calcolo metriche

│   ├── models.py            # Strutture datifrom src.graph_metrics import calculate_all_metrics_from_cypher## 📁 File MACM

│   ├── database_manager.py  # Neo4j manager

│   └── utils.py             # Helper functions

├── app.py                   # Flask microservice

├── Dockerfile               # Docker imagemetrics = calculate_all_metrics_from_cypher("graph1.macm", "graph2.macm")### `wrong_architecture.macm`

├── docker-compose.yml       # Orchestrazione

└── requirements.txt         # Dipendenze Python- **Descrizione**: Architettura errata del sistema

```

# Access all metrics- **Componenti**: 15 nodi, 14 relazioni

## 📝 License

print(f"Edit Distance: {metrics.edit_distance}")- **Tipo**: Sistema IoT robotico con Niryo

MIT License - see LICENSE file

print(f"Normalized Edit Distance: {metrics.normalized_edit_distance:.3f}")

print(f"MCS Size: {metrics.mcs_size}")### `correct_architecture.macm`

print(f"MCS Ratio: {metrics.mcs_ratio_graph1:.3f}")- **Descrizione**: Architettura corretta del sistema

print(f"Node Type Modifications: {metrics.node_type_modifications}")- **Componenti**: 28 nodi, 31 relazioni

```- **Tipo**: Sistema industriale SCADA complesso



## 🎓 Examples## ⚙️ Configurazione



Run the examples to see the module in action:### Variabili d'Ambiente

```bash

```bashexport NEO4J_URI="bolt://localhost:7687"

cd macm-graph-comparisonexport NEO4J_USERNAME="neo4j"

source .venv/bin/activateexport NEO4J_PASSWORD="mypassword"

python examples/usage_examples.pyexport CYPHER_FILE_1="data/macm_files/wrong_architecture.macm"

```export CYPHER_FILE_2="data/macm_files/correct_architecture.macm"

export LOG_LEVEL="INFO"

### Example Outputexport VERBOSE="true"

export OUTPUT_FORMAT="json"

``````

EXAMPLE 1: Simple Edit Distance Calculation

Comparing: test_identical_A.macm vs test_identical_B.macm### Configurazione File

Edit Distance: 0.0Il file `src/config.py` contiene tutte le impostazioni di configurazione.



EXAMPLE 3: All Metrics Calculation## 🔧 Comandi Utili

Edit Distance: 1.0

Normalized Edit Distance: 0.143### Sviluppo

MCS Ratio (Graph 1): 0.571```bash

Type Modifications: 1# Formattazione codice

  Modification 1: Security.Firewall → Network.Firewallblack src/

```

# Linting

## 🔍 Algorithm Highlightsflake8 src/



### Edit Distance with Type Modification Detection# Test

pytest tests/

Traditional graph edit distance counts changing a node type as deletion + insertion (2+ operations).```



Our algorithm detects **type modifications** as a single operation:### Neo4j Browser

Una volta caricati i dati, puoi esplorare con:

**Example:**```cypher

```cypher// Vedi tutti i nodi

// BeforeMATCH (n) RETURN n LIMIT 25

(fw:Security:Firewall {type: 'Security.Firewall'})

// Vedi le relazioni

// After  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25

(fw:Network:Firewall {type: 'Network.Firewall'})

```// Statistiche

MATCH (n) RETURN labels(n), count(*) as count

**Traditional**: 6 operations (1 delete + 1 insert + 4 edge changes)  ```

**Our algorithm**: 1 operation (type modification) ✅

## 📈 Risultati Tipici

This better reflects semantic reality where changing an asset type is a refinement, not a complete restructure.

Per i file MACM inclusi:

## 🔧 Integration in Your Application- **Similarity Score**: 0.000 (grafi completamente diversi)

- **Edit Distance**: 87

### Web API Example- **Common Nodes**: 0

- **Interpretation**: "Very different graphs"

```python

from flask import Flask, request, jsonify## 🤝 Contributi

from src.graph_metrics import calculate_all_metrics_from_cypher

1. Fork del repository

app = Flask(__name__)2. Crea un branch per la feature

3. Implementa le modifiche

@app.route('/compare', methods=['POST'])4. Aggiungi test

def compare_graphs():5. Submit pull request

    data = request.json

    metrics = calculate_all_metrics_from_cypher(## 📄 Licenza

        data['graph1'], 

        data['graph2']MIT License

    )

    ## 🆘 Supporto

    return jsonify({

        'edit_distance': metrics.edit_distance,Per problemi e domande, crea un issue nel repository del progetto.
        'mcs_ratio': metrics.mcs_ratio_graph1,
        'similarity_score': 1 - metrics.normalized_edit_distance
    })
```

### Batch Processing

```python
from src.graph_metrics import calculate_all_metrics_from_cypher

graph_pairs = [
    ("v1.macm", "v2.macm"),
    ("v2.macm", "v3.macm"),
]

for g1, g2 in graph_pairs:
    metrics = calculate_all_metrics_from_cypher(g1, g2)
    print(f"{g1} vs {g2}: Edit Distance = {metrics.edit_distance}")
```

## 📊 Test Cases

The module includes comprehensive test cases in `data/test_macm/`:

- **test_identical**: Edit Distance = 0 (100% similar)
- **test_type_change**: Edit Distance = 1 (type modification only)
- **test_one_node_diff**: Edit Distance = 4 (one node difference)
- **test_medium_similarity**: Edit Distance = 3 (partial overlap)
- **test_completely_different**: High edit distance (no overlap)

Run tests to validate functionality:

```python
from src.graph_metrics import calculate_edit_distance_from_cypher

# Test type modification detection
edit_dist, ops = calculate_edit_distance_from_cypher(
    "data/test_macm/test_type_change_A.macm",
    "data/test_macm/test_type_change_B.macm"
)

assert edit_dist == 1.0  # Should be 1, not 6!
assert ops['node_type_modifications'] == 1
print("✅ Type modification detection works!")
```

## 🛠️ Configuration

### Neo4j Connection

Default connection: `bolt://localhost:7687` with user `neo4j` and password `password`

To use custom credentials:

```python
from src.graph_metrics import calculate_all_metrics_from_cypher

metrics = calculate_all_metrics_from_cypher(
    "graph1.macm",
    "graph2.macm",
    neo4j_uri="bolt://custom-host:7687",
    neo4j_user="custom_user",
    neo4j_password="custom_password"
)
```

### Custom Logger

```python
import logging
from src.graph_metrics import GraphMetricsCalculator

logger = logging.getLogger("my_app")
logger.setLevel(logging.DEBUG)

calculator = GraphMetricsCalculator(logger=logger)
```

## 📦 Dependencies

- `neo4j`: Neo4j Python driver
- Python 3.8+

See `requirements.txt` for complete list.

## 🤝 Contributing

This is a clean, focused module. To contribute:

1. Keep it simple - this is a library, not an application
2. Add tests for new metrics in `data/test_macm/`
3. Update documentation in `docs/GRAPH_METRICS_MODULE.md`
4. Follow the existing code style

## 📝 License

See [LICENSE](LICENSE) file.

## 👤 Author

Daniele Granata - November 2025

## 📖 Citation

If you use this module in research, please cite:

```bibtex
@software{macm_graph_metrics,
  title = {MACM Graph Metrics Module},
  author = {Granata, Daniele},
  year = {2025},
  url = {https://github.com/DanieleGranata94/macm-graph-comparison}
}
```

## 🔗 Related Resources

- **MACM Schema Documentation**: See project wiki
- **Asset Type Catalogue**: `AssetTypes.xlsm`
- **Full API Reference**: `docs/GRAPH_METRICS_MODULE.md`

---

**Ready to use?** Start with `examples/usage_examples.py` to see the module in action! 🚀
