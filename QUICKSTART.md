# MACM Graph Metrics - Guida Rapida

## 📦 Struttura Finale (Minimalista)

```
Metrics LLM MACM/
├── src/                      # Modulo Python
│   ├── __init__.py          # Exports (GraphMetrics, funzioni)
│   ├── graph_metrics.py     # Logica calcolo metriche
│   ├── models.py            # Graph, GraphNode, GraphRelationship
│   ├── database_manager.py  # Neo4jManager (DatabaseManager)
│   └── utils.py             # load_graph_from_cypher
├── app.py                   # Flask microservice
├── Dockerfile               # Container image
├── docker-compose.yml       # Neo4j + API orchestration
├── test_api.py              # Test script per API
├── requirements.txt         # Dipendenze
├── .env                     # Config locale (gitignored)
└── README.md                # Documentazione
```

## 🚀 Uso Come Modulo Python

```python
from src import calculate_all_metrics_from_cypher

# Opzione 1: Usa parametri espliciti
metrics = calculate_all_metrics_from_cypher(
    "graph1.macm",
    "graph2.macm",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# Opzione 2: Usa defaults (bolt://localhost:7687, neo4j/password)
metrics = calculate_all_metrics_from_cypher("graph1.macm", "graph2.macm")

# Accedi alle metriche
print(f"Edit Distance: {metrics.edit_distance}")
print(f"Normalized: {metrics.normalized_edit_distance}")
print(f"MCS Size: {metrics.mcs_size}")
print(f"MCS Ratio Graph1: {metrics.mcs_ratio_graph1}")
print(f"MCS Ratio Graph2: {metrics.mcs_ratio_graph2}")
```

## 🐳 Uso Come Microservizio Docker

### Avvio Rapido

```bash
# 1. Build e start (Neo4j + API)
docker-compose up -d

# 2. Verifica status
docker-compose ps

# 3. Test health
curl http://localhost:5000/health
```

### Test con i tuoi file MACM

```bash
curl -X POST http://localhost:5000/metrics \
  -F "file1=@data/macm_files/Ewelink_correct.macm" \
  -F "file2=@data/macm_files/Ewelink_incorrect.macm"
```

### Risposta Esempio

```json
{
  "edit_distance": 87.0,
  "normalized_edit_distance": 0.756,
  "mcs_size": 0,
  "mcs_ratio_graph1": 0.0,
  "mcs_ratio_graph2": 0.0
}
```

### Stop

```bash
docker-compose down
```

## 🔌 API Endpoints

### `GET /health`
Health check del servizio

### `POST /metrics`
Calcola metriche tra due grafi MACM

**Parameters:**
- `file1`: File MACM (multipart/form-data)
- `file2`: File MACM (multipart/form-data)

**Response:**
```json
{
  "edit_distance": 5.0,
  "normalized_edit_distance": 0.25,
  "mcs_size": 10,
  "mcs_ratio_graph1": 0.8,
  "mcs_ratio_graph2": 0.75
}
```

## 📊 Metriche

| Metrica | Descrizione | Range |
|---------|-------------|-------|
| `edit_distance` | Numero operazioni (add/remove/modify) | [0, ∞) |
| `normalized_edit_distance` | Edit distance normalizzata | [0, 1] |
| `mcs_size` | Dimensione Maximum Common Subgraph | [0, ∞) |
| `mcs_ratio_graph1` | MCS / size(graph1) | [0, 1] |
| `mcs_ratio_graph2` | MCS / size(graph2) | [0, 1] |

## 🛠 Configurazione

### Variabili d'Ambiente (Docker)

```bash
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### File .env (Locale)

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password
```

## 🧪 Test Rapido

### Test Modulo Python

```bash
# Avvia Neo4j locale
neo4j start

# Test con Python
python -c "
from src import calculate_all_metrics_from_cypher
m = calculate_all_metrics_from_cypher(
    'data/macm_files/Ewelink_correct.macm',
    'data/macm_files/Ewelink_incorrect.macm'
)
print(f'Edit Distance: {m.edit_distance}')
"
```

### Test Microservizio

```bash
# Start servizio
python app.py &

# Attendi 2 secondi
sleep 2

# Test
python test_api.py

# Stop servizio
pkill -f "python app.py"
```

## 🔧 Troubleshooting

### Problema: "Cannot connect to Neo4j"

**Soluzione:**
```bash
# Verifica Neo4j sia attivo
neo4j status

# O con Docker
docker-compose ps neo4j
```

### Problema: "Module 'src' not found"

**Soluzione:**
```bash
# Verifica di essere nella root del progetto
pwd  # Dovrebbe essere .../Metrics LLM MACM

# Verifica src/__init__.py esiste
ls src/__init__.py
```

### Problema: "Port 5000 already in use"

**Soluzione:**
```bash
# Trova processo sulla porta 5000
lsof -ti:5000

# Termina processo
kill -9 $(lsof -ti:5000)
```

## 📝 Note di Sviluppo

- **Algoritmo Edit Distance**: Riconosce modifiche di tipo come 1 operazione (non 6)
- **Neo4j Requirement**: Necessario per parsing file MACM
- **Stateless API**: Ogni richiesta è indipendente
- **Temp Files**: File MACM caricati temporaneamente, cancellati dopo uso

## 🎯 Next Steps

1. ✅ Modulo Python funzionante
2. ✅ Microservizio Docker
3. ⏳ Deploy su cloud (AWS/GCP/Azure)
4. ⏳ Autenticazione API (JWT)
5. ⏳ Rate limiting
6. ⏳ Caching risultati

## 📄 License

MIT License
