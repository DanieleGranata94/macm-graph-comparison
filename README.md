# MACM - Model for Architecture and Component Management

Un tool professionale per il confronto di architetture di sistemi utilizzando Neo4j e metriche di similarità grafi.

## 🏗️ Struttura del Progetto

```
MACM/
├── src/                          # Codice sorgente Python
│   ├── main.py                   # Entry point principale
│   ├── config.py                 # Configurazione
│   ├── database_manager.py       # Gestione Neo4j
│   ├── models.py                 # Modelli dati
│   ├── metrics.py                # Algoritmi di confronto
│   ├── utils.py                  # Utility e helpers
│   ├── load_results_to_neo4j.py  # Script di caricamento Neo4j
│   └── query_macm_results.py     # Script di query Neo4j
├── data/                         # Dati del progetto
│   └── macm_files/              # File MACM
│       ├── wrong_architecture.macm    # Architettura errata
│       └── correct_architecture.macm  # Architettura corretta
├── docs/                         # Documentazione
│   └── README.md                # Documentazione principale
├── scripts/                      # Script di utilità
├── tests/                        # Test unitari
├── requirements.txt              # Dipendenze Python
└── venv/                        # Ambiente virtuale Python
```

## 🚀 Installazione e Setup

### 1. Clona e Prepara l'Ambiente
```bash
cd "Metrics LLM MACM"
python3 -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Avvia Neo4j
- Installa e avvia Neo4j Database
- URI di default: `bolt://localhost:7687`
- Credenziali di default: `neo4j/neo4j`

## 📊 Utilizzo

### Confronto Grafi
```bash
# Confronto con configurazione di default
cd src
python main.py

# Confronto con file specifici
python main.py --file1 ../data/macm_files/wrong_architecture.macm --file2 ../data/macm_files/correct_architecture.macm

# Output dettagliato
python main.py --verbose

# Output in formato JSON
python main.py --output-format json
```

### Caricamento su Neo4j
```bash
cd src
python load_results_to_neo4j.py
```

### Query Neo4j
```bash
cd src
python query_macm_results.py
```

## 🔍 Metriche di Confronto

Il tool implementa diverse metriche per confrontare architetture:

### 1. **Edit Distance**
Misura il numero minimo di operazioni (add, delete, substitute) per trasformare un grafo nell'altro.

### 2. **Maximum Common Subgraph (MCS)**
Trova il più grande sottografo comune tra i due grafi.

### 3. **Minimum Common Supergraph**
Determina il più piccolo grafo che contiene entrambi i grafi.

### 4. **Isomorphism Invariants**
Confronta proprietà strutturali invarianti:
- Conteggio nodi e relazioni
- Distribuzione dei gradi
- Distribuzione delle etichette

### 5. **Similarity Score**
Score complessivo tra 0 e 1 che combina tutte le metriche.

## 📁 File MACM

### `wrong_architecture.macm`
- **Descrizione**: Architettura errata del sistema
- **Componenti**: 15 nodi, 14 relazioni
- **Tipo**: Sistema IoT robotico con Niryo

### `correct_architecture.macm`
- **Descrizione**: Architettura corretta del sistema
- **Componenti**: 28 nodi, 31 relazioni
- **Tipo**: Sistema industriale SCADA complesso

## ⚙️ Configurazione

### Variabili d'Ambiente
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="mypassword"
export CYPHER_FILE_1="data/macm_files/wrong_architecture.macm"
export CYPHER_FILE_2="data/macm_files/correct_architecture.macm"
export LOG_LEVEL="INFO"
export VERBOSE="true"
export OUTPUT_FORMAT="json"
```

### Configurazione File
Il file `src/config.py` contiene tutte le impostazioni di configurazione.

## 🔧 Comandi Utili

### Sviluppo
```bash
# Formattazione codice
black src/

# Linting
flake8 src/

# Test
pytest tests/
```

### Neo4j Browser
Una volta caricati i dati, puoi esplorare con:
```cypher
// Vedi tutti i nodi
MATCH (n) RETURN n LIMIT 25

// Vedi le relazioni
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25

// Statistiche
MATCH (n) RETURN labels(n), count(*) as count
```

## 📈 Risultati Tipici

Per i file MACM inclusi:
- **Similarity Score**: 0.000 (grafi completamente diversi)
- **Edit Distance**: 87
- **Common Nodes**: 0
- **Interpretation**: "Very different graphs"

## 🤝 Contributi

1. Fork del repository
2. Crea un branch per la feature
3. Implementa le modifiche
4. Aggiungi test
5. Submit pull request

## 📄 Licenza

MIT License

## 🆘 Supporto

Per problemi e domande, crea un issue nel repository del progetto.