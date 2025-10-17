# Test MACM Files - Documentazione

Questa cartella contiene file MACM semplici per testare le metriche di confronto grafi.

## 📋 Test Cases

### Test 1: Grafi Identici
- **File**: `test_identical_A.macm`, `test_identical_B.macm`
- **Descrizione**: Due grafi identici con 4 nodi e 3 relazioni
- **Risultati attesi**:
  - Edit Distance: **0**
  - MCS Ratio: **100%**
  - Supergraph Ratio: **100%**
  - Common Nodes: **4**

### Test 2: Un Nodo di Differenza
- **File**: `test_one_node_diff_A.macm`, `test_one_node_diff_B.macm`
- **Descrizione**: Grafo B ha un nodo aggiuntivo (Firewall) rispetto ad A
- **Risultati attesi**:
  - Edit Distance: **~2** (1 nodo + 1 relazione aggiuntiva)
  - MCS Size: **3** nodi comuni
  - MCS Ratio: **~75%** per A, **~75%** per B

### Test 3: Grafi Completamente Diversi
- **File**: `test_completely_different_A.macm`, `test_completely_different_B.macm`
- **Descrizione**: Architettura Web vs IoT - nessuna sovrapposizione
- **Risultati attesi**:
  - Edit Distance: **Alta** (~14)
  - MCS Size: **0** (nessun nodo in comune)
  - MCS Ratio: **0%**
  - Supergraph Ratio: **Basso** (~25%)

### Test 4: Stessi Nodi, Relazioni Diverse
- **File**: `test_same_nodes_diff_edges_A.macm`, `test_same_nodes_diff_edges_B.macm`
- **Descrizione**: 3 nodi Device con topologia diversa (lineare vs stella)
- **Risultati attesi**:
  - MCS Size: **0** (signature diverse per pattern di connessioni)
  - Edit Distance: **~4** (relazioni diverse)
  - **Nota**: Dimostra che stessi nodi ≠ grafi simili se le connessioni sono diverse

### Test 5: Sottoinsieme (Grafo Piccolo vs Grande)
- **File**: `test_subset_small.macm`, `test_subset_large.macm`
- **Descrizione**: Il grafo piccolo è contenuto nel grafo grande
- **Risultati attesi**:
  - MCS Size: **2** (Server + Database)
  - MCS Ratio Small: **100%** (tutto il grafo piccolo è comune)
  - MCS Ratio Large: **~40%** (solo parte del grafo grande)
  - Supergraph = dimensione del grafo grande

### Test 6: Grafi Minimali
- **File**: `test_minimal_single_node.macm`, `test_minimal_two_nodes.macm`
- **Descrizione**: Caso edge: 1 nodo vs 2 nodi connessi
- **Risultati attesi**:
  - Edit Distance: **2** (1 nodo + 1 relazione)
  - MCS Size: **1** (il singolo nodo comune)

### Test 7: Alta Similarità (90%)
- **File**: `test_high_similarity_A.macm`, `test_high_similarity_B.macm`
- **Descrizione**: Grafi quasi identici, solo ordine degli edge diverso
- **Risultati attesi**:
  - Edit Distance: **0** (stessa struttura)
  - MCS Ratio: **100%**
  - Normalized Edit Distance: **0.0**

### Test 8: Media Similarità (50-60%)
- **File**: `test_medium_similarity_A.macm`, `test_medium_similarity_B.macm`
- **Descrizione**: Architettura simile ma componenti parzialmente diversi (LoadBalancer vs Firewall)
- **Risultati attesi**:
  - Edit Distance: **Media** (~6-8)
  - MCS Ratio: **~50-60%**
  - Normalized Edit Distance: **0.4-0.6**
  - Common Nodes: **~3** (Server, Database, Client)

### Test 9: Bassa Similarità (20-30%)
- **File**: `test_low_similarity_A.macm`, `test_low_similarity_B.macm`
- **Descrizione**: Solo Database in comune, topologie completamente diverse
- **Risultati attesi**:
  - Edit Distance: **Alta** (~10-12)
  - MCS Ratio: **~20-30%**
  - Normalized Edit Distance: **0.7-0.8**
  - Common Nodes: **1** (solo Database)

### Test 10: Grande Differenza di Dimensione
- **File**: `test_size_difference_small.macm`, `test_size_difference_large.macm`
- **Descrizione**: Grafo piccolo (3 nodi) vs grande (8 nodi), piccolo è sottoinsieme
- **Risultati attesi**:
  - MCS Ratio (Small): **~66-100%** (maggior parte del piccolo è comune)
  - MCS Ratio (Large): **~20-30%** (piccola parte del grande)
  - Supergraph Size: **~dimensione del grafo grande**

### Test 11: Overlap Parziale
- **File**: `test_partial_overlap_A.macm`, `test_partial_overlap_B.macm`
- **Descrizione**: Web tradizionale vs Microservizi - Database e Cache in comune
- **Risultati attesi**:
  - Edit Distance: **Media** (~8-10)
  - MCS Ratio: **~40-50%**
  - Common Nodes: **2** (Database, Cache)
  - Dimostra come architetture diverse possano condividere componenti

## 🚀 Come Eseguire i Test

### Opzione 1: Script automatico
```bash
python run_tests.py
```

### Opzione 2: Test singolo con main.py
```bash
python src/main.py data/test_macm/test_identical_A.macm data/test_macm/test_identical_B.macm
```

### Opzione 3: Usando il modulo direttamente
```python
from src.database_manager import Neo4jManager
from src.config import DatabaseConfig
from src.utils import load_graph_from_cypher_file
from src.metrics import GraphMetricsCalculator

db_config = DatabaseConfig()

with Neo4jManager(db_config) as neo4j_manager:
    graph1 = load_graph_from_cypher_file(neo4j_manager, "data/test_macm/test_identical_A.macm")
    graph2 = load_graph_from_cypher_file(neo4j_manager, "data/test_macm/test_identical_B.macm")

calculator = GraphMetricsCalculator()
result = calculator.compare_graphs(graph1, graph2)

print(f"Edit Distance: {result.edit_distance}")
print(f"MCS Ratio: {result.mcs_ratio_graph1}")
```

## 📊 Metriche Spiegate

### Edit Distance
Numero di operazioni (aggiungi, rimuovi, sostituisci) per trasformare un grafo nell'altro.
- **0** = Grafi identici
- **Bassa** = Grafi simili
- **Alta** = Grafi molto diversi

### Maximum Common Subgraph (MCS)
Il sottografo più grande presente in entrambi i grafi.
- **MCS Size** = Numero di elementi comuni
- **MCS Ratio** = Percentuale del grafo che è comune

### Minimum Common Supergraph
Il grafo più piccolo che contiene entrambi i grafi.
- **Supergraph Ratio** = Quanto ciascun grafo occupa del supergraph
- **Alto** = Grafi compatti e simili
- **Basso** = Grafi con molta distanza strutturale

## 🎯 Obiettivi dei Test

1. **Validazione**: Verificare che le metriche funzionino correttamente
2. **Comprensione**: Capire come le metriche reagiscono a diversi scenari
3. **Debug**: Identificare edge cases e comportamenti inattesi
4. **Documentazione**: Fornire esempi chiari di utilizzo

## 📝 Note Importanti

- I nodi sono considerati **equivalenti** se hanno stesso `type` e stessa signature di relazioni
- I **nomi** dei nodi possono essere diversi (es: "Server" vs "WebServer")
- Le **relazioni** devono avere stesso tipo e connettere nodi equivalenti
- La **topologia** è importante: stessi nodi con connessioni diverse = grafi diversi
