# Formule delle Metriche per Confronto Grafi

## Panoramica Generale
Il sistema calcola 7 metriche principali per confrontare due grafi G1 e G2. Ogni metrica cattura un aspetto diverso della somiglianza strutturale.

## 1. Edit Distance (Distanza di Modifica)

### Formula Base:
```
edit_distance = node_insertions + node_deletions + edge_insertions + edge_deletions
```

### Come viene calcolata:
1. **Signature Canonica dei Nodi**: `(type, sorted_multiset_local_relations)`
   - `type` = proprietà "type" del nodo (es. "HW.Device")
   - `local_relations` = Counter delle relazioni per tipo e direzione: `[("hosts","in"), ("connects","out")]`

2. **Signature Canonica delle Relazioni**: `(start_node_sig, rel_type, [protocol], end_node_sig)`

3. **Calcolo Operazioni**:
   - `node_insertions = |signatures_G2 - signatures_G1|`
   - `node_deletions = |signatures_G1 - signatures_G2|` 
   - `edge_insertions = |edge_sigs_G2 - edge_sigs_G1|`
   - `edge_deletions = |edge_sigs_G1 - edge_sigs_G2|`

### Normalizzazione:
```
normalized_edit_distance = edit_distance / max(total_elements_G1, total_elements_G2)
normalized_edit_distance = clamp(normalized_edit_distance, 0, 1)
```
Dove `total_elements = |nodes| + |relationships|`

**Range**: [0, 1] dove 0 = identici, 1 = completamente diversi

---

## 2. Maximum Common Subgraph (MCS)

### Formula:
```
mcs_size = |common_nodes| + |common_relationships|
mcs_ratio_G1 = mcs_size / |nodes_G1|
mcs_ratio_G2 = mcs_size / |nodes_G2|
```

### Come vengono trovati i nodi comuni:
- Due nodi sono equivalenti se hanno:
  - Stesso `type` property
  - Stessa signature delle relazioni locali (multiset di (rel_type, direction))

### Come vengono trovate le relazioni comuni:
- Due relazioni sono equivalenti se:
  - Stesso `relationship_type`
  - Endpoint equivalenti (nodi mappati)
  - Stesso `protocol` (se presente)

**Range**: [0, 1] dove 1 = tutti i nodi sono comuni

---

## 3. Minimum Common Supergraph

### Formula:
```
supergraph_size = (|nodes_G1| + |nodes_G2| - |common_nodes|) + 
                  (|relationships_G1| + |relationships_G2| - |common_relationships|)

supergraph_ratio_G1 = supergraph_size / |total_elements_G1|
supergraph_ratio_G2 = supergraph_size / |total_elements_G2|
```

**Nota**: Nel codice attuale, `supergraph_ratio` è sostituito da `structural_jaccard`

---

## 4. Structural Jaccard Similarity

### Formula:
```
canonical_signatures_G1 = {(type, local_relations_multiset) for each node in G1}
canonical_signatures_G2 = {(type, local_relations_multiset) for each node in G2}

intersection = |canonical_signatures_G1 ∩ canonical_signatures_G2|
union = |canonical_signatures_G1 ∪ canonical_signatures_G2|

structural_jaccard = intersection / union (se union > 0, altrimenti 0)
```

**Range**: [0, 1] dove 1 = signature identiche, 0 = signature completamente diverse

---

## 5. Degree Distribution Similarity

### Formula (Cosine Similarity):
```
all_degrees = degree_keys_G1 ∪ degree_keys_G2
vector_G1 = [degree_count_G1.get(d, 0) for d in all_degrees]
vector_G2 = [degree_count_G2.get(d, 0) for d in all_degrees]

cosine_similarity = (vector_G1 · vector_G2) / (||vector_G1|| × ||vector_G2||)
```

Dove `degree_count_Gi[d]` = numero di nodi con grado `d` nel grafo `Gi`

**Range**: [0, 1] dove 1 = distribuzione identica

---

## 6. Label Distribution Similarity

### Formula (Cosine Similarity):
```
all_labels = label_keys_G1 ∪ label_keys_G2  
vector_G1 = [label_count_G1.get(l, 0) for l in all_labels]
vector_G2 = [label_count_G2.get(l, 0) for l in all_labels]

cosine_similarity = (vector_G1 · vector_G2) / (||vector_G1|| × ||vector_G2||)
```

Dove `label_count_Gi[l]` = numero di nodi con etichetta `l` nel grafo `Gi`

**Range**: [0, 1] dove 1 = distribuzione identica

---

## 7. Overall Similarity Score

### Formula Pesata:
```
weights = {
    'edit_distance': 0.3,
    'mcs_ratio': 0.3, 
    'degree_similarity': 0.2,
    'label_similarity': 0.2
}

edit_similarity = 1.0 - normalized_edit_distance
mcs_similarity = (mcs_ratio_G1 + mcs_ratio_G2) / 2.0

similarity_score = (
    weights['edit_distance'] × edit_similarity +
    weights['mcs_ratio'] × mcs_similarity +
    weights['degree_similarity'] × degree_distribution_similarity +
    weights['label_similarity'] × label_distribution_similarity
)

similarity_score = clamp(similarity_score, 0, 1)
```

**Range**: [0, 1] dove 1 = grafi identici

---

## Differenze Chiave tra le Metriche

### 1. **Edit Distance vs MCS**:
- **Edit Distance**: conta operazioni necessarie per trasformare G1 in G2
- **MCS**: conta elementi comuni già esistenti
- Sono **complementari**: bassa edit distance ⟺ alto MCS

### 2. **Structural Jaccard vs MCS**:
- **Structural Jaccard**: usa signature canoniche (tipo + relazioni locali)
- **MCS**: usa equivalenza nodo-per-nodo più rigorosa
- **Jaccard**: più tollerante a cambiamenti di id/nomi se struttura locale è simile

### 3. **Distribution Similarities**:
- **Degree Distribution**: confronta solo la topologia (numero connessioni)
- **Label Distribution**: confronta solo i tipi/etichette dei nodi
- Sono **invarianti isomorfici**: grafi isomorfi hanno stesse distribuzioni

### 4. **Normalizzazioni**:
- **Edit Distance**: normalizzato su dimensione massima dei grafi
- **MCS Ratio**: normalizzato su dimensione di ciascun grafo
- **Jaccard**: intrinsecamente normalizzato [0,1]
- **Distributions**: cosine similarity già in [0,1]

---

## Esempio Pratico (JetRacer)

Dal tuo JSON:
- `edit_distance: 4.0` → 4 operazioni canoniche
- `normalized_edit_distance: 0.129` → 4/(16+15) = 0.129
- `mcs_ratio: 0.9375` → 15/16 = 0.9375 (93.75% nodi comuni)
- `structural_jaccard: 0.882` → 88.2% signature canoniche comuni
- `similarity_score: 0.940` → 94% similarità pesata complessiva

L'alta similarità deriva da:
- Poche operazioni di edit (4/31 = 13%)
- Alto MCS (93.75% nodi comuni)  
- Distribuzioni quasi identiche (99.8% degree, 99.1% label)