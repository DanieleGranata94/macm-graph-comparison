"""
Generate a standard JSON diff between two Graph objects.

Exports:
- generate_graph_diff(graph1, graph2, comparison_result, output_path='output/graph_diff.json')

The JSON schema (compact):
{
  "summary": { ... metrics ... },
  "node_mappings": { "g1_id": "g2_id", ... },
  "added_nodes": [ {id, labels, properties} ],
  "removed_nodes": [ {id, labels, properties} ],
  "modified_nodes": [ { id_graph1, id_graph2, diffs: {added, removed, changed}} ],
  "added_relationships": [ {id, start, end, type, properties} ],
  "removed_relationships": [ ... ],
  "modified_relationships": [ { id_graph1, id_graph2, diffs } ]
}

This file is intentionally conservative and depends on the equivalence
checks implemented in `metrics.GraphMetricsCalculator`.
"""

import os
import json
from typing import Dict, Any, List
from collections import defaultdict

from models import Graph, GraphNode, GraphRelationship
from metrics import GraphMetricsCalculator


def _property_diffs(p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict describing added/removed/changed properties between two dicts."""
    added = {}
    removed = {}
    changed = {}

    keys1 = set(p1.keys())
    keys2 = set(p2.keys())

    for k in keys2 - keys1:
        added[k] = p2[k]
    for k in keys1 - keys2:
        removed[k] = p1[k]
    for k in keys1 & keys2:
        if p1.get(k) != p2.get(k):
            changed[k] = {"from": p1.get(k), "to": p2.get(k)}

    return {"added": added, "removed": removed, "changed": changed}


def _node_to_dict(node: GraphNode) -> Dict[str, Any]:
    return {
        "id": node.id,
        "labels": list(node.labels) if node.labels is not None else [],
        "properties": dict(node.properties) if node.properties is not None else {}
    }


def _rel_to_dict(rel: GraphRelationship) -> Dict[str, Any]:
    return {
        "id": rel.id,
        "start_node_id": rel.start_node_id,
        "end_node_id": rel.end_node_id,
        "relationship_type": rel.relationship_type,
        "properties": dict(rel.properties) if rel.properties is not None else {}
    }


def generate_graph_diff(graph1: Graph, graph2: Graph, comparison_result, output_path: str = "output/graph_diff.json") -> str:
    """Generate a JSON diff between graph1 and graph2 and write it to output_path.

    Returns the absolute path to the written file.
    """
    calc = GraphMetricsCalculator()

    # Node mappings: for each node in graph1 try to find equivalent in graph2
    mapping_1_to_2 = {}
    mapping_2_to_1 = {}

    for n1 in graph1.nodes.values():
        found = None
        for n2 in graph2.nodes.values():
            if calc._nodes_equivalent(n1, n2):
                found = n2
                break
        if found:
            mapping_1_to_2[n1.id] = found.id
            mapping_2_to_1[found.id] = n1.id

    # Nodes classification
    added_nodes = []
    removed_nodes = []
    modified_nodes = []

    # Nodes in graph1 not mapped => removed
    for n1 in graph1.nodes.values():
        if n1.id not in mapping_1_to_2:
            d = _node_to_dict(n1)
            d["origin"] = "g1"
            removed_nodes.append(d)
        else:
            # check properties differences
            n2 = graph2.nodes.get(mapping_1_to_2[n1.id])
            diffs = _property_diffs(n1.properties or {}, n2.properties or {})
            if diffs["added"] or diffs["removed"] or diffs["changed"]:
                modified_nodes.append({
                    "id_graph1": n1.id,
                    "id_graph2": n2.id,
                    "diffs": diffs
                })

    # Nodes in graph2 not mapped => added
    for n2 in graph2.nodes.values():
        if n2.id not in mapping_2_to_1:
            d = _node_to_dict(n2)
            d["origin"] = "g2"
            added_nodes.append(d)

    # Relationships classification using node mapping
    added_relationships = []
    removed_relationships = []
    modified_relationships = []

    # Build an index for relationships in graph2 by (start,end,type) to speed lookup
    rel_index_g2 = defaultdict(list)
    for r2 in graph2.relationships.values():
        key = (r2.start_node_id, r2.end_node_id, r2.relationship_type)
        rel_index_g2[key].append(r2)

    mapped_rels_in_g2 = set()

    for r1 in graph1.relationships.values():
        start_mapped = mapping_1_to_2.get(r1.start_node_id)
        end_mapped = mapping_1_to_2.get(r1.end_node_id)
        matched = None
        if start_mapped and end_mapped:
            candidates = rel_index_g2.get((start_mapped, end_mapped, r1.relationship_type), [])
            for cand in candidates:
                if calc._relationships_equivalent(r1, cand):
                    matched = cand
                    break
        if matched:
            mapped_rels_in_g2.add(matched.id)
            # check property diffs
            diffs = _property_diffs(r1.properties or {}, matched.properties or {})
            if diffs["added"] or diffs["removed"] or diffs["changed"]:
                modified_relationships.append({
                    "id_graph1": r1.id,
                    "id_graph2": matched.id,
                    "diffs": diffs
                })
        else:
            d = _rel_to_dict(r1)
            d["origin"] = "g1"
            removed_relationships.append(d)

    # Relationships in graph2 not mapped => added
    for r2 in graph2.relationships.values():
        if r2.id not in mapped_rels_in_g2:
            d = _rel_to_dict(r2)
            d["origin"] = "g2"
            added_relationships.append(d)

    # Build summary metrics with detailed edit operations
    summary = {
        "similarity_score": float(comparison_result.get_similarity_score()),
        "mcs_ratio_graph1": float(getattr(comparison_result, 'mcs_ratio_graph1', 0.0)),
        "mcs_ratio_graph2": float(getattr(comparison_result, 'mcs_ratio_graph2', 0.0)),
        "edit_distance": float(getattr(comparison_result, 'edit_distance', 0.0)),
        "normalized_edit_distance": float(getattr(comparison_result, 'normalized_edit_distance', 0.0)),
        "structural_jaccard": float(getattr(comparison_result, 'structural_jaccard', getattr(comparison_result, 'supergraph_ratio_graph1', 0.0))),
        "common_nodes_count": len(getattr(comparison_result, 'common_nodes', [])),
        "common_relationships_count": len(getattr(comparison_result, 'common_relationships', []))
    }
    
    # Add detailed edit operations if available
    if hasattr(comparison_result, 'edit_operations_detail'):
        summary["edit_operations_detail"] = comparison_result.edit_operations_detail

    diff_obj = {
        "summary": summary,
        "node_mappings": mapping_1_to_2,
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "modified_nodes": modified_nodes,
        "added_relationships": added_relationships,
        "removed_relationships": removed_relationships,
        "modified_relationships": modified_relationships
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path) or "output"
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write JSON to file
    with open(output_path, 'w', encoding='utf-8') as fh:
        json.dump(diff_obj, fh, indent=2, ensure_ascii=False, default=str)

    return os.path.abspath(output_path)
