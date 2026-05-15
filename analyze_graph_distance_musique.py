"""Graph distance analysis on MuSiQue 1000-sample (PID 34587).

Compute shortest-path distance in the entity graph from:
  (a) query entities → gold passage nodes
  (b) bridge entities (LLM-discovered) → gold passage nodes

Compare by hop count (2/3/4) to test the hypothesis:
  - For deep-hop questions, query entities are far from gold in graph
  - Bridge entities are closer (LLM identifies relay nodes)
  - Multi-source PPR with bridges achieves better coverage

Also compute stability metrics: variance of bridge distances vs query distances.
"""
import json
import pickle
import numpy as np
import sys
import re
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '.')
from evaluate_musique_ner_pipeline import NERIndex  # for unpickling

PKL = 'outputs/musique_ner_pipeline_eval/global_ner_index.pkl'
RESULTS = 'outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260402_235319_34587.json'
MUSIQUE = 'musique.json'
OUT = 'outputs/musique_graph_distance_analysis.json'


def find_entity_vid(index, entity_str):
    """Find vid for an entity string (case-insensitive)."""
    e_lower = entity_str.lower().strip()
    for v in index.graph.vs:
        c = v.attributes().get('content', '').lower()
        if c == e_lower:
            return v.index
    return None


def find_passage_vid(index, paragraph_text):
    """Find vid for a passage by exact match (after dedupe)."""
    # passage_keys are mdhash IDs of passages; passages dict maps id → text
    # passage_node_idxs maps key index → vid
    for i, key in enumerate(index.passage_keys):
        if index.passages[key] == paragraph_text:
            return index.passage_node_idxs[i]
    return None


def main():
    print("Loading NER index (631 MB)...")
    with open(PKL, 'rb') as f:
        index = pickle.load(f)
    print(f"  Graph: {index.graph.vcount()} nodes, {index.graph.ecount()} edges")

    print("Loading results + musique...")
    results_data = json.load(open(RESULTS))
    results = results_data['results']
    musique = json.load(open(MUSIQUE))
    q_to_sample = {x['question']: x for x in musique}
    print(f"  Results: {len(results)} samples")

    # Build entity vid lookup (case-insensitive)
    print("Building entity vid lookup...")
    entity_to_vid = {}
    passage_vid_set = set(index.passage_node_idxs)
    for v in index.graph.vs:
        c = v.attributes().get('content', '')
        if c and v.index not in passage_vid_set:
            entity_to_vid[c.lower()] = v.index
    print(f"  {len(entity_to_vid)} entity vids indexed")

    # Build passage_text → vid lookup
    print("Building passage vid lookup...")
    passage_text_to_vid = {}
    for i, key in enumerate(index.passage_keys):
        passage_text_to_vid[index.passages[key]] = index.passage_node_idxs[i]
    print(f"  {len(passage_text_to_vid)} passage vids indexed")

    # Process each sample
    per_sample_data = []  # list of {hop, q_dists, b_dists, ...}

    print("Computing distances...")
    for i, r in enumerate(results):
        q = r['question']
        ms = q_to_sample.get(q)
        if ms is None:
            continue
        hop = len(ms.get('question_decomposition', []))
        if hop not in (2, 3, 4):
            continue

        # Query entities: substring match in question
        q_lower = q.lower()
        query_entities = set()
        for ent_lower, vid in entity_to_vid.items():
            if len(ent_lower) >= 4 and ent_lower in q_lower:
                query_entities.add(vid)

        # Bridge entities: union from all rounds
        rounds = r.get('round_diagnostics', [])
        bridge_entities = set()
        for rd in rounds:
            for ent in rd.get('discovered_entities_total', []):
                vid = entity_to_vid.get(ent.lower())
                if vid is not None:
                    bridge_entities.add(vid)
        bridge_entities -= query_entities  # only count bridges NOT in query

        # Gold passage vids
        gold_passages = set()
        for para in ms['paragraphs']:
            if para.get('is_supporting'):
                pt = para.get('paragraph_text', '')
                vid = passage_text_to_vid.get(pt)
                if vid is not None:
                    gold_passages.add(vid)

        if not query_entities or not bridge_entities or not gold_passages:
            continue

        # Compute shortest paths
        # graph.shortest_paths returns 2D matrix [source][target]
        try:
            q_dists_matrix = index.graph.shortest_paths(
                source=list(query_entities),
                target=list(gold_passages)
            )
            b_dists_matrix = index.graph.shortest_paths(
                source=list(bridge_entities),
                target=list(gold_passages)
            )
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue

        # Convert to flat lists, filter inf
        q_dists = [d for row in q_dists_matrix for d in row if d != float('inf') and d > 0]
        b_dists = [d for row in b_dists_matrix for d in row if d != float('inf') and d > 0]

        if not q_dists or not b_dists:
            continue

        per_sample_data.append({
            'idx': i,
            'hop': hop,
            'n_query_entities': len(query_entities),
            'n_bridges': len(bridge_entities),
            'n_gold': len(gold_passages),
            'query_dist_mean': float(np.mean(q_dists)),
            'query_dist_median': float(np.median(q_dists)),
            'query_dist_min': float(np.min(q_dists)),
            'bridge_dist_mean': float(np.mean(b_dists)),
            'bridge_dist_median': float(np.median(b_dists)),
            'bridge_dist_min': float(np.min(b_dists)),
            'query_dist_std': float(np.std(q_dists)),
            'bridge_dist_std': float(np.std(b_dists)),
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(results)} done")

    print(f"\n=== Aggregated by hop ===")
    by_hop = defaultdict(list)
    for d in per_sample_data:
        by_hop[d['hop']].append(d)

    summary = {}
    for hop in sorted(by_hop):
        rows = by_hop[hop]
        n = len(rows)
        # Mean of per-sample mean distances
        q_mean = np.mean([r['query_dist_mean'] for r in rows])
        b_mean = np.mean([r['bridge_dist_mean'] for r in rows])
        q_median = np.mean([r['query_dist_median'] for r in rows])
        b_median = np.mean([r['bridge_dist_median'] for r in rows])
        q_min = np.mean([r['query_dist_min'] for r in rows])
        b_min = np.mean([r['bridge_dist_min'] for r in rows])
        # Stability (std of distances WITHIN sample, averaged across samples)
        q_within_std = np.mean([r['query_dist_std'] for r in rows])
        b_within_std = np.mean([r['bridge_dist_std'] for r in rows])
        # Cross-sample variance of mean distance
        q_across_var = np.var([r['query_dist_mean'] for r in rows])
        b_across_var = np.var([r['bridge_dist_mean'] for r in rows])

        summary[hop] = {
            'n_samples': n,
            'query_dist': {'mean_of_means': q_mean, 'mean_of_medians': q_median, 'mean_of_mins': q_min,
                           'within_sample_std_avg': q_within_std, 'across_sample_var': q_across_var},
            'bridge_dist': {'mean_of_means': b_mean, 'mean_of_medians': b_median, 'mean_of_mins': b_min,
                            'within_sample_std_avg': b_within_std, 'across_sample_var': b_across_var},
            'delta': {'mean': q_mean - b_mean, 'median': q_median - b_median, 'min': q_min - b_min},
        }
        print(f"\nHop={hop} (n={n}):")
        print(f"  Query entity → gold: mean dist = {q_mean:.2f}, median = {q_median:.2f}, min = {q_min:.2f}")
        print(f"  Bridge entity → gold: mean dist = {b_mean:.2f}, median = {b_median:.2f}, min = {b_min:.2f}")
        print(f"  Δ (q - b):           mean = {q_mean - b_mean:+.2f}, median = {q_median - b_median:+.2f}")
        print(f"  Within-sample std:   query = {q_within_std:.2f}, bridge = {b_within_std:.2f}")
        print(f"  Across-sample var:   query = {q_across_var:.2f}, bridge = {b_across_var:.2f}")

    # Save
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'summary': summary, 'per_sample': per_sample_data}, f, indent=2)
    print(f"\nSaved to {OUT}")
    print(f"Total analyzed: {len(per_sample_data)}/{len(results)}")


if __name__ == "__main__":
    main()
