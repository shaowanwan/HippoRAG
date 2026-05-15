"""Coverage analysis: % of gold passages reachable within k hops of seeds (MuSiQue 1000).

For each sample, measure how many gold passages are within k graph hops of:
  (a) query-entity seeds only
  (b) query + bridge entity seeds

This quantifies the "graph-side reach" of multi-source PPR. Since PPR mass
concentrates near seeds with geometric decay, gold passages within 1-2 hops
of seeds receive substantially more retrieval probability.
"""
import json
import pickle
import numpy as np
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '.')
from evaluate_musique_ner_pipeline import NERIndex  # unpickling

PKL = 'outputs/musique_ner_pipeline_eval/global_ner_index.pkl'
RESULTS = 'outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260402_235319_34587.json'
MUSIQUE = 'musique.json'
OUT = 'outputs/musique_gold_coverage_at_k_hops.json'
SAMPLE_LIMIT = 200
K_VALUES = (1, 2, 3, 4, 5)


def main():
    print("Loading NER index (631 MB)...")
    t0 = time.time()
    with open(PKL, 'rb') as f:
        index = pickle.load(f)
    print(f"  Graph: {index.graph.vcount()} nodes, {index.graph.ecount()} edges ({time.time()-t0:.0f}s)")

    results = json.load(open(RESULTS))['results']
    musique = json.load(open(MUSIQUE))
    q_to_sample = {x['question']: x for x in musique}

    passage_vid_set = set(index.passage_node_idxs)
    entity_to_vid = {}
    for v in index.graph.vs:
        c = v.attributes().get('content', '')
        if c and v.index not in passage_vid_set:
            entity_to_vid[c.lower()] = v.index
    passage_text_to_vid = {index.passages[k]: index.passage_node_idxs[i]
                           for i, k in enumerate(index.passage_keys)}
    g = index.graph

    per_sample = []
    for i, r in enumerate(results[:SAMPLE_LIMIT]):
        q = r['question']
        ms = q_to_sample.get(q)
        if ms is None: continue
        hop = len(ms.get('question_decomposition', []))
        if hop not in (2, 3, 4): continue

        q_lower = q.lower()
        query_vids = {vid for ent_lower, vid in entity_to_vid.items()
                      if len(ent_lower) >= 4 and ent_lower in q_lower}

        rounds = r.get('round_diagnostics', [])
        bridge_vids = set()
        for rd in rounds:
            for ent in rd.get('discovered_entities_total', []):
                vid = entity_to_vid.get(ent.lower())
                if vid is not None and vid not in query_vids:
                    bridge_vids.add(vid)

        gold_vids = set()
        for para in ms['paragraphs']:
            if para.get('is_supporting'):
                vid = passage_text_to_vid.get(para.get('paragraph_text', ''))
                if vid is not None: gold_vids.add(vid)

        if not query_vids or not bridge_vids or not gold_vids: continue

        def coverage(seeds):
            d_matrix = g.distances(source=list(seeds), target=list(gold_vids))
            min_dists = [min(col) if min(col) != float('inf') else 99 for col in zip(*d_matrix)]
            return {k: sum(1 for d in min_dists if d <= k) / len(min_dists) for k in K_VALUES}

        q_cov = coverage(query_vids)
        qb_cov = coverage(query_vids | bridge_vids)

        per_sample.append({
            'idx': i, 'hop': hop,
            'n_query': len(query_vids), 'n_bridge': len(bridge_vids), 'n_gold': len(gold_vids),
            'q_cov': q_cov, 'qb_cov': qb_cov,
            'cov_delta': {k: qb_cov[k] - q_cov[k] for k in K_VALUES},
        })
        if len(per_sample) % 50 == 0:
            print(f"  {len(per_sample)} done ({time.time()-t0:.0f}s)")

    print(f"\n=== COVERAGE @ k HOPS (n={len(per_sample)}) ===\n")
    by_hop = defaultdict(list)
    for d in per_sample: by_hop[d['hop']].append(d)
    summary = {}
    header = f"{'Hop':<6}{'n':<6}" + "".join(f"Q@{k}    Q+B@{k}  " for k in K_VALUES)
    print(header)
    for hop in sorted(by_hop):
        rows = by_hop[hop]
        n = len(rows)
        line = f"{hop:<6}{n:<6}"
        cov_data = {}
        for k in K_VALUES:
            qc = np.mean([r['q_cov'][k] for r in rows])
            qbc = np.mean([r['qb_cov'][k] for r in rows])
            line += f"{qc:.2f}    {qbc:.2f}    "
            cov_data[f'k{k}_q'] = qc
            cov_data[f'k{k}_qb'] = qbc
            cov_data[f'k{k}_delta'] = qbc - qc
        print(line)
        summary[hop] = {'n': n, **cov_data}

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'summary': summary, 'per_sample': per_sample,
                   'sample_limit': SAMPLE_LIMIT, 'k_values': list(K_VALUES)}, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
