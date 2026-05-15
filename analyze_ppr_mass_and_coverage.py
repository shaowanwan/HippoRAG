"""Direct mechanism analysis on MuSiQue 1000-sample:

(1) PPR mass on gold passages: query-only vs query+bridge seeds
(2) Coverage analysis: % of gold passages reachable within k hops
"""
import json
import pickle
import numpy as np
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '.')
from evaluate_musique_ner_pipeline import NERIndex

PKL = 'outputs/musique_ner_pipeline_eval/global_ner_index.pkl'
RESULTS = 'outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260402_235319_34587.json'
MUSIQUE = 'musique.json'
OUT = 'outputs/musique_ppr_mass_coverage_analysis.json'
SAMPLE_LIMIT = 200


def main():
    print("Loading NER index (631 MB)...")
    t0 = time.time()
    with open(PKL, 'rb') as f:
        index = pickle.load(f)
    print(f"  Graph: {index.graph.vcount()} nodes, {index.graph.ecount()} edges ({time.time()-t0:.0f}s)")

    results = json.load(open(RESULTS))['results']
    musique = json.load(open(MUSIQUE))
    q_to_sample = {x['question']: x for x in musique}

    print("Building lookups...")
    passage_vid_set = set(index.passage_node_idxs)
    entity_to_vid = {}
    for v in index.graph.vs:
        c = v.attributes().get('content', '')
        if c and v.index not in passage_vid_set:
            entity_to_vid[c.lower()] = v.index

    passage_text_to_vid = {index.passages[k]: index.passage_node_idxs[i]
                           for i, k in enumerate(index.passage_keys)}
    g = index.graph
    n_nodes = g.vcount()

    per_sample = []
    for i, r in enumerate(results[:SAMPLE_LIMIT]):
        q = r['question']
        ms = q_to_sample.get(q)
        if ms is None: continue
        hop = len(ms.get('question_decomposition', []))
        if hop not in (2, 3, 4): continue

        q_lower = q.lower()
        query_vids = set()
        for ent_lower, vid in entity_to_vid.items():
            if len(ent_lower) >= 4 and ent_lower in q_lower:
                query_vids.add(vid)

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
                if vid is not None:
                    gold_vids.add(vid)

        if not query_vids or not bridge_vids or not gold_vids: continue

        def ppr_mass(seeds):
            reset = np.zeros(n_nodes)
            for s in seeds: reset[s] = 1.0
            reset /= reset.sum()
            pr = g.personalized_pagerank(damping=0.85, reset=reset.tolist(), implementation="prpack")
            return sum(pr[v] for v in gold_vids)

        try:
            q_mass = ppr_mass(query_vids)
            qb_mass = ppr_mass(query_vids | bridge_vids)
        except Exception as e:
            print(f"  Sample {i} PPR failed: {e}")
            continue

        def coverage(seeds, ks=(1, 2, 3, 4, 5)):
            d_matrix = g.distances(source=list(seeds), target=list(gold_vids))
            min_dists = [min(col) if min(col) != float('inf') else 99 for col in zip(*d_matrix)]
            return {k: sum(1 for d in min_dists if d <= k) / len(min_dists) for k in ks}

        q_cov = coverage(query_vids)
        qb_cov = coverage(query_vids | bridge_vids)

        per_sample.append({
            'idx': i, 'hop': hop,
            'n_query': len(query_vids), 'n_bridge': len(bridge_vids), 'n_gold': len(gold_vids),
            'q_mass': q_mass, 'qb_mass': qb_mass,
            'mass_ratio': qb_mass / q_mass if q_mass > 0 else None,
            'q_cov': q_cov, 'qb_cov': qb_cov,
        })

        if len(per_sample) % 50 == 0:
            print(f"  {len(per_sample)} done ({time.time()-t0:.0f}s)")

    print(f"\n=== Aggregated (n={len(per_sample)}) ===\n")
    by_hop = defaultdict(list)
    for d in per_sample: by_hop[d['hop']].append(d)

    summary = {}
    print("PPR MASS ON GOLD PASSAGES")
    print(f"{'Hop':<6}{'n':<6}{'Q-only':<15}{'Q+B':<15}{'ratio':<10}")
    for hop in sorted(by_hop):
        rows = by_hop[hop]
        n = len(rows)
        q_m = np.mean([r['q_mass'] for r in rows])
        qb_m = np.mean([r['qb_mass'] for r in rows])
        ratio = np.mean([r['mass_ratio'] for r in rows if r['mass_ratio']])
        print(f"{hop:<6}{n:<6}{q_m:<15.5f}{qb_m:<15.5f}{ratio:<10.2f}")
        summary[hop] = {'n': n, 'q_mass': q_m, 'qb_mass': qb_m, 'mass_ratio': ratio}

    print("\nCOVERAGE @ k hops")
    print(f"{'Hop':<6}{'n':<6}{'Q@1':<8}{'Q+B@1':<8}{'Q@2':<8}{'Q+B@2':<8}{'Q@3':<8}{'Q+B@3':<8}{'Q@4':<8}{'Q+B@4':<8}{'Q@5':<8}{'Q+B@5':<8}")
    for hop in sorted(by_hop):
        rows = by_hop[hop]
        n = len(rows)
        line = f"{hop:<6}{n:<6}"
        cov_data = {}
        for k in [1, 2, 3, 4, 5]:
            qc = np.mean([r['q_cov'][k] for r in rows])
            qbc = np.mean([r['qb_cov'][k] for r in rows])
            line += f"{qc:<8.2f}{qbc:<8.2f}"
            cov_data[f'k{k}_q'] = qc
            cov_data[f'k{k}_qb'] = qbc
        print(line)
        summary[hop]['coverage'] = cov_data

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'summary': summary, 'per_sample': per_sample, 'sample_limit': SAMPLE_LIMIT}, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
