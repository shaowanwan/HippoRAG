"""PPR mass on gold passages: query-only vs query+bridge seeds (MuSiQue 1000).

Direct mechanism measurement: when we add LLM-discovered bridge entities as
additional PPR personalization seeds, how much more probability mass lands
on gold passages?

Run on cached NER index (PID 34587 era) + result data. Aggregated by hop count.
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
OUT = 'outputs/musique_ppr_mass_on_gold.json'
SAMPLE_LIMIT = 200
PPR_DAMPING = 0.85


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
    n_nodes = g.vcount()

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

        def ppr_mass(seeds):
            reset = np.zeros(n_nodes)
            for s in seeds: reset[s] = 1.0
            reset /= reset.sum()
            pr = g.personalized_pagerank(damping=PPR_DAMPING, reset=reset.tolist(), implementation="prpack")
            return sum(pr[v] for v in gold_vids)

        try:
            q_mass = ppr_mass(query_vids)
            qb_mass = ppr_mass(query_vids | bridge_vids)
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue

        per_sample.append({
            'idx': i, 'hop': hop,
            'n_query': len(query_vids), 'n_bridge': len(bridge_vids), 'n_gold': len(gold_vids),
            'q_mass': q_mass, 'qb_mass': qb_mass,
            'mass_ratio': qb_mass / q_mass if q_mass > 0 else None,
            'mass_delta': qb_mass - q_mass,
        })
        if len(per_sample) % 50 == 0:
            print(f"  {len(per_sample)} done ({time.time()-t0:.0f}s)")

    print(f"\n=== PPR MASS ON GOLD (damping={PPR_DAMPING}, n={len(per_sample)}) ===\n")
    by_hop = defaultdict(list)
    for d in per_sample: by_hop[d['hop']].append(d)
    summary = {}
    print(f"{'Hop':<6}{'n':<6}{'Q-only':<15}{'Q+B':<15}{'Δ mass':<15}{'mean ratio':<12}")
    for hop in sorted(by_hop):
        rows = by_hop[hop]
        n = len(rows)
        q_m = np.mean([r['q_mass'] for r in rows])
        qb_m = np.mean([r['qb_mass'] for r in rows])
        delta = np.mean([r['mass_delta'] for r in rows])
        ratio = np.mean([r['mass_ratio'] for r in rows if r['mass_ratio']])
        print(f"{hop:<6}{n:<6}{q_m:<15.5f}{qb_m:<15.5f}{delta:<15.5f}{ratio:<12.2f}")
        summary[hop] = {'n': n, 'q_mass_mean': q_m, 'qb_mass_mean': qb_m,
                        'mass_delta_mean': delta, 'mass_ratio_mean': ratio}

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'summary': summary, 'per_sample': per_sample,
                   'sample_limit': SAMPLE_LIMIT, 'ppr_damping': PPR_DAMPING}, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
