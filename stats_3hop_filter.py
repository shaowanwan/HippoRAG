"""Stats for 3-hop queries: top-3 NN, sim > 0.8, vote >= 2 filter.

For each 3-hop query in 1000:
  - find top-3 same-hop NN (excluding self)
  - keep only NN with sim > 0.8 → "qualified" neighbors
  - bridges = union of qualified NN bridges, with vote count
  - filter to bridges appearing in ≥ 2 qualified NN (cap by vote)

Report:
  - how many 3-hop queries have ≥ 1 / 2 / 3 qualified NN
  - average bridges after vote filter
  - recall (own ∩ filtered) / |own|
  - whether the filter actually helps (vs raw top-3 union)
"""
import json
import os
import sys

os.environ['OPENAI_API_KEY'] = 'sk-396199ed7af84eff8a0cf7a71b797601'
sys.path.insert(0, 'src')


def main():
    import numpy as np
    from collections import Counter
    from src.hipporag.embedding_model import _get_embedding_model_class

    d = json.load(open('outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260421_143341_41732.json'))
    data = json.load(open('musique.json'))
    recs = []
    for r in d['results']:
        idx = r['idx']
        rds = r.get('round_diagnostics') or []
        bridges = rds[-1].get('discovered_entities_total', []) if rds else []
        try:
            hop = int(data[idx]['id'].split('hop')[0])
        except:
            hop = 0
        recs.append({'idx': idx, 'q': r['question'], 'hop': hop, 'bridges': bridges})
    print(f"Loaded {len(recs)} records")

    emb_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    em = _get_embedding_model_class(embedding_model_name=emb_name)(embedding_model_name=emb_name)
    embs = em.batch_encode([r['q'] for r in recs], norm=True)

    SIM_TH = 0.8
    K = 3
    VOTE_TH = 2  # bridge must appear in >= 2 qualified NN

    # 3-hop only, with non-empty bridges
    pool = [i for i in range(1000) if recs[i]['hop'] == 3]
    with_bridges = [i for i in pool if len(recs[i]['bridges']) > 0]
    print(f"\n=== 3-hop pool: {len(pool)} queries  ({len(with_bridges)} with bridges) ===")

    # Per-query analysis
    n_with_any_qualified = 0
    n_with_ge_vote_th = 0   # has at least 1 bridge surviving vote >= 2
    qualified_count_dist = Counter()
    recall_raw_top3 = []     # recall using raw top-3 union (no sim filter, no vote)
    recall_with_filter = []  # recall using top-3 + sim>0.8 + vote>=2
    recall_with_simfilt_only = []  # recall using top-3 + sim>0.8 only (no vote)
    bridges_kept_after_filter = []

    for ti in with_bridges:
        own = set(recs[ti]['bridges'])
        sims = embs @ embs[ti]
        cand = [(i, sims[i]) for i in pool if i != ti]
        cand.sort(key=lambda x: -x[1])
        top = cand[:K]

        # Raw top-3 union
        raw_borrowed = set()
        for j, _ in top:
            raw_borrowed.update(recs[j]['bridges'])
        recall_raw_top3.append(len(own & raw_borrowed) / max(len(own), 1))

        # Apply sim > 0.8 filter
        qualified = [(j, s) for j, s in top if s > SIM_TH]
        qualified_count_dist[len(qualified)] += 1
        if len(qualified) >= 1:
            n_with_any_qualified += 1

        # Recall using sim-filter only (no vote)
        sim_filt_borrowed = set()
        for j, _ in qualified:
            sim_filt_borrowed.update(recs[j]['bridges'])
        recall_with_simfilt_only.append(len(own & sim_filt_borrowed) / max(len(own), 1))

        # Vote count among qualified
        vote_counter = Counter()
        for j, _ in qualified:
            for b in set(recs[j]['bridges']):
                vote_counter[b] += 1
        kept = {b for b, v in vote_counter.items() if v >= VOTE_TH}
        bridges_kept_after_filter.append(len(kept))
        recall_with_filter.append(len(own & kept) / max(len(own), 1))
        if len(kept) > 0:
            n_with_ge_vote_th += 1

    n = len(with_bridges)
    print(f"\n--- Filter coverage (top-3 NN, sim > {SIM_TH}) ---")
    print(f"  Queries with ≥ 1 qualified NN:        {n_with_any_qualified}/{n} = {n_with_any_qualified/n:.1%}")
    print(f"  Distribution of #qualified NN:        {dict(sorted(qualified_count_dist.items()))}")
    print(f"  Queries with ≥ 1 bridge after vote≥{VOTE_TH}: {n_with_ge_vote_th}/{n} = {n_with_ge_vote_th/n:.1%}")
    print()
    print(f"--- Bridges kept per query (vote ≥ {VOTE_TH}) ---")
    print(f"  mean: {np.mean(bridges_kept_after_filter):.2f}")
    print(f"  median: {int(np.median(bridges_kept_after_filter))}")
    print(f"  >0: {sum(1 for x in bridges_kept_after_filter if x>0)}/{n}")
    print()
    print(f"--- Recall (|own ∩ borrowed| / |own|) ---")
    print(f"  raw top-3 union (no filter):                       mean={np.mean(recall_raw_top3):.3f}")
    print(f"  top-3 + sim>{SIM_TH} only (no vote):                  mean={np.mean(recall_with_simfilt_only):.3f}")
    print(f"  top-3 + sim>{SIM_TH} + vote≥{VOTE_TH} (your config):    mean={np.mean(recall_with_filter):.3f}")
    print()
    # Conditional recall: only over queries that HAVE qualified NN
    qualified_subset = [(r1, r2, r3) for r1, r2, r3, c in
                        zip(recall_raw_top3, recall_with_simfilt_only,
                            recall_with_filter, [qualified_count_dist[i] for i in []])
                        if True]
    # Simpler: collect per-query, mark which had qualified
    print(f"--- Conditional: among queries that HAVE ≥ 1 qualified NN ({n_with_any_qualified}) ---")
    rraw, rsimf, rfilt = [], [], []
    for ti, raw, sf, fl in zip(with_bridges, recall_raw_top3, recall_with_simfilt_only, recall_with_filter):
        sims = embs @ embs[ti]
        cand = [(i, sims[i]) for i in pool if i != ti]
        cand.sort(key=lambda x: -x[1])
        n_qual = sum(1 for _, s in cand[:K] if s > SIM_TH)
        if n_qual >= 1:
            rraw.append(raw); rsimf.append(sf); rfilt.append(fl)
    if rraw:
        print(f"  raw top-3:           mean={np.mean(rraw):.3f}")
        print(f"  sim-filt only:       mean={np.mean(rsimf):.3f}")
        print(f"  sim-filt + vote≥{VOTE_TH}:  mean={np.mean(rfilt):.3f}")


if __name__ == "__main__":
    main()
