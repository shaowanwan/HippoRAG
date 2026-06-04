"""Stats: similarity vs bridge overlap across 1000 samples (same-hop NN)."""
import json
import os
import sys

os.environ['OPENAI_API_KEY'] = 'sk-396199ed7af84eff8a0cf7a71b797601'
sys.path.insert(0, 'src')


def main():
    import numpy as np
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
    print("Encoding queries (FIXED Transformers.py)...")
    embs = em.batch_encode([r['q'] for r in recs], norm=True)
    print(f"embs={embs.shape}\n")

    KS = [1, 3, 5, 10]

    def compute_for_pool(pool_idx_set, pool_label):
        """Compute K-sweep stats for queries in pool_idx_set, neighbors restricted to pool_idx_set\\{self}."""
        print(f"=== {pool_label}  (n={len(pool_idx_set)}) ===")
        pool_list = sorted(pool_idx_set)
        # Pre-filter records that have NON-EMPTY bridges (sample with 0 bridges → overlap_rate undefined)
        has_bridges = [i for i in pool_list if len(recs[i]['bridges']) > 0]
        print(f"  with non-empty bridges: {len(has_bridges)}")

        # For each K, accumulate stats over queries with non-empty bridges
        for K in KS:
            sim_topK_list = []
            own_count_list = []
            borrowed_count_list = []
            overlap_rate_list = []   # |own ∩ borrowed| / |own|
            precision_list = []      # |own ∩ borrowed| / |borrowed|
            for ti in has_bridges:
                own = set(recs[ti]['bridges'])
                # cosine vector to all
                sims = embs @ embs[ti]
                # same-hop pool, exclude self
                cand = [(i, sims[i]) for i in pool_list
                        if i != ti and recs[i]['hop'] == recs[ti]['hop']]
                if not cand:
                    continue
                cand.sort(key=lambda x: -x[1])
                top = cand[:K]
                if not top: continue
                sim_topK_list.append(np.mean([s for _, s in top]))
                borrowed = set()
                for j, _ in top:
                    borrowed.update(recs[j]['bridges'])
                own_count_list.append(len(own))
                borrowed_count_list.append(len(borrowed))
                ov = len(own & borrowed)
                overlap_rate_list.append(ov / max(len(own), 1))
                if len(borrowed) > 0:
                    precision_list.append(ov / len(borrowed))
                else:
                    precision_list.append(0.0)
            n = len(overlap_rate_list)
            if n == 0:
                print(f"  K={K}: no data")
                continue
            print(f"  K={K:<3} (n={n}):  "
                  f"mean sim={np.mean(sim_topK_list):.3f}  "
                  f"own bridges={np.mean(own_count_list):.2f}  "
                  f"borrowed={np.mean(borrowed_count_list):.2f}  "
                  f"overlap rate (recall)={np.mean(overlap_rate_list):.3f}  "
                  f"precision={np.mean(precision_list):.3f}")
        print()

    # Overall (all 1000)
    compute_for_pool(set(range(1000)), "ALL 1000 (full bank, same-hop NN)")

    # Per-hop subsets (bank restricted to same hop, which it already is via filter — but pool same)
    for hop in [2, 3, 4]:
        pool = {i for i in range(1000) if recs[i]['hop'] == hop}
        compute_for_pool(pool, f"hop={hop} only")

    # Similarity-stratified analysis: take K=5, bucket overlap by sim of nearest neighbor
    print("=== Sim-stratified overlap (K=5 NN, same hop, all 1000) ===")
    buckets = {(0.0, 0.4): [], (0.4, 0.6): [], (0.6, 0.8): [], (0.8, 1.01): []}
    for ti in range(1000):
        own = set(recs[ti]['bridges'])
        if not own: continue
        sims = embs @ embs[ti]
        cand = [(i, sims[i]) for i in range(1000)
                if i != ti and recs[i]['hop'] == recs[ti]['hop']]
        if not cand: continue
        cand.sort(key=lambda x: -x[1])
        top = cand[:5]
        avg_sim = np.mean([s for _, s in top])
        borrowed = set()
        for j, _ in top:
            borrowed.update(recs[j]['bridges'])
        ov = len(own & borrowed)
        recall = ov / max(len(own), 1)
        for (lo, hi), arr in buckets.items():
            if lo <= avg_sim < hi:
                arr.append(recall)
                break
    print(f"{'avg sim bin':>15} | {'n queries':>10} | {'mean recall':>12}")
    print('-' * 45)
    for (lo, hi), arr in buckets.items():
        if arr:
            print(f"  [{lo:.1f}, {hi:.1f})  | {len(arr):>10} | {np.mean(arr):>12.3f}")
        else:
            print(f"  [{lo:.1f}, {hi:.1f})  | {len(arr):>10} | n/a")


if __name__ == "__main__":
    main()
