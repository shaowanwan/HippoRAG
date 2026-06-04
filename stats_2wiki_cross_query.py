"""Cross-query bridge stats on 2Wiki (same as MuSiQue analysis)."""
import json
import os
import sys

os.environ['OPENAI_API_KEY'] = 'sk-396199ed7af84eff8a0cf7a71b797601'
sys.path.insert(0, 'src')


def main():
    import numpy as np
    from src.hipporag.embedding_model import _get_embedding_model_class

    d = json.load(open('outputs/2wikimultihopqa_ner_pipeline_eval/comparison_results_rounds3_20260519_225606_28808.json'))
    data = json.load(open('2wikimultihopqa.json'))
    recs = []
    for r in d['results']:
        idx = r['idx']
        rds = r.get('round_diagnostics') or []
        bridges = rds[-1].get('discovered_entities_total', []) if rds else []
        hop = data[idx].get('_2wiki_hop', 0)
        type_ = data[idx].get('_2wiki_type', '?')
        recs.append({'idx': idx, 'q': r['question'], 'hop': hop, 'type': type_,
                     'bridges': bridges})
    print(f"Loaded {len(recs)} records")
    from collections import Counter
    print(f"Hop dist: {dict(sorted(Counter(r['hop'] for r in recs).items()))}")
    print(f"Type dist: {dict(sorted(Counter(r['type'] for r in recs).items()))}")

    em_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    emb_model = _get_embedding_model_class(embedding_model_name=em_name)(embedding_model_name=em_name)
    print("Encoding queries (FIXED MiniLM)...")
    embs = emb_model.batch_encode([r['q'] for r in recs], norm=True)

    KS = [1, 3, 5, 10]

    def compute_for_pool(pool_label, pool_set, group_key):
        """Compute K-sweep stats; neighbors restricted to same group_key value (hop or type)."""
        pool_list = sorted(pool_set)
        with_bridges = [i for i in pool_list if len(recs[i]['bridges']) > 0]
        print(f"\n=== {pool_label}  (n={len(pool_list)}, with bridges={len(with_bridges)}) ===")
        for K in KS:
            sim_list = []; own_list = []; borrow_list = []; recall_list = []; prec_list = []
            for ti in with_bridges:
                own = set(recs[ti]['bridges'])
                sims = embs @ embs[ti]
                cand = [(i, sims[i]) for i in pool_list
                        if i != ti and recs[i][group_key] == recs[ti][group_key]]
                if not cand: continue
                cand.sort(key=lambda x: -x[1])
                top = cand[:K]
                if not top: continue
                sim_list.append(np.mean([s for _, s in top]))
                borrowed = set()
                for j, _ in top:
                    borrowed.update(recs[j]['bridges'])
                own_list.append(len(own))
                borrow_list.append(len(borrowed))
                ov = len(own & borrowed)
                recall_list.append(ov / max(len(own), 1))
                prec_list.append(ov / max(len(borrowed), 1))
            if recall_list:
                print(f"  K={K:<3} (n={len(recall_list)}): "
                      f"sim={np.mean(sim_list):.3f}  "
                      f"own={np.mean(own_list):.2f}  "
                      f"borrowed={np.mean(borrow_list):.2f}  "
                      f"recall={np.mean(recall_list):.3f}  "
                      f"precision={np.mean(prec_list):.3f}")

    # Overall, restricted to same hop
    compute_for_pool("All 1000 (same hop NN)", set(range(1000)), 'hop')

    # Per hop
    for h in [2, 4]:
        pool = {i for i in range(1000) if recs[i]['hop'] == h}
        compute_for_pool(f"hop={h} only", pool, 'hop')

    # Per type
    for t in ['bridge_comparison', 'comparison', 'compositional', 'inference']:
        pool = {i for i in range(1000) if recs[i]['type'] == t}
        compute_for_pool(f"type={t}", pool, 'type')


if __name__ == "__main__":
    main()
