"""Show top-1 most similar query and its bridges, for selected cases."""
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
        recs.append({'idx': idx, 'q': r['question'], 'hop': hop, 'bridges': bridges,
                     'gold': r['gold_answer'], 'em': r['ner_em']})

    emb_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    em = _get_embedding_model_class(embedding_model_name=emb_name)(embedding_model_name=emb_name)
    print("Encoding queries (FIXED MiniLM)...")
    embs = em.batch_encode([r['q'] for r in recs], norm=True)

    # Pick 12 cases mixed
    np.random.seed(7)
    pool_2 = [i for i in range(1000) if recs[i]['hop'] == 2]
    pool_3 = [i for i in range(1000) if recs[i]['hop'] == 3]
    pool_4 = [i for i in range(1000) if recs[i]['hop'] == 4]
    sample_idxs = list(np.random.choice(pool_2, 4, replace=False)) + \
                  list(np.random.choice(pool_3, 4, replace=False)) + \
                  list(np.random.choice(pool_4, 4, replace=False))

    for ti in sample_idxs:
        rs = recs[ti]
        sims = embs @ embs[ti]
        cand = [(j, sims[j]) for j in range(1000)
                if j != ti and recs[j]['hop'] == rs['hop']]
        cand.sort(key=lambda x: -x[1])
        j, sim = cand[0]  # top-1 neighbor
        rn = recs[j]

        own = set(b.lower() for b in rs['bridges'])
        borrowed = set(b.lower() for b in rn['bridges'])
        overlap = own & borrowed
        new_from_borrowed = borrowed - own
        missed = own - borrowed

        print(f"\n{'='*80}")
        print(f"idx={ti}  hop={rs['hop']}  gold='{rs['gold']}'  (canonical em={rs['em']})")
        print(f"  Q: {rs['q']}")
        print(f"  own bridges ({len(rs['bridges'])}): {rs['bridges']}")
        print()
        print(f"  Top-1 NN: idx={rn['idx']}  sim={sim:.3f}")
        print(f"    NN Q: {rn['q']}")
        print(f"    NN gold: '{rn['gold']}'")
        print(f"    NN bridges ({len(rn['bridges'])}): {rn['bridges']}")
        print()
        print(f"  → Overlap own ∩ borrowed: {len(overlap)}    {sorted(overlap)}")
        print(f"  → New from NN (not in own): {len(new_from_borrowed)}    {sorted(new_from_borrowed)}")
        print(f"  → Missed by NN (in own only): {len(missed)}    {sorted(missed)}")


if __name__ == "__main__":
    main()
