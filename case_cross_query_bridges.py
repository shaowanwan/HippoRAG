"""Case study: cross-query bridge sharing on existing 1000 samples."""
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
        hop_str = data[idx]['id'].split('hop')[0]
        try:
            hop = int(hop_str)
        except:
            hop = 0
        recs.append({'idx': idx, 'q': r['question'], 'hop': hop,
                     'bridges': bridges, 'gold': r['gold_answer'],
                     'em': r['ner_em']})

    print(f"Loaded {len(recs)} records  "
          f"(2-hop: {sum(1 for r in recs if r['hop']==2)}, "
          f"3-hop: {sum(1 for r in recs if r['hop']==3)}, "
          f"4-hop: {sum(1 for r in recs if r['hop']==4)})")

    emb_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    emb_model = _get_embedding_model_class(embedding_model_name=emb_name)(embedding_model_name=emb_name)
    print("Encoding 1000 queries (with FIXED Transformers.py)...")
    qs = [r['q'] for r in recs]
    embs = emb_model.batch_encode(qs, norm=True)
    print(f"Embedded: {embs.shape}\n")

    test_idxs = [0, 1, 5, 17, 23, 100]
    K = 5

    for ti in test_idxs:
        rs = recs[ti]
        sims = embs @ embs[ti]
        cand = [(i, sims[i]) for i in range(len(recs))
                if i != ti and recs[i]['hop'] == rs['hop']]
        cand.sort(key=lambda x: -x[1])
        print(f"=== idx={ti}  hop={rs['hop']}  gold={rs['gold']!r}  (canonical em={rs['em']}) ===")
        print(f"    Q: {rs['q']}")
        print(f"    own bridges: {rs['bridges']}")
        print(f"    --- Top-{K} same-hop NN ---")
        borrowed = set()
        for j, (i, sim) in enumerate(cand[:K]):
            rn = recs[i]
            print(f"      [{j}] sim={sim:.3f}  idx={i}  Q: {rn['q'][:70]}")
            print(f"          bridges: {rn['bridges']}")
            borrowed.update(rn['bridges'])
        own = set(rs['bridges'])
        overlap = own & borrowed
        new_b = borrowed - own
        missed = own - borrowed
        print(f"    --- Bridge analysis ---")
        print(f"      Own bridges:               {len(own):>3}")
        print(f"      Borrowed (union of {K}):    {len(borrowed):>3}")
        print(f"      Overlap (own ∩ borrowed):  {len(overlap):>3}    {sorted(overlap)}")
        print(f"      New from neighbors:        {len(new_b):>3}    {sorted(new_b)[:5]}{'...' if len(new_b)>5 else ''}")
        print(f"      Missed by neighbors:       {len(missed):>3}    {sorted(missed)}")
        print()


if __name__ == "__main__":
    main()
