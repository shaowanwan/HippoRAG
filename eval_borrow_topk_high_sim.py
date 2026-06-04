"""Adaptive policy validation: for queries with same-hop top-1 NN at sim > 0.85,
borrow NN's bridges directly (no LLM filter), inject into NER PPR, QA.

Compare:
  - 41732 baseline_em (no bridges, 4/21 LLM)         [ref]
  - 41732 ner_em (full reasoning, 4/21 LLM)          [ref upper bound]
  - borrowed (NN's bridges directly, 6/2 LLM)        [main]
"""
import json
import os
import sys
import argparse

os.environ['OPENAI_API_KEY'] = 'sk-396199ed7af84eff8a0cf7a71b797601'
os.environ.setdefault("NO_SIM_FACTOR", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

import evaluate_musique_ner_pipeline as ner
import sys as _sys
_sys.modules['__main__'].NERIndex = ner.NERIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_threshold", type=float, default=0.85)
    parser.add_argument("--n_max", type=int, default=30)
    parser.add_argument("--out", default="outputs/borrow_high_sim_results.json")
    args = parser.parse_args()

    import numpy as np
    from src.hipporag.embedding_model import _get_embedding_model_class

    emb_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    emb_model = _get_embedding_model_class(embedding_model_name=emb_name)(embedding_model_name=emb_name)

    # Load 41732 + musique
    d = json.load(open('outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260421_143341_41732.json'))
    musique = json.load(open('musique.json'))
    recs = []
    for r in d['results']:
        idx = r['idx']
        rds = r.get('round_diagnostics') or []
        bridges = rds[-1].get('discovered_entities_total', []) if rds else []
        try:
            hop = int(musique[idx]['id'].split('hop')[0])
        except:
            hop = 0
        recs.append({'idx': idx, 'q': r['question'], 'hop': hop, 'bridges': bridges,
                     'gold': r['gold_answer'], 'aliases': r.get('gold_aliases', []),
                     'ref_baseline_em': r['baseline_em'], 'ref_ner_em': r['ner_em']})

    print("Encoding queries (FIXED MiniLM)...")
    embs = emb_model.batch_encode([r['q'] for r in recs], norm=True)

    # Find queries with top-1 same-hop NN at sim > threshold
    qualified = []
    for ti in range(1000):
        rs = recs[ti]
        sims = embs @ embs[ti]
        cand = [(j, sims[j]) for j in range(1000)
                if j != ti and recs[j]['hop'] == rs['hop']]
        if not cand: continue
        cand.sort(key=lambda x: -x[1])
        j, sim = cand[0]
        if sim > args.sim_threshold:
            qualified.append((ti, j, sim))
    qualified.sort(key=lambda x: -x[2])  # by similarity desc
    print(f"Queries with top-1 same-hop NN sim > {args.sim_threshold}: {len(qualified)}")

    # Take top N by sim
    test = qualified[:args.n_max]
    print(f"Taking top {len(test)}")
    hops = [recs[ti]['hop'] for ti, _, _ in test]
    from collections import Counter
    print(f"Hop distribution: {dict(Counter(hops))}")

    # Load global NER index
    print("\nLoading global NER index...")
    global_index = ner.NERIndex.load('outputs/musique_ner_pipeline_eval/global_ner_index.pkl',
                                       embedding_model=emb_model)
    g = global_index.graph
    print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges")

    # LLM client
    llm_client = ner.SimpleLLM(
        model_name=os.getenv("LLM_MODEL_NAME", "qwen-plus"),
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )

    # Per-sample: borrow top-1 NN's bridges, inject, retrieve, QA
    results = []
    for ctr, (ti, j, sim) in enumerate(test):
        rs = recs[ti]; rn = recs[j]
        borrowed = list(rn['bridges'])
        logger.info(f"\n[{ctr+1}/{len(test)}] idx={rs['idx']} hop={rs['hop']} sim={sim:.3f}  borrowed={len(borrowed)}")

        # Resolve borrowed entities
        resolved = ner._resolve_entities_in_graph(global_index, borrowed)
        ews = np.zeros(g.vcount())
        for name, (vid, _) in resolved.items():
            ews[vid] += ner._degree_adaptive_weight(global_index, vid, 0.5, sim=1.0)
        try:
            sorted_ids, _ = global_index.retrieve(rs['q'], extra_node_weights=ews)
            docs = [global_index.passages[global_index.passage_keys[did]] for did in sorted_ids[:5]]
            ans = ner.llm_qa(rs['q'], docs, llm_client)
            em_ = ner.check_em(ans, rs['gold'], rs['aliases'])
        except Exception as e:
            logger.error(f"  failed: {e}")
            ans = "Error"; em_ = 0

        result = {
            'idx': rs['idx'], 'hop': rs['hop'], 'sim_to_nn': float(sim),
            'q': rs['q'], 'gold': rs['gold'],
            'nn_idx': rn['idx'], 'nn_q': rn['q'], 'nn_gold': rn['gold'],
            'nn_bridges': borrowed, 'n_resolved': len(resolved),
            'borrowed_ans': ans, 'borrowed_em': int(em_),
            'ref_baseline_em': rs['ref_baseline_em'],
            'ref_ner_em': rs['ref_ner_em'],
        }
        results.append(result)
        b = "Y" if em_ else "N"
        rb = "Y" if rs['ref_baseline_em'] else "N"
        rn_e = "Y" if rs['ref_ner_em'] else "N"
        logger.info(f"   base41732={rb}  ner41732={rn_e}  borrowed={b}  '{ans[:30]}'  gold='{rs['gold']}'")

    # Aggregate
    n = len(results)
    em_base = sum(r['ref_baseline_em'] for r in results) / n
    em_ner_full = sum(r['ref_ner_em'] for r in results) / n
    em_borrow = sum(r['borrowed_em'] for r in results) / n
    print("\n" + "="*70)
    print(f"SUMMARY (n={n}, top-1 same-hop NN sim > {args.sim_threshold})")
    print("="*70)
    print(f"  41732 baseline (no bridges, 4/21 LLM):       {em_base:.3f}")
    print(f"  41732 full reasoning (4/21 LLM):             {em_ner_full:.3f}")
    print(f"  Borrowed from NN, no filter, no rewrite (6/2): {em_borrow:.3f}")
    print()
    print("Per-hop:")
    by_hop = {}
    for r in results: by_hop.setdefault(r['hop'], []).append(r)
    for h, rs in sorted(by_hop.items()):
        if not rs: continue
        b = sum(r['ref_baseline_em'] for r in rs) / len(rs)
        n_ = sum(r['ref_ner_em'] for r in rs) / len(rs)
        br = sum(r['borrowed_em'] for r in rs) / len(rs)
        print(f"  hop={h} (n={len(rs):2d}):  baseline={b:.3f}  reasoning={n_:.3f}  borrowed={br:.3f}")

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'config': vars(args),
            'summary': {'n': n, 'baseline_em_4_21': em_base,
                       'full_reasoning_em_4_21': em_ner_full,
                       'borrowed_em_6_2': em_borrow},
            'results': results,
        }, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
