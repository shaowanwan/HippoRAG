"""Phase 1 — borrow from top-1 NN (sim>0.85) gated by LLM equivalence judge.

Pipeline per query:
  1. find top-1 same-hop NN by cosine
  2. if sim < 0.85: skip (no borrow)
  3. LLM judge: "Are Q and NN_Q semantically equivalent? yes/no"
  4. if yes: borrow NN bridges, inject NER PPR, QA
  5. if no:  do NOT borrow; record decision

Run on same top 30 by sim (mix of 2/3/4-hop) — compare to previous unconditional borrow.
"""
import json
import os
import sys
import argparse
import re

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


JUDGE_PROMPT = """Decide if two multi-hop questions are semantically equivalent — i.e., they would have the EXACTLY SAME answer about the same entities.

Be strict:
- Paraphrases (same meaning, different wording) = equivalent
- Structural swaps (e.g., "X gain control of Y" vs "Y gain control of X") = NOT equivalent
- Different entities (even one mention) = NOT equivalent
- Different time periods or geographic scope = NOT equivalent

Q1: {q1}
Q2: {q2}

Respond with exactly one word: yes or no."""


def llm_judge(q1, q2, llm_client):
    msg = [{"role": "user", "content": JUDGE_PROMPT.format(q1=q1, q2=q2)}]
    try:
        result = llm_client.infer(msg)
        text = result[0] if isinstance(result, tuple) else result
        if not isinstance(text, str):
            text = text[0]["content"]
        return text.strip().lower().startswith("yes")
    except Exception as e:
        logger.warning(f"  judge failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_threshold", type=float, default=0.85)
    parser.add_argument("--n_max", type=int, default=30)
    parser.add_argument("--out", default="outputs/borrow_with_judge_results.json")
    parser.add_argument("--no_hop_filter", action="store_true",
                        help="If set, NN candidates come from ALL hops (realistic scenario, "
                             "no oracle hop label).")
    args = parser.parse_args()

    import numpy as np
    from src.hipporag.embedding_model import _get_embedding_model_class

    em_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    emb_model = _get_embedding_model_class(embedding_model_name=em_name)(embedding_model_name=em_name)

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

    print("Encoding...")
    embs = emb_model.batch_encode([r['q'] for r in recs], norm=True)

    # Compute top-1 NN sim for ALL 1000 queries (not just qualified)
    all_q_nn = []
    for ti in range(1000):
        rs = recs[ti]
        sims = embs @ embs[ti]
        if args.no_hop_filter:
            cand = [(j, sims[j]) for j in range(1000) if j != ti]
        else:
            cand = [(j, sims[j]) for j in range(1000)
                    if j != ti and recs[j]['hop'] == rs['hop']]
        if not cand:
            all_q_nn.append((ti, -1, 0.0))
            continue
        cand.sort(key=lambda x: -x[1])
        j, sim = cand[0]
        all_q_nn.append((ti, j, sim))

    # Run policy on all 1000 (or up to n_max for testing)
    test = all_q_nn[:args.n_max]
    n_qual = sum(1 for _, _, s in test if s > args.sim_threshold)
    print(f"Test set: {len(test)} queries  ({n_qual} with NN sim > {args.sim_threshold})")

    print("Loading global NER index...")
    global_index = ner.NERIndex.load('outputs/musique_ner_pipeline_eval/global_ner_index.pkl',
                                       embedding_model=emb_model)
    g = global_index.graph

    llm_client = ner.SimpleLLM(
        model_name=os.getenv("LLM_MODEL_NAME", "qwen-plus"),
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )

    # Resume from existing if present
    results = []
    done_idxs = set()
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            results = prev.get('results', [])
            done_idxs = {r['idx'] for r in results}
            print(f"Resuming from {len(results)} done")
        except: pass

    for ctr, (ti, j, sim) in enumerate(test):
        rs = recs[ti]
        if rs['idx'] in done_idxs:
            continue
        # If sim < threshold or no NN: skip borrow, mark as fallback
        if sim <= args.sim_threshold or j < 0:
            results.append({
                'idx': rs['idx'], 'hop': rs['hop'], 'sim_to_nn': float(sim),
                'q': rs['q'], 'nn_q': recs[j]['q'] if j >= 0 else None,
                'gold': rs['gold'], 'nn_gold': recs[j]['gold'] if j >= 0 else None,
                'judge': 'skip_low_sim', 'borrowed': False,
                'borrowed_ans': None, 'borrowed_em': None,
                'ref_baseline_em': rs['ref_baseline_em'], 'ref_ner_em': rs['ref_ner_em'],
            })
            if (ctr + 1) % 25 == 0:
                with open(args.out, 'w') as f:
                    json.dump({'config': vars(args), 'results': results}, f, indent=2)
            continue

        rn = recs[j]
        # Step A: LLM equivalence judge
        is_eq = llm_judge(rs['q'], rn['q'], llm_client)
        logger.info(f"\n[{ctr+1}/{len(test)}] idx={rs['idx']} hop={rs['hop']} sim={sim:.3f}  judge={'yes' if is_eq else 'no'}")

        if not is_eq:
            results.append({
                'idx': rs['idx'], 'hop': rs['hop'], 'sim_to_nn': float(sim),
                'q': rs['q'], 'nn_q': rn['q'], 'gold': rs['gold'], 'nn_gold': rn['gold'],
                'judge': 'no', 'borrowed': False,
                'borrowed_ans': None, 'borrowed_em': None,
                'ref_baseline_em': rs['ref_baseline_em'], 'ref_ner_em': rs['ref_ner_em'],
            })
            if (ctr + 1) % 25 == 0:
                with open(args.out, 'w') as f:
                    json.dump({'config': vars(args), 'results': results}, f, indent=2)
            continue

        # Step B: borrow NN bridges, inject, retrieve, QA
        borrowed = list(rn['bridges'])
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

        results.append({
            'idx': rs['idx'], 'hop': rs['hop'], 'sim_to_nn': float(sim),
            'q': rs['q'], 'nn_q': rn['q'], 'gold': rs['gold'], 'nn_gold': rn['gold'],
            'judge': 'yes', 'borrowed': True,
            'nn_bridges': borrowed, 'borrowed_ans': ans, 'borrowed_em': int(em_),
            'ref_baseline_em': rs['ref_baseline_em'], 'ref_ner_em': rs['ref_ner_em'],
        })
        b = "Y" if em_ else "N"
        logger.info(f"   borrowed={b}  '{ans[:30]}'  gold='{rs['gold']}'")

    # Aggregate
    n = len(results)
    n_yes = sum(1 for r in results if r['judge'] == 'yes')
    n_no = n - n_yes
    em_all_borrowed_yes = [r['borrowed_em'] for r in results if r['borrowed'] and r['borrowed_em'] is not None]
    em_borrowed_avg = sum(em_all_borrowed_yes) / max(len(em_all_borrowed_yes), 1) if em_all_borrowed_yes else 0
    # If "no", we'd fall back to full reasoning; ref_ner_em as proxy
    em_total_adaptive = sum(
        r['borrowed_em'] if r['borrowed'] and r['borrowed_em'] is not None else r['ref_ner_em']
        for r in results
    ) / n

    # Compare with previous unconditional borrow
    prev = json.load(open('outputs/borrow_high_sim_results.json'))
    prev_em_uncond = prev['summary']['borrowed_em_6_2']

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"n={n}, judge: yes={n_yes}, no={n_no}")
    print()
    print(f"  Unconditional borrow (prev exp):                  {prev_em_uncond:.3f}")
    print(f"  Borrow only on judge=yes (over those n={n_yes}):       {em_borrowed_avg:.3f}")
    print(f"  Adaptive: borrow if yes else fallback to reasoning: {em_total_adaptive:.3f}")
    print(f"  41732 full reasoning (4/21 LLM) on these {n}:       {sum(r['ref_ner_em'] for r in results)/n:.3f}")
    print(f"  41732 baseline (no bridges, 4/21 LLM):              {sum(r['ref_baseline_em'] for r in results)/n:.3f}")
    print()

    # Per-hop
    by_hop = {}
    for r in results: by_hop.setdefault(r['hop'], []).append(r)
    print("Per-hop:")
    for h, rs in sorted(by_hop.items()):
        ny = sum(1 for r in rs if r['judge'] == 'yes')
        em_y = [r['borrowed_em'] for r in rs if r['borrowed'] and r['borrowed_em'] is not None]
        ref_n = sum(r['ref_ner_em'] for r in rs) / max(len(rs), 1)
        print(f"  hop={h} (n={len(rs)}): judge_yes={ny}/{len(rs)},  borrowed_em={sum(em_y)/max(len(em_y),1) if em_y else 0:.3f},  ref_reasoning={ref_n:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'config': vars(args),
                   'summary': {'n': n, 'n_judge_yes': n_yes,
                              'unconditional_borrow_em': prev_em_uncond,
                              'borrowed_on_yes_em': em_borrowed_avg,
                              'adaptive_total_em': em_total_adaptive},
                   'results': results}, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
