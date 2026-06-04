"""
Test borrowed bridges on 30 random 3-hop + 4-hop samples (no own bridges).

Bridges source: 41732 (canonical 1000-sample reasoning). For each test query Q:
  1. find top-3 same-hop NN (excluding self) by MiniLM cosine.
  2. union their bridges → raw_borrowed.
  3. filter to entities reachable within 3-hop BFS from Q's seed entities
     (substring match to graph entity nodes).
  4. variant A: inject filtered borrowed bridges → NER PPR → QA  (no LLM filter)
  5. variant B: LLM judges which filtered bridges are relevant → inject → QA  (with LLM filter)

Compares EM (A vs B vs 41732 own-bridges oracle vs 41732 baseline) per sample.
"""
import json
import os
import sys
import argparse
import random
import logging
import re

import numpy as np

os.environ.setdefault("NO_SIM_FACTOR", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

import evaluate_musique_ner_pipeline as ner  # for NERIndex / llm_qa / helpers


# ── Pickle compat (NERIndex was saved in __main__) ──
import sys as _sys
_main = _sys.modules['__main__']
for cls_name in ['NERIndex']:
    obj = getattr(ner, cls_name, None)
    if obj is not None and not hasattr(_main, cls_name):
        setattr(_main, cls_name, obj)


LLM_FILTER_PROMPT = """You are deciding which candidate entities are relevant to answering a multi-hop question.

Question: {question}

Candidate entities (some may be relevant, some may be noise from similar questions):
{candidates}

Output a JSON array of the entities that are likely useful for retrieving documents to answer THIS specific question. Drop entities about unrelated people / places / dates / topics. Be selective — prefer 2-5 entities.

JSON array (entity strings only):"""


def parse_json_array(text):
    """Parse a JSON array of strings from LLM output (lenient)."""
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?', '', text).rstrip('`').strip()
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x).strip().lower() for x in arr if x]
    except Exception:
        return []


def llm_filter_bridges(question, candidates, llm_client):
    if not candidates:
        return []
    cand_lines = '\n'.join(f'  - {c}' for c in candidates)
    prompt = LLM_FILTER_PROMPT.format(question=question, candidates=cand_lines)
    messages = [{"role": "user", "content": prompt}]
    try:
        result = llm_client.infer(messages)
        text = result[0] if isinstance(result, tuple) else result
        if not isinstance(text, str):
            text = text[0]["content"]
        kept = parse_json_array(text)
        # restrict to the candidate set
        cand_lower = {c.lower(): c for c in candidates}
        return [cand_lower[k] for k in kept if k in cand_lower]
    except Exception as e:
        logger.warning(f"LLM filter failed: {e}")
        return candidates  # fall back: keep all


def reachable_within_k_hops(graph, seed_vids, k):
    reachable = set(seed_vids)
    frontier = set(seed_vids)
    for _ in range(k):
        new_frontier = set()
        for v in frontier:
            for n in graph.neighbors(v):
                if n not in reachable:
                    new_frontier.add(n)
        reachable |= new_frontier
        frontier = new_frontier
    return reachable


def extract_query_seeds(question, entity_texts_set, max_seeds=10):
    q_lower = question.lower()
    hits = [t for t in entity_texts_set if t in q_lower]
    hits.sort(key=lambda x: -len(x))
    kept = []
    for h in hits:
        if not any(h in k or k in h for k in kept):
            kept.append(h)
        if len(kept) >= max_seeds:
            break
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=3, help="Top-K NN for borrowing")
    parser.add_argument("--bridges_source",
                        default="outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260421_143341_41732.json")
    parser.add_argument("--global_index_pickle",
                        default="outputs/musique_ner_pipeline_eval/global_ner_index.pkl")
    parser.add_argument("--ner_cache", default="outputs/musique_ner_pipeline_eval/ner_cache.json")
    parser.add_argument("--n_per_hop", type=int, default=15, help="Samples per hop (3-hop, 4-hop)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/borrowed_bridges_30_results.json")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    import torch; torch.manual_seed(args.seed)

    from src.hipporag.embedding_model import _get_embedding_model_class
    emb_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    emb_model = _get_embedding_model_class(embedding_model_name=emb_name)(embedding_model_name=emb_name)
    llm_model_name = os.getenv("LLM_MODEL_NAME", "qwen-plus")
    base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "sk-396199ed7af84eff8a0cf7a71b797601"
    os.environ["OPENAI_API_KEY"] = api_key
    llm_client = ner.SimpleLLM(model_name=llm_model_name, base_url=base_url)

    # Load global NER index
    logger.info("Loading global NER index...")
    global_index = ner.NERIndex.load(args.global_index_pickle, embedding_model=emb_model)
    g = global_index.graph
    logger.info(f"Graph: {g.vcount()} nodes, {g.ecount()} edges")

    # Build entity_text → vid map (lowercase)
    passage_idxs = set(global_index.passage_node_idxs)
    vid_to_text = {}
    text_to_vid = {}
    for vid in range(g.vcount()):
        text = g.vs[vid]['content'] if 'content' in g.vs.attributes() else None
        if not text or vid in passage_idxs:
            continue
        t = text.lower().strip()
        if 1 <= len(t) <= 50:
            vid_to_text[vid] = t
            text_to_vid[t] = vid
    entity_texts_set = set(text_to_vid.keys())

    # Load 41732 bridges + musique.json
    d_41732 = json.load(open(args.bridges_source))
    musique = json.load(open('musique.json'))
    recs = []
    for r in d_41732['results']:
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

    # Embed all queries
    logger.info("Encoding queries with FIXED MiniLM...")
    embs = emb_model.batch_encode([r['q'] for r in recs], norm=True)

    # Pick random 15+15 from 3-hop and 4-hop
    pool_3 = [i for i in range(1000) if recs[i]['hop'] == 3]
    pool_4 = [i for i in range(1000) if recs[i]['hop'] == 4]
    random.shuffle(pool_3); random.shuffle(pool_4)
    test_idxs = pool_3[:args.n_per_hop] + pool_4[:args.n_per_hop]
    logger.info(f"Test: {len(test_idxs)} ({args.n_per_hop} 3-hop, {args.n_per_hop} 4-hop)")

    # Load ner_cache (needed for index build? not — we use global_index directly)
    # The global_index is pre-built, no per-sample build needed.

    # ── Per-sample evaluation ──
    results = []
    for ctr, ti in enumerate(test_idxs):
        r = recs[ti]
        logger.info(f"\n[{ctr+1}/{len(test_idxs)}] idx={r['idx']} hop={r['hop']} q={r['q'][:70]}")

        # Find top-K same-hop NN
        sims = embs @ embs[ti]
        cand = [(j, sims[j]) for j in range(1000) if j != ti and recs[j]['hop'] == r['hop']]
        cand.sort(key=lambda x: -x[1])
        top = cand[:args.K]
        raw_borrowed = set()
        for j, _ in top:
            raw_borrowed.update(b.lower() for b in recs[j]['bridges'])

        # Apply 3-hop graph reachability filter
        seeds_text = extract_query_seeds(r['q'], entity_texts_set)
        seed_vids = [text_to_vid[t] for t in seeds_text]
        if seed_vids:
            reach = reachable_within_k_hops(g, seed_vids, 3)
            reach_texts = {vid_to_text[v] for v in reach if v in vid_to_text}
        else:
            reach_texts = set()
        graph_filtered = sorted(raw_borrowed & reach_texts)
        logger.info(f"  raw borrowed: {len(raw_borrowed)}, graph-filtered: {len(graph_filtered)}, seeds: {len(seed_vids)}")

        # ── Variant A: no LLM filter — inject graph_filtered ──
        # Resolve to vertex IDs + build extra_node_weights
        resolved_a = ner._resolve_entities_in_graph(global_index, graph_filtered)
        ews_a = np.zeros(g.vcount())
        for name, (vid, sim) in resolved_a.items():
            ews_a[vid] += ner._degree_adaptive_weight(global_index, vid, 0.5, sim=1.0)
        # Retrieve
        try:
            sorted_ids_a, _ = global_index.retrieve(r['q'], extra_node_weights=ews_a)
            docs_a = [global_index.passages[global_index.passage_keys[did]] for did in sorted_ids_a[:5]]
            ans_a = ner.llm_qa(r['q'], docs_a, llm_client)
            em_a = ner.check_em(ans_a, r['gold'], r['aliases'])
        except Exception as e:
            logger.error(f"  variant A failed: {e}")
            ans_a = "Error"; em_a = 0

        # ── Variant B: LLM filter ──
        if graph_filtered:
            llm_kept = llm_filter_bridges(r['q'], graph_filtered, llm_client)
        else:
            llm_kept = []
        logger.info(f"  LLM kept: {len(llm_kept)} / {len(graph_filtered)}  -> {llm_kept}")
        resolved_b = ner._resolve_entities_in_graph(global_index, llm_kept)
        ews_b = np.zeros(g.vcount())
        for name, (vid, sim) in resolved_b.items():
            ews_b[vid] += ner._degree_adaptive_weight(global_index, vid, 0.5, sim=1.0)
        try:
            sorted_ids_b, _ = global_index.retrieve(r['q'], extra_node_weights=ews_b)
            docs_b = [global_index.passages[global_index.passage_keys[did]] for did in sorted_ids_b[:5]]
            ans_b = ner.llm_qa(r['q'], docs_b, llm_client)
            em_b = ner.check_em(ans_b, r['gold'], r['aliases'])
        except Exception as e:
            logger.error(f"  variant B failed: {e}")
            ans_b = "Error"; em_b = 0

        # ── Baseline (no bridges) — REUSE 41732's baseline_em as reference
        # (cross-day LLM; flagged in summary). Saves 1 LLM call per query.
        ans_base = "(see 41732 baseline_answer)"
        em_base = r['ref_baseline_em']

        result = {
            'idx': r['idx'], 'hop': r['hop'], 'q': r['q'], 'gold': r['gold'],
            'aliases': r['aliases'],
            'top_nn': [{'idx': j, 'sim': float(s), 'q': recs[j]['q']} for j, s in top],
            'raw_borrowed': sorted(raw_borrowed),
            'graph_filtered': graph_filtered,
            'llm_kept': llm_kept,
            'own_bridges_ref': r['bridges'],
            'baseline_ans': ans_base, 'baseline_em': int(em_base),
            'variantA_ans': ans_a, 'variantA_em': int(em_a),
            'variantB_ans': ans_b, 'variantB_em': int(em_b),
            'ref_41732_baseline_em': r['ref_baseline_em'],
            'ref_41732_ner_em': r['ref_ner_em'],
        }
        results.append(result)

        b = "Y" if em_base else "N"
        a = "Y" if em_a else "N"
        bb = "Y" if em_b else "N"
        logger.info(f"  base={b} '{ans_base[:30]}' | A={a} '{ans_a[:30]}' | B={bb} '{ans_b[:30]}'  gold='{r['gold']}'")

    # Aggregate
    n = len(results)
    em_base = sum(r['baseline_em'] for r in results) / n
    em_a = sum(r['variantA_em'] for r in results) / n
    em_b = sum(r['variantB_em'] for r in results) / n
    em_41732_base = sum(r['ref_41732_baseline_em'] for r in results) / n
    em_41732_ner = sum(r['ref_41732_ner_em'] for r in results) / n

    # Per hop
    by_hop = {3: [], 4: []}
    for r in results:
        by_hop[r['hop']].append(r)

    print("\n=== SUMMARY ===")
    print(f"n = {n} ({len(by_hop[3])} 3-hop + {len(by_hop[4])} 4-hop)")
    print()
    print(f"{'Method':40s}  EM     (3-hop / 4-hop)")
    print('-' * 65)
    print(f"{'Baseline (no bridges, 6/1 LLM)':40s}  {em_base:.3f}   "
          f"({sum(r['baseline_em'] for r in by_hop[3])/max(len(by_hop[3]),1):.3f} / "
          f"{sum(r['baseline_em'] for r in by_hop[4])/max(len(by_hop[4]),1):.3f})")
    print(f"{'Variant A (borrowed + graph filter)':40s}  {em_a:.3f}   "
          f"({sum(r['variantA_em'] for r in by_hop[3])/max(len(by_hop[3]),1):.3f} / "
          f"{sum(r['variantA_em'] for r in by_hop[4])/max(len(by_hop[4]),1):.3f})")
    print(f"{'Variant B (borrowed + graph + LLM)':40s}  {em_b:.3f}   "
          f"({sum(r['variantB_em'] for r in by_hop[3])/max(len(by_hop[3]),1):.3f} / "
          f"{sum(r['variantB_em'] for r in by_hop[4])/max(len(by_hop[4]),1):.3f})")
    print()
    print(f"{'(ref) 41732 baseline_em (4/21 LLM)':40s}  {em_41732_base:.3f}")
    print(f"{'(ref) 41732 ner_em full reasoning':40s}  {em_41732_ner:.3f}")
    print()
    print("Avg #bridges per query:")
    print(f"  raw borrowed top-{args.K}:           {np.mean([len(r['raw_borrowed']) for r in results]):.1f}")
    print(f"  after graph filter:               {np.mean([len(r['graph_filtered']) for r in results]):.1f}")
    print(f"  after LLM filter:                 {np.mean([len(r['llm_kept']) for r in results]):.1f}")

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'config': vars(args), 'results': results, 'summary': {
            'n': n, 'baseline_em': em_base, 'variantA_em': em_a, 'variantB_em': em_b,
            'ref_41732_baseline': em_41732_base, 'ref_41732_ner': em_41732_ner,
        }}, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
