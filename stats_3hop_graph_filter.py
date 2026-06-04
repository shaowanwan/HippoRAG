"""Filter borrowed bridges by 3-hop reachability from query seed entities on the NER graph.

For each 3-hop query:
  pair_seeds = entities mentioned in the query (substring-match against graph node names)
  reachable_set = vertices within 3-hop BFS of pair_seeds
  filtered_borrowed = borrowed bridges ∩ reachable_set

Compares recall of:
  raw top-3 union (no filter)
  + sim > 0.8 + vote >= 2 (lexical filter)
  + 3-hop graph reachability (graph filter)
  + sim filter + 3-hop graph (combined)
"""
import json
import os
import sys
import pickle
import re

os.environ['OPENAI_API_KEY'] = 'sk-396199ed7af84eff8a0cf7a71b797601'
sys.path.insert(0, 'src')


def main():
    import numpy as np
    from collections import Counter
    import evaluate_musique_ner_pipeline as ner
    from src.hipporag.embedding_model import _get_embedding_model_class

    # Load 41732 + musique
    d = json.load(open('outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260421_143341_41732.json'))
    data = json.load(open('musique.json'))

    # Load global NER index (it has the graph)
    print("Loading global NER index pickle (661 MB)...")
    pickle_path = 'outputs/musique_ner_pipeline_eval/global_ner_index.pkl'
    emb_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    emb_model = _get_embedding_model_class(embedding_model_name=emb_name)(embedding_model_name=emb_name)
    # Pickle was saved with NERIndex in __main__; alias so unpickle resolves
    import sys as _sys
    _sys.modules['__main__'].NERIndex = ner.NERIndex
    global_index = ner.NERIndex.load(pickle_path, embedding_model=emb_model)
    print(f"  Graph: {global_index.graph.vcount()} nodes, {global_index.graph.ecount()} edges")

    # Build vertex_id → entity_text map and entity_text → vertex_id (lowercase)
    g = global_index.graph
    vid_to_text = {}
    text_to_vid = {}
    for vid in range(g.vcount()):
        text = g.vs[vid]['content'] if 'content' in g.vs.attributes() else None
        if not text: continue
        t = text.lower().strip()
        vid_to_text[vid] = t
        text_to_vid[t] = vid  # last write wins on duplicates, fine
    passage_idxs = set(global_index.passage_node_idxs)
    entity_vids = [vid for vid in range(g.vcount()) if vid not in passage_idxs]
    print(f"  Entity nodes: {len(entity_vids)}")

    # Build records
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

    # Embed queries
    print("Encoding queries...")
    embs = emb_model.batch_encode([r['q'] for r in recs], norm=True)

    # ── Helper: extract candidate seeds from query text (substring-match against entity vids) ──
    # Simple: tokenize query, look for 1-4 word n-grams matching entity node text (lowercase)
    print("Building entity-text trie / index for matching...")
    entity_texts_set = set()
    for vid in entity_vids:
        text = vid_to_text.get(vid)
        if text and 1 <= len(text) <= 50:  # filter very long ones (likely passages)
            entity_texts_set.add(text)
    print(f"  Distinct entity texts: {len(entity_texts_set)}")

    def extract_query_seeds(question, max_seeds=10):
        """Find graph entity nodes whose text appears as substring in the question."""
        q_lower = question.lower()
        hits = []
        # Length-sorted descending so longer matches preferred
        # Naive O(|texts|) per query — slow, so prefilter by first-word lookup
        for t in entity_texts_set:
            if t in q_lower:
                hits.append(t)
        # Sort by length desc, dedup substrings
        hits.sort(key=lambda x: -len(x))
        kept = []
        for h in hits:
            if not any(h in k or k in h for k in kept):
                kept.append(h)
            if len(kept) >= max_seeds:
                break
        return kept

    # ── Helper: 3-hop BFS from seed vids on graph ──
    def reachable_within_3_hops(seed_vids):
        """Return set of entity vertex IDs reachable within 3 hops (BFS-3) on graph."""
        reachable = set(seed_vids)
        frontier = set(seed_vids)
        for hop in range(3):
            new_frontier = set()
            for v in frontier:
                for n in g.neighbors(v):
                    if n not in reachable:
                        new_frontier.add(n)
            reachable |= new_frontier
            frontier = new_frontier
        # Filter to entity (non-passage) nodes
        return reachable - passage_idxs

    # ── Main loop: 3-hop pool, with bridges ──
    pool = [i for i in range(1000) if recs[i]['hop'] == 3]
    with_bridges = [i for i in pool if len(recs[i]['bridges']) > 0]
    print(f"\n=== 3-hop pool: {len(pool)} queries  ({len(with_bridges)} with bridges) ===")

    K = 3
    SIM_TH = 0.8
    VOTE_TH = 2

    n_seeds_dist = []
    recall_raw = []
    recall_simvote = []  # sim>0.8 + vote≥2
    recall_graph = []    # graph-reachable filter
    recall_combined = [] # both filters
    n_kept_dist = {'raw': [], 'simvote': [], 'graph': [], 'combined': []}

    print(f"Processing {len(with_bridges)} 3-hop queries with bridges...")
    for ctr, ti in enumerate(with_bridges):
        if ctr % 50 == 0:
            print(f"  {ctr}/{len(with_bridges)}")
        own = set(b.lower() for b in recs[ti]['bridges'])

        # Top-K same-hop NN
        sims = embs @ embs[ti]
        cand = [(i, sims[i]) for i in pool if i != ti]
        cand.sort(key=lambda x: -x[1])
        top = cand[:K]

        # Raw borrowed (union of top-K bridges, lowercase)
        raw_borrowed = set()
        for j, _ in top:
            raw_borrowed.update(b.lower() for b in recs[j]['bridges'])

        # Sim-vote filter
        qualified = [(j, s) for j, s in top if s > SIM_TH]
        vote_counter = Counter()
        for j, _ in qualified:
            for b in set(recs[j]['bridges']):
                vote_counter[b.lower()] += 1
        simvote_borrowed = {b for b, v in vote_counter.items() if v >= VOTE_TH}

        # Graph reachability filter
        seeds_text = extract_query_seeds(recs[ti]['q'])
        n_seeds_dist.append(len(seeds_text))
        seed_vids = [text_to_vid[t] for t in seeds_text if t in text_to_vid]
        if seed_vids:
            reachable_vids = reachable_within_3_hops(seed_vids)
            reachable_texts = {vid_to_text[v] for v in reachable_vids if v in vid_to_text}
        else:
            reachable_texts = set()
        graph_borrowed = raw_borrowed & reachable_texts
        combined_borrowed = simvote_borrowed & reachable_texts

        # Recall computations
        def rec(borrowed):
            return len(own & borrowed) / max(len(own), 1)
        recall_raw.append(rec(raw_borrowed))
        recall_simvote.append(rec(simvote_borrowed))
        recall_graph.append(rec(graph_borrowed))
        recall_combined.append(rec(combined_borrowed))
        n_kept_dist['raw'].append(len(raw_borrowed))
        n_kept_dist['simvote'].append(len(simvote_borrowed))
        n_kept_dist['graph'].append(len(graph_borrowed))
        n_kept_dist['combined'].append(len(combined_borrowed))

    n = len(with_bridges)
    print(f"\n=== Results (3-hop, 285 queries with bridges) ===")
    print()
    print(f"{'Filter':35s} | {'mean recall':>12} | {'mean #kept':>11} | {'>0 kept':>9}")
    print('-' * 80)
    for label, recs_list, kept_key in [
        ('raw top-3 union (no filter)',  recall_raw,      'raw'),
        ('+ sim>0.8 + vote≥2',           recall_simvote,  'simvote'),
        ('+ 3-hop graph reach',          recall_graph,    'graph'),
        ('+ sim/vote + 3-hop graph',     recall_combined, 'combined'),
    ]:
        nz = sum(1 for x in n_kept_dist[kept_key] if x > 0)
        print(f"{label:35s} | {np.mean(recs_list):>12.3f} | {np.mean(n_kept_dist[kept_key]):>11.2f} | {nz}/{n} ({nz/n:.0%})")

    print()
    print(f"Seed extraction stats:")
    print(f"  mean seeds per query: {np.mean(n_seeds_dist):.2f}")
    print(f"  queries with ≥1 seed: {sum(1 for x in n_seeds_dist if x>0)}/{n}")


if __name__ == "__main__":
    main()
