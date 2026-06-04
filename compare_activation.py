"""Compare HippoRAG vs NER entity activation: diversity, sparsity, mass distribution.

For 50 random queries, run both methods' seed computation (before PPR), record entity-side
node weights. No PPR, no LLM — just the seed vector that goes into PPR.

Stats:
  - # entity nodes activated (weight > 0)
  - # entity nodes "significantly activated" (weight > 0.01)
  - top-1 / top-5 / top-20 mass fraction (concentration)
  - Entropy of entity weight distribution
"""
import json
import os
import sys
import pickle

os.environ['OPENAI_API_KEY'] = 'sk-396199ed7af84eff8a0cf7a71b797601'
sys.path.insert(0, 'src')


def main():
    import numpy as np
    import evaluate_musique_ner_pipeline as ner

    # Pickle compat
    import sys as _sys
    _sys.modules['__main__'].NERIndex = ner.NERIndex

    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.embedding_model import _get_embedding_model_class

    em_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    emb = _get_embedding_model_class(embedding_model_name=em_name)(embedding_model_name=em_name)

    # ── Load NER global index ──
    print("Loading NER global index...")
    ner_idx = ner.NERIndex.load('outputs/musique_ner_pipeline_eval/global_ner_index.pkl',
                                  embedding_model=emb)
    ner_passage_set = set(ner_idx.passage_node_idxs)
    print(f"  NER graph: {ner_idx.graph.vcount()} nodes, {ner_idx.graph.ecount()} edges")
    print(f"  NER entity nodes: {ner_idx.graph.vcount() - len(ner_passage_set)}")

    # ── Load HippoRAG global instance ──
    print("Loading HippoRAG global...")
    hipporag = HippoRAG(
        save_dir='outputs/musique_hipporag_fullcorpus_qwen-plus_all-MiniLM-L6-v2',
        llm_model_name='qwen-plus',
        embedding_model_name=em_name,
        llm_base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        embedding_model=emb,
    )
    # Need to index from corpus
    corpus = json.load(open('reproduce/dataset/musique_corpus.json'))
    full_docs = [f"{c['title']}\n{c['text']}" for c in corpus]
    hipporag.index(docs=full_docs)
    hipporag.prepare_retrieval_objects()
    hp_passage_set = set(hipporag.passage_node_idxs)
    print(f"  HippoRAG graph: {hipporag.graph.vcount()} nodes, {hipporag.graph.ecount()} edges")
    print(f"  HippoRAG entity nodes: {hipporag.graph.vcount() - len(hp_passage_set)}")

    # ── Pick 50 sample queries ──
    np.random.seed(42)
    sample = json.load(open('musique.json'))
    qs = np.random.choice(1000, 50, replace=False)

    # ── For each query, get NER + HippoRAG seed vectors (entity portion) ──
    def stats(weights, passage_set, label):
        entity_idxs = [v for v in range(len(weights)) if v not in passage_set]
        e_weights = np.array([weights[v] for v in entity_idxs])
        n_activated = int((e_weights > 0).sum())
        n_significant = int((e_weights > 0.01).sum())
        n_strong = int((e_weights > 0.1).sum())
        total_mass = float(e_weights.sum())
        if total_mass > 0:
            sorted_w = np.sort(e_weights)[::-1]
            top1 = sorted_w[0] / total_mass
            top5 = sorted_w[:5].sum() / total_mass
            top20 = sorted_w[:20].sum() / total_mass
            # Normalized entropy (Shannon, base e, divided by log(n_activated) → 0-1)
            p = e_weights / total_mass
            p = p[p > 0]
            entropy = -float((p * np.log(p)).sum())
            norm_entropy = entropy / np.log(max(n_activated, 2)) if n_activated > 1 else 0
        else:
            top1 = top5 = top20 = entropy = norm_entropy = 0
        return {
            'n_activated': n_activated,
            'n_significant': n_significant,
            'n_strong': n_strong,
            'total_mass': total_mass,
            'top1_frac': top1, 'top5_frac': top5, 'top20_frac': top20,
            'norm_entropy': norm_entropy,
        }

    ner_stats_all = []
    hp_stats_all = []
    print(f"\nProcessing {len(qs)} queries...")
    for ci, qi in enumerate(qs):
        question = sample[qi]['question']
        # NER seeds
        ner_w = ner_idx._compute_node_weights(query=question)
        ner_stats_all.append(stats(ner_w, ner_passage_set, 'NER'))

        # HippoRAG seeds — replicate graph_search_with_fact_entities flow but stop before PPR
        hipporag.get_query_embeddings([question])
        fact_scores = hipporag.get_fact_scores(question)
        if len(fact_scores) == 0:
            hp_stats_all.append({k: 0 for k in ner_stats_all[-1]})
            continue
        top_k_fact_indices, top_k_facts, _ = hipporag.rerank_facts(question, fact_scores)
        if len(top_k_facts) == 0:
            hp_stats_all.append({k: 0 for k in ner_stats_all[-1]})
            continue
        # Compute node weights without running PPR — replicate the math
        # Easiest path: call graph_search and capture node_weights via return_node_weights
        try:
            _, _, hp_w = hipporag.graph_search_with_fact_entities(
                query=question,
                link_top_k=hipporag.global_config.linking_top_k,
                query_fact_scores=fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=hipporag.global_config.passage_node_weight,
                return_node_weights=True,
            )
            hp_stats_all.append(stats(hp_w, hp_passage_set, 'HippoRAG'))
        except Exception as e:
            print(f"  HP query {qi} failed: {e}")
            hp_stats_all.append({k: 0 for k in ner_stats_all[-1]})

        if (ci + 1) % 10 == 0:
            print(f"  {ci+1}/{len(qs)}")

    # Aggregate
    def agg(stats_list, key):
        return float(np.mean([s[key] for s in stats_list if isinstance(s, dict)]))

    print("\n" + "="*70)
    print("ENTITY ACTIVATION COMPARISON (mean over 50 random queries)")
    print("="*70)
    print(f"{'metric':35s} | {'NER':>10} | {'HippoRAG':>10}")
    print('-' * 60)
    for k in ['n_activated', 'n_significant', 'n_strong']:
        n = agg(ner_stats_all, k); h = agg(hp_stats_all, k)
        print(f"{k:35s} | {n:>10.1f} | {h:>10.1f}")
    print()
    for k in ['total_mass', 'top1_frac', 'top5_frac', 'top20_frac', 'norm_entropy']:
        n = agg(ner_stats_all, k); h = agg(hp_stats_all, k)
        print(f"{k:35s} | {n:>10.4f} | {h:>10.4f}")
    print()
    print("Interpretation key:")
    print("  more 'n_activated' = more entities have non-zero seed (diversity)")
    print("  more 'norm_entropy' = more even mass spread (sparsity inversely)")
    print("  higher 'top1/top5_frac' = mass more concentrated at top (less diverse)")


if __name__ == "__main__":
    main()
