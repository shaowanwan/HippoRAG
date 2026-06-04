"""Compare HippoRAG vs NER POST-PPR node distribution.

For 50 random queries:
  1. Run each method's seed computation (same as before).
  2. Run PPR on each graph with those seeds.
  3. Count nodes with PPR score > threshold (entity nodes only).
  4. Report distribution shape.
"""
import json
import os
import sys

os.environ['OPENAI_API_KEY'] = 'sk-396199ed7af84eff8a0cf7a71b797601'
sys.path.insert(0, 'src')


def main():
    import numpy as np
    import evaluate_musique_ner_pipeline as ner
    import sys as _sys
    _sys.modules['__main__'].NERIndex = ner.NERIndex

    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.embedding_model import _get_embedding_model_class

    em_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    emb = _get_embedding_model_class(embedding_model_name=em_name)(embedding_model_name=em_name)

    print("Loading NER global index...")
    ner_idx = ner.NERIndex.load('outputs/musique_ner_pipeline_eval/global_ner_index.pkl',
                                  embedding_model=emb)
    ner_passage_set = set(ner_idx.passage_node_idxs)

    print("Loading HippoRAG global...")
    hipporag = HippoRAG(
        save_dir='outputs/musique_hipporag_fullcorpus_qwen-plus_all-MiniLM-L6-v2',
        llm_model_name='qwen-plus',
        embedding_model_name=em_name,
        llm_base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        embedding_model=emb,
    )
    corpus = json.load(open('reproduce/dataset/musique_corpus.json'))
    full_docs = [f"{c['title']}\n{c['text']}" for c in corpus]
    hipporag.index(docs=full_docs)
    hipporag.prepare_retrieval_objects()
    hp_passage_set = set(hipporag.passage_node_idxs)

    print(f"NER graph: {ner_idx.graph.vcount()} nodes")
    print(f"HippoRAG graph: {hipporag.graph.vcount()} nodes")

    np.random.seed(42)
    sample = json.load(open('musique.json'))
    qs = np.random.choice(1000, 50, replace=False)

    def run_ppr_count(graph, reset, passage_set, thresholds):
        reset = np.array(reset, dtype=float)
        reset = np.where(np.isnan(reset) | (reset < 0), 0, reset)
        if reset.sum() == 0:
            return None
        reset /= reset.sum()
        try:
            scores = graph.personalized_pagerank(
                vertices=range(graph.vcount()),
                damping=0.5,
                directed=False,
                weights='weight' if 'weight' in graph.es.attributes() else None,
                reset=reset,
                implementation='prpack',
            )
        except Exception as e:
            return None
        scores = np.array(scores)
        # Count entity nodes (not passage) above threshold
        entity_mask = np.array([v not in passage_set for v in range(graph.vcount())])
        entity_scores = scores[entity_mask]
        counts = {th: int((entity_scores > th).sum()) for th in thresholds}
        # Mass distribution: top-k mass fraction (among entities)
        sorted_e = np.sort(entity_scores)[::-1]
        total = entity_scores.sum()
        if total > 0:
            top1_frac = sorted_e[0] / total
            top10_frac = sorted_e[:10].sum() / total
            top100_frac = sorted_e[:100].sum() / total
        else:
            top1_frac = top10_frac = top100_frac = 0
        return counts, top1_frac, top10_frac, top100_frac

    thresholds = [1e-6, 1e-5, 1e-4, 1e-3]
    ner_count_all = {th: [] for th in thresholds}
    hp_count_all = {th: [] for th in thresholds}
    ner_top_all = {'1': [], '10': [], '100': []}
    hp_top_all = {'1': [], '10': [], '100': []}

    print(f"\nRunning PPR for {len(qs)} queries on each graph...")
    for ci, qi in enumerate(qs):
        question = sample[qi]['question']
        # NER
        ner_w = ner_idx._compute_node_weights(query=question)
        res_n = run_ppr_count(ner_idx.graph, ner_w, ner_passage_set, thresholds)
        if res_n:
            counts, t1, t10, t100 = res_n
            for th, v in counts.items(): ner_count_all[th].append(v)
            ner_top_all['1'].append(t1); ner_top_all['10'].append(t10); ner_top_all['100'].append(t100)

        # HippoRAG
        hipporag.get_query_embeddings([question])
        fact_scores = hipporag.get_fact_scores(question)
        if len(fact_scores) > 0:
            top_k_fact_indices, top_k_facts, _ = hipporag.rerank_facts(question, fact_scores)
            if len(top_k_facts) > 0:
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
                    res_h = run_ppr_count(hipporag.graph, hp_w, hp_passage_set, thresholds)
                    if res_h:
                        counts, t1, t10, t100 = res_h
                        for th, v in counts.items(): hp_count_all[th].append(v)
                        hp_top_all['1'].append(t1); hp_top_all['10'].append(t10); hp_top_all['100'].append(t100)
                except Exception as e:
                    pass

        if (ci + 1) % 10 == 0:
            print(f"  {ci+1}/{len(qs)}")

    # Report
    print("\n" + "="*70)
    print("POST-PPR ENTITY NODE COUNT > threshold (mean over 50 queries)")
    print("="*70)
    print(f"{'threshold':>12} | {'NER':>10} | {'HippoRAG':>10} | {'HP/NER':>8}")
    print('-' * 50)
    for th in thresholds:
        n = float(np.mean(ner_count_all[th])) if ner_count_all[th] else 0
        h = float(np.mean(hp_count_all[th])) if hp_count_all[th] else 0
        ratio = h / max(n, 1)
        print(f"  > {th:.0e} | {n:>10.1f} | {h:>10.1f} | {ratio:>7.2f}x")

    print("\nPost-PPR mass concentration (entity-only, mean over 50 queries):")
    print(f"{'metric':>20} | {'NER':>10} | {'HippoRAG':>10}")
    print('-' * 50)
    for k, vs_ner, vs_hp in [
        ('top1 frac', ner_top_all['1'], hp_top_all['1']),
        ('top10 frac', ner_top_all['10'], hp_top_all['10']),
        ('top100 frac', ner_top_all['100'], hp_top_all['100']),
    ]:
        n = float(np.mean(vs_ner)) if vs_ner else 0
        h = float(np.mean(vs_hp)) if vs_hp else 0
        print(f"{k:>20} | {n:>10.4f} | {h:>10.4f}")


if __name__ == "__main__":
    main()
