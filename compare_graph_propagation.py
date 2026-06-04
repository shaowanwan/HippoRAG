"""Compare NER graph vs HippoRAG OpenIE graph: propagation breadth on actual queries."""
import json
import os
import sys
import pickle

os.environ['OPENAI_API_KEY'] = 'sk-396199ed7af84eff8a0cf7a71b797601'
sys.path.insert(0, 'src')


def main():
    import numpy as np
    import evaluate_musique_ner_pipeline as ner
    from src.hipporag.embedding_model import _get_embedding_model_class

    # 1. Graph stats
    print("="*70)
    print("GRAPH STATS")
    print("="*70)

    import sys as _sys
    _sys.modules['__main__'].NERIndex = ner.NERIndex
    em_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    emb = _get_embedding_model_class(embedding_model_name=em_name)(embedding_model_name=em_name)

    print("\nLoading NER global graph...")
    ner_idx = ner.NERIndex.load('outputs/musique_ner_pipeline_eval/global_ner_index.pkl',
                                  embedding_model=emb)
    g_ner = ner_idx.graph
    ner_passage_set = set(ner_idx.passage_node_idxs)
    ner_entity_vids = [v for v in range(g_ner.vcount()) if v not in ner_passage_set]
    ner_degrees = [g_ner.degree(v) for v in ner_entity_vids]
    print(f"NER:")
    print(f"  total nodes: {g_ner.vcount()}  (passages: {len(ner_passage_set)}, entities: {len(ner_entity_vids)})")
    print(f"  edges:        {g_ner.ecount()}")
    print(f"  entity degree:  mean={np.mean(ner_degrees):.2f}  median={int(np.median(ner_degrees))}  max={max(ner_degrees)}")

    print("\nLoading HippoRAG global graph...")
    g_hp = pickle.load(open(
        'outputs/musique_hipporag_fullcorpus_qwen-plus_all-MiniLM-L6-v2/'
        'qwen-plus_Transformers_sentence-transformers_all-MiniLM-L6-v2/graph.pickle', 'rb'))
    # Identify passage nodes (HippoRAG uses 'hash_id' starting with 'chunk-' for passages)
    # Look at hash_id attribute
    hash_ids = g_hp.vs['hash_id'] if 'hash_id' in g_hp.vs.attributes() else None
    if hash_ids:
        hp_passage_set = {i for i, h in enumerate(hash_ids) if h.startswith('chunk-')}
        hp_entity_vids = [v for v in range(g_hp.vcount()) if v not in hp_passage_set]
    else:
        hp_passage_set = set()
        hp_entity_vids = list(range(g_hp.vcount()))
    hp_degrees = [g_hp.degree(v) for v in hp_entity_vids]
    print(f"HippoRAG:")
    print(f"  total nodes: {g_hp.vcount()}  (passages: {len(hp_passage_set)}, entities: {len(hp_entity_vids)})")
    print(f"  edges:        {g_hp.ecount()}")
    print(f"  entity degree: mean={np.mean(hp_degrees):.2f}  median={int(np.median(hp_degrees))}  max={max(hp_degrees)}")

    print(f"\nRatio HippoRAG/NER:")
    print(f"  entity count:  {len(hp_entity_vids)/len(ner_entity_vids):.2f}x")
    print(f"  edges:         {g_hp.ecount()/g_ner.ecount():.2f}x")
    print(f"  mean degree:   {np.mean(hp_degrees)/np.mean(ner_degrees):.2f}x")

    # 2. PPR propagation breadth — for a few sample seeds, run PPR and count activated entities
    print("\n" + "="*70)
    print("PPR PROPAGATION BREADTH (uniform seed on 5 random entity, damping=0.5)")
    print("="*70)

    np.random.seed(42)
    # Same number of seeds for both graphs (5 random entity vids)
    K_seeds = 5

    def ppr_top_count(graph, n_total, seed_vids, thresholds=[1e-5, 1e-4, 1e-3]):
        reset = np.zeros(n_total)
        for v in seed_vids:
            reset[v] = 1.0
        reset /= reset.sum()
        try:
            scores = graph.personalized_pagerank(
                vertices=range(n_total),
                damping=0.5,
                directed=False,
                weights='weight' if 'weight' in graph.es.attributes() else None,
                reset=reset,
                implementation='prpack',
            )
            scores = np.array(scores)
            counts = {th: int((scores > th).sum()) for th in thresholds}
            return scores, counts
        except Exception as e:
            print(f"  PPR failed: {e}")
            return None, None

    print(f"\nRunning PPR on each graph with random {K_seeds} seeds (10 trials)...")
    ner_counts = {1e-5: [], 1e-4: [], 1e-3: []}
    hp_counts = {1e-5: [], 1e-4: [], 1e-3: []}
    for trial in range(10):
        ner_seeds = list(np.random.choice(ner_entity_vids, K_seeds, replace=False))
        hp_seeds = list(np.random.choice(hp_entity_vids, K_seeds, replace=False))
        _, nc = ppr_top_count(g_ner, g_ner.vcount(), ner_seeds)
        _, hc = ppr_top_count(g_hp, g_hp.vcount(), hp_seeds)
        if nc:
            for th, v in nc.items(): ner_counts[th].append(v)
        if hc:
            for th, v in hc.items(): hp_counts[th].append(v)

    print(f"\nNodes with PPR score > threshold (averaged over 10 trials):")
    print(f"{'threshold':>10} | {'NER':>10} | {'HippoRAG':>10} | {'HP/NER':>7}")
    print('-' * 50)
    for th in [1e-5, 1e-4, 1e-3]:
        n_mean = np.mean(ner_counts[th]) if ner_counts[th] else 0
        h_mean = np.mean(hp_counts[th]) if hp_counts[th] else 0
        ratio = h_mean / max(n_mean, 1)
        print(f"  > {th:.0e} | {n_mean:>10.1f} | {h_mean:>10.1f} | {ratio:>6.2f}x")

    print(f"\n(Higher count = wider propagation. HP/NER > 1 means HippoRAG activates more entities at that threshold.)")


if __name__ == "__main__":
    main()
