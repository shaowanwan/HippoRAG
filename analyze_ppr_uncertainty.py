"""Analyze PPR distribution uncertainty and query-subgraph semantic coverage.

For each sample:
1. Load graph + embeddings directly (no HippoRAG init)
2. Compute PPR with base seeds
3. Compute entropy, concentration, query-subgraph coverage
4. Correlate with retrieval success
"""

import json
import os
import sys
import pickle
import numpy as np
from scipy import stats
import igraph as ig

sys.path.insert(0, os.path.dirname(__file__))


def load_embeddings(emb_dir):
    """Load embeddings from a store directory."""
    emb_file = os.path.join(emb_dir, "embeddings.npy")
    if os.path.exists(emb_file):
        return np.load(emb_file)
    # Try loading from individual files
    keys_file = os.path.join(emb_dir, "keys.json")
    if os.path.exists(keys_file):
        with open(keys_file) as f:
            keys = json.load(f)
        embs = []
        for k in keys:
            e = np.load(os.path.join(emb_dir, f"{k}.npy"))
            embs.append(e)
        return np.array(embs)
    return None


def compute_ppr_entropy(ppr_scores, entity_idxs):
    """Entropy of PPR distribution over entity nodes."""
    entity_scores = np.array([ppr_scores[i] for i in entity_idxs])
    entity_scores = entity_scores[entity_scores > 0]
    if len(entity_scores) == 0:
        return 0.0
    p = entity_scores / entity_scores.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_ppr_concentration(ppr_scores, entity_idxs, top_k=10):
    """Fraction of PPR mass in top-K entities."""
    entity_scores = np.array([ppr_scores[i] for i in entity_idxs])
    total = entity_scores.sum()
    if total == 0:
        return 0.0
    top_k_sum = np.sort(entity_scores)[-top_k:].sum()
    return float(top_k_sum / total)


def compute_query_coverage(query_emb, entity_embs, ppr_scores, entity_idxs, top_k=20):
    """Avg cosine similarity between query and top PPR-activated entities."""
    entity_scores = np.array([ppr_scores[i] for i in entity_idxs])
    sorted_idx = np.argsort(entity_scores)[-top_k:]
    sorted_idx = sorted_idx[entity_scores[sorted_idx] > 0]

    if len(sorted_idx) == 0:
        return 0.0, 0.0, 1.0  # avg, max, gap

    top_embs = entity_embs[sorted_idx]
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    emb_norms = top_embs / (np.linalg.norm(top_embs, axis=1, keepdims=True) + 1e-12)
    sims = np.dot(emb_norms, query_norm)

    # Coverage gap: distance between query and PPR-weighted centroid
    weights = entity_scores[sorted_idx]
    weights = weights / weights.sum()
    centroid = np.average(top_embs, axis=0, weights=weights)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    gap = 1.0 - float(np.dot(query_norm, centroid_norm))

    return float(np.mean(sims)), float(np.max(sims)), gap


def main():
    results_file = "outputs/musique_reasoning_eval/comparison_results.json"
    with open(results_file) as f:
        all_results = json.load(f)

    sample_base = "outputs/musique_reasoning_eval"
    model_dir = "qwen-plus_Transformers_sentence-transformers_all-MiniLM-L6-v2"

    # Load embedding model for query encoding
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    analysis_data = []

    for item in all_results["results"]:
        idx = item["idx"]
        sample_dir = os.path.join(sample_base, f"sample_{idx:06d}", model_dir)

        if not os.path.exists(sample_dir):
            continue

        # Load graph
        graph_path = os.path.join(sample_dir, "graph.pickle")
        if not os.path.exists(graph_path):
            continue

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        # Load node mapping
        node_map_path = os.path.join(sample_dir, "node_name_to_vertex_idx.json")
        if os.path.exists(node_map_path):
            with open(node_map_path) as f:
                node_map = json.load(f)
        else:
            node_map = {v["name"]: v.index for v in graph.vs}

        # Load entity embeddings from parquet
        import pandas as pd
        entity_parquet = os.path.join(sample_dir, "entity_embeddings", "vdb_entity.parquet")
        if not os.path.exists(entity_parquet):
            continue
        entity_df = pd.read_parquet(entity_parquet)
        entity_embs = np.array(entity_df["embedding"].tolist())
        entity_keys = entity_df["hash_id"].tolist()

        # Build entity_idxs from parquet keys matching graph nodes
        entity_idxs = []
        emb_idx_map = []  # maps position in entity_idxs to position in entity_embs
        for ei, key in enumerate(entity_keys):
            vid = node_map.get(key)
            if vid is not None:
                entity_idxs.append(vid)
                emb_idx_map.append(ei)

        # Encode query
        query = item["question"]
        query_emb = embed_model.encode(query, normalize_embeddings=True)

        # Build simple PPR seeds: uniform weight on entity nodes connected to query-relevant facts
        # For simplicity, use uniform seeds on all entity nodes (we just want the PPR distribution shape)
        # Actually, use DPR-like approach: find entities most similar to query
        # Use only matched embeddings
        entity_embs_matched = entity_embs[emb_idx_map]

        if len(entity_idxs) == 0:
            continue

        entity_embs_norm = entity_embs_matched / (np.linalg.norm(entity_embs_matched, axis=1, keepdims=True) + 1e-12)
        query_entity_sims = np.dot(entity_embs_norm, query_emb)

        # Top-5 most similar entities as seeds (mimics fact matching)
        top_seed_idx = np.argsort(query_entity_sims)[-5:]
        seed_weights = np.zeros(graph.vcount())
        for si in top_seed_idx:
            vid = entity_idxs[si]
            seed_weights[vid] = max(0, query_entity_sims[si])

        if seed_weights.sum() == 0:
            continue

        # Normalize
        seed_weights = seed_weights / seed_weights.sum()

        # Run PPR
        try:
            ppr_scores = graph.personalized_pagerank(
                vertices=range(graph.vcount()),
                damping=0.5,
                directed=False,
                weights='weight',
                reset=seed_weights.tolist(),
                implementation='prpack',
            )
            ppr_scores = np.array(ppr_scores)
        except Exception as e:
            continue

        # Compute metrics
        entropy = compute_ppr_entropy(ppr_scores, entity_idxs)
        concentration = compute_ppr_concentration(ppr_scores, entity_idxs, top_k=10)
        avg_sim, max_sim, coverage_gap = compute_query_coverage(
            query_emb, entity_embs_matched, ppr_scores, entity_idxs, top_k=20
        )

        # Number of activated entities (PPR > threshold)
        activated = sum(1 for i in entity_idxs if ppr_scores[i] > 0.001)

        # Retrieval outcomes
        baseline_r5 = item["baseline_retrieval"].get("Recall@5", 0)
        reasoning_r5 = item["reasoning_retrieval"].get("final", {}).get("Recall@5", 0)

        # EM
        gold = item["gold_answer"].strip().lower()
        baseline_em = 1 if item.get("baseline_answer", "").strip().lower() == gold else 0
        reasoning_em = 1 if item.get("reasoning_answer", "").strip().lower() == gold else 0

        row = {
            "idx": idx,
            "entropy": round(entropy, 4),
            "concentration": round(concentration, 4),
            "avg_sim": round(avg_sim, 4),
            "max_sim": round(max_sim, 4),
            "coverage_gap": round(coverage_gap, 4),
            "activated_entities": activated,
            "baseline_r5": baseline_r5,
            "reasoning_r5": reasoning_r5,
            "r5_improved": round(reasoning_r5 - baseline_r5, 4),
            "baseline_em": baseline_em,
            "reasoning_em": reasoning_em,
        }
        analysis_data.append(row)

        if len(analysis_data) % 20 == 0:
            print(f"Processed {len(analysis_data)} samples...")

    print(f"\nTotal analyzed: {len(analysis_data)} samples")
    print("=" * 70)

    if len(analysis_data) < 10:
        print("Too few samples, aborting analysis.")
        return

    # Convert to arrays
    entropies = np.array([d["entropy"] for d in analysis_data])
    concentrations = np.array([d["concentration"] for d in analysis_data])
    avg_sims = np.array([d["avg_sim"] for d in analysis_data])
    coverage_gaps = np.array([d["coverage_gap"] for d in analysis_data])
    activated = np.array([d["activated_entities"] for d in analysis_data])
    baseline_r5s = np.array([d["baseline_r5"] for d in analysis_data])
    reasoning_r5s = np.array([d["reasoning_r5"] for d in analysis_data])
    r5_improvements = np.array([d["r5_improved"] for d in analysis_data])
    baseline_ems = np.array([d["baseline_em"] for d in analysis_data])
    reasoning_ems = np.array([d["reasoning_em"] for d in analysis_data])

    print("\n=== PPR DISTRIBUTION STATISTICS ===")
    print(f"Entropy:        mean={entropies.mean():.3f}, std={entropies.std():.3f}, range=[{entropies.min():.3f}, {entropies.max():.3f}]")
    print(f"Concentration:  mean={concentrations.mean():.3f}, std={concentrations.std():.3f}")
    print(f"Avg similarity: mean={avg_sims.mean():.3f}, std={avg_sims.std():.3f}")
    print(f"Coverage gap:   mean={coverage_gaps.mean():.3f}, std={coverage_gaps.std():.3f}")
    print(f"Activated ents: mean={activated.mean():.1f}, std={activated.std():.1f}")

    print("\n=== CORRELATION WITH BASELINE R@5 ===")
    for name, values in [("Entropy", entropies), ("Concentration", concentrations),
                          ("Avg_sim", avg_sims), ("Coverage_gap", coverage_gaps),
                          ("Activated", activated.astype(float))]:
        r, p = stats.pearsonr(values, baseline_r5s)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:15s} vs baseline_R@5:  r={r:+.3f}, p={p:.4f} {sig}")

    print("\n=== CORRELATION WITH R@5 IMPROVEMENT ===")
    for name, values in [("Entropy", entropies), ("Concentration", concentrations),
                          ("Avg_sim", avg_sims), ("Coverage_gap", coverage_gaps),
                          ("Activated", activated.astype(float))]:
        r, p = stats.pearsonr(values, r5_improvements)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:15s} vs R@5_improve:   r={r:+.3f}, p={p:.4f} {sig}")

    print("\n=== CORRELATION WITH REASONING R@5 ===")
    for name, values in [("Entropy", entropies), ("Concentration", concentrations),
                          ("Avg_sim", avg_sims), ("Coverage_gap", coverage_gaps),
                          ("Activated", activated.astype(float))]:
        r, p = stats.pearsonr(values, reasoning_r5s)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:15s} vs reasoning_R@5: r={r:+.3f}, p={p:.4f} {sig}")

    # Group analysis
    median_entropy = np.median(entropies)
    high_ent = entropies >= median_entropy
    low_ent = ~high_ent

    print(f"\n=== GROUP: ENTROPY (median={median_entropy:.3f}) ===")
    print(f"High entropy ({high_ent.sum():3d}): base_R@5={baseline_r5s[high_ent].mean():.3f}, reas_R@5={reasoning_r5s[high_ent].mean():.3f}, improve={r5_improvements[high_ent].mean():+.3f}, base_EM={baseline_ems[high_ent].mean():.3f}, reas_EM={reasoning_ems[high_ent].mean():.3f}")
    print(f"Low entropy  ({low_ent.sum():3d}): base_R@5={baseline_r5s[low_ent].mean():.3f}, reas_R@5={reasoning_r5s[low_ent].mean():.3f}, improve={r5_improvements[low_ent].mean():+.3f}, base_EM={baseline_ems[low_ent].mean():.3f}, reas_EM={reasoning_ems[low_ent].mean():.3f}")

    median_gap = np.median(coverage_gaps)
    high_gap = coverage_gaps >= median_gap
    low_gap = ~high_gap

    print(f"\n=== GROUP: COVERAGE GAP (median={median_gap:.3f}) ===")
    print(f"High gap ({high_gap.sum():3d}): base_R@5={baseline_r5s[high_gap].mean():.3f}, reas_R@5={reasoning_r5s[high_gap].mean():.3f}, improve={r5_improvements[high_gap].mean():+.3f}, base_EM={baseline_ems[high_gap].mean():.3f}, reas_EM={reasoning_ems[high_gap].mean():.3f}")
    print(f"Low gap  ({low_gap.sum():3d}): base_R@5={baseline_r5s[low_gap].mean():.3f}, reas_R@5={reasoning_r5s[low_gap].mean():.3f}, improve={r5_improvements[low_gap].mean():+.3f}, base_EM={baseline_ems[low_gap].mean():.3f}, reas_EM={reasoning_ems[low_gap].mean():.3f}")

    # Quartile analysis for coverage gap
    q25, q50, q75 = np.percentile(coverage_gaps, [25, 50, 75])
    print(f"\n=== QUARTILE: COVERAGE GAP (q25={q25:.3f}, q50={q50:.3f}, q75={q75:.3f}) ===")
    for lo, hi, label in [(0, q25, "Q1 (lowest gap)"), (q25, q50, "Q2"), (q50, q75, "Q3"), (q75, 2.0, "Q4 (highest gap)")]:
        mask = (coverage_gaps >= lo) & (coverage_gaps < hi)
        if mask.sum() == 0:
            continue
        print(f"  {label:18s} ({mask.sum():3d}): base_R@5={baseline_r5s[mask].mean():.3f}, reas_R@5={reasoning_r5s[mask].mean():.3f}, improve={r5_improvements[mask].mean():+.3f}")

    # Save
    output_file = "outputs/musique_reasoning_eval/ppr_uncertainty_analysis.json"
    with open(output_file, "w") as f:
        json.dump(analysis_data, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
