"""Recompute Recall@K for HippoRAG+bridge 1000 result using saved doc IDs.

The original run had a recall bug (gold docs format vs global corpus format mismatch).
Doc IDs were saved in round_diagnostics.rrf_top_doc_ids; we look them up against the
musique_corpus.json and match to gold supporting paragraphs from musique.json.

Adds: per-sample {baseline_recall_fixed, bridge_recall_fixed}
       summary  {baseline_recall_fixed, bridge_recall_fixed}
       config  {recall_recomputed: true}
"""
import json
import numpy as np


def main():
    res_path = 'outputs/musique_hipporag_bridge_global_FIXED_eval/comparison_results.json'
    corpus_path = 'reproduce/dataset/musique_corpus.json'
    musique_path = 'musique.json'

    print("Loading data...")
    d = json.load(open(res_path))
    corpus = json.load(open(corpus_path))  # [{title, text}, ...]
    musique = json.load(open(musique_path))

    # Build doc_id → text (without title) for matching
    corpus_text = [c['text'].strip() for c in corpus]
    print(f"Corpus: {len(corpus)} docs")

    KS = [1, 2, 5, 10, 20]
    results = d['results']

    def recall_at_k(retrieved_ids, gold_texts, k):
        """Fraction of gold_texts hit within top-k retrieved (substring or text match)."""
        if not gold_texts:
            return 0.0
        retrieved_top_k = retrieved_ids[:k]
        retrieved_texts = [corpus_text[idx] for idx in retrieved_top_k if 0 <= idx < len(corpus_text)]
        hit = 0
        for g in gold_texts:
            g_stripped = g.strip()
            for r_text in retrieved_texts:
                # Try exact text match (corpus is 'text' without title; gold is paragraph_text)
                if r_text == g_stripped:
                    hit += 1
                    break
                # Fallback: substring (corpus text contains gold OR vice versa)
                if g_stripped in r_text or r_text in g_stripped:
                    hit += 1
                    break
        return hit / len(gold_texts)

    # Per-sample
    base_recalls = {k: [] for k in KS}
    bridge_recalls = {k: [] for k in KS}
    n_with_gold = 0

    for r in results:
        idx = r['idx']
        sample = musique[idx]
        gold_texts = [p['paragraph_text'].strip() for p in sample['paragraphs']
                      if p.get('is_supporting', False)]
        if not gold_texts:
            continue
        n_with_gold += 1
        rds = r.get('round_diagnostics') or []
        if not rds:
            for k in KS:
                base_recalls[k].append(0.0); bridge_recalls[k].append(0.0)
            continue
        # Baseline = round 0 base_top_doc_ids
        base_ids = rds[0].get('base_top_doc_ids', [])
        # Bridge (final method) = last round's rrf_top_doc_ids
        bridge_ids = rds[-1].get('rrf_top_doc_ids', [])

        per_sample_base = {}
        per_sample_bridge = {}
        for k in KS:
            br = recall_at_k(base_ids, gold_texts, k)
            ir = recall_at_k(bridge_ids, gold_texts, k)
            base_recalls[k].append(br)
            bridge_recalls[k].append(ir)
            per_sample_base[f'Recall@{k}'] = round(br, 4)
            per_sample_bridge[f'Recall@{k}'] = round(ir, 4)
        r['baseline_recall_fixed'] = per_sample_base
        r['bridge_recall_fixed'] = per_sample_bridge

    print(f"\nProcessed {n_with_gold}/{len(results)} samples with gold supporting paragraphs")

    print("\nRecomputed Recall:")
    print(f"{'K':>5} | {'Baseline':>10} | {'Bridge':>10} | {'Δ':>8}")
    print('-' * 50)
    for k in KS:
        b = np.mean(base_recalls[k])
        i = np.mean(bridge_recalls[k])
        d_k = i - b
        print(f"R@{k:<3} | {b:>10.4f} | {i:>10.4f} | {d_k:>+8.4f}")

    # Update summary
    summary = d.setdefault('summary', {})
    summary['baseline_recall_fixed'] = {
        f'Recall@{k}': round(float(np.mean(base_recalls[k])), 4) for k in KS
    }
    summary['bridge_recall_fixed'] = {
        f'Recall@{k}': round(float(np.mean(bridge_recalls[k])), 4) for k in KS
    }
    summary['recall_recompute_note'] = (
        "Recall was 0 in original run due to format mismatch between gold_docs "
        "(paragraph_text only) and global corpus ('title\\ntext'). Recomputed here "
        "from saved round_diagnostics doc IDs vs musique_corpus.json text fields "
        "(exact + substring match). EM/F1 in original run are unaffected."
    )

    config = d.setdefault('config', {})
    config['recall_recomputed'] = True

    # Save in place
    with open(res_path, 'w') as f:
        json.dump(d, f, indent=2)
    print(f"\nUpdated: {res_path}")


if __name__ == "__main__":
    main()
