"""Graph-side distribution drift: Full vs Naive (qwen-plus MuSiQue 1000).

Measures how much the retrieved doc set CHANGES across rounds. If bridge
re-seeding activates new graph regions, Full should have HIGHER drift
(less overlap between round-R and round-0 retrievals) than Naive.

Metrics:
1. Jaccard(top5_round_R, top5_round_0): lower = more drift
2. Δgold@5 from round 0 to round R: how many NEW gold docs added
3. Region-novelty: fraction of round R top-5 that's NOT in round 0 top-5
"""
import json
import numpy as np

FULL = 'outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260402_235319_34587.json'
NAIVE = 'outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260403_010535_35866.json'


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b: return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def overlap(a, b):
    """|a ∩ b| / |a| — what fraction of a is in b."""
    a, b = set(a), set(b)
    return len(a & b) / len(a) if a else 0.0


def analyze(d, label, n=1000):
    """Return per-round arrays of (jaccard_with_r0, novelty_frac, gold_r5)."""
    metrics = {1: [], 2: [], 3: []}
    novelty = {1: [], 2: [], 3: []}
    gold_recall_per_round = {0: [], 1: [], 2: [], 3: []}

    for r in d['results'][:n]:
        rounds = r.get('round_diagnostics', [])
        if not rounds: continue
        # Round 0 top-5
        r0_top5 = rounds[0].get('rrf_top_doc_ids', [])[:5]
        gold_recall_per_round[0].append(rounds[0].get('rrf_recall', {}).get('R@5', 0))

        for ridx in [1, 2, 3]:
            if ridx < len(rounds):
                rR_top5 = rounds[ridx].get('rrf_top_doc_ids', [])[:5]
                metrics[ridx].append(jaccard(r0_top5, rR_top5))
                # Novelty = |round R top5 NOT in round 0 top5| / 5
                new_docs = set(rR_top5) - set(r0_top5)
                novelty[ridx].append(len(new_docs) / 5)
                gold_recall_per_round[ridx].append(rounds[ridx].get('rrf_recall', {}).get('R@5', 0))

    return metrics, novelty, gold_recall_per_round


def main():
    full = json.load(open(FULL))
    naive = json.load(open(NAIVE))
    print(f"Full PID 34587 n={full['summary']['n_completed']}, Naive PID 35866 n={naive['summary']['n_completed']}")

    f_jaccard, f_novelty, f_gold = analyze(full, 'Full')
    n_jaccard, n_novelty, n_gold = analyze(naive, 'Naive')

    print(f"\n=== Jaccard(round R top-5, round 0 top-5)  [lower = more region drift] ===\n")
    print(f"{'Method':<10} | round 1 | round 2 | round 3")
    print('-' * 60)
    for label, j in [('Full', f_jaccard), ('Naive', n_jaccard)]:
        cells = [f"{np.mean(j[r]):.3f} (n={len(j[r])})" if j[r] else "—" for r in [1,2,3]]
        print(f"{label:<10} | {' | '.join(cells)}")
    print()
    for r in [1,2,3]:
        if f_jaccard[r] and n_jaccard[r]:
            df = np.mean(f_jaccard[r]) - np.mean(n_jaccard[r])
            print(f"  Δ Jaccard (Full − Naive) round {r}: {df:+.3f}")

    print(f"\n=== Novelty rate (fraction of round R top-5 NOT in round 0 top-5)  [higher = more region activation] ===\n")
    print(f"{'Method':<10} | round 1 | round 2 | round 3")
    print('-' * 60)
    for label, nv in [('Full', f_novelty), ('Naive', n_novelty)]:
        cells = [f"{np.mean(nv[r]):.3f} (n={len(nv[r])})" if nv[r] else "—" for r in [1,2,3]]
        print(f"{label:<10} | {' | '.join(cells)}")
    print()
    for r in [1,2,3]:
        if f_novelty[r] and n_novelty[r]:
            df = np.mean(f_novelty[r]) - np.mean(n_novelty[r])
            print(f"  Δ Novelty (Full − Naive) round {r}: {df:+.3f}")

    print(f"\n=== Per-round R@5 (gold recall trajectory) ===\n")
    print(f"{'Method':<10} | round 0 | round 1 | round 2 | round 3")
    print('-' * 70)
    for label, g in [('Full', f_gold), ('Naive', n_gold)]:
        cells = [f"{np.mean(g[r]):.3f}" if g[r] else "—" for r in [0,1,2,3]]
        print(f"{label:<10} | {' | '.join(cells)}")

    print(f"\n=== Round R gold gain over round 0 (R@5 round_R - R@5 round_0) ===\n")
    print(f"{'Method':<10} | gain r1 | gain r2 | gain r3")
    print('-' * 60)
    for label, g in [('Full', f_gold), ('Naive', n_gold)]:
        if not g[0]: continue
        r0_mean = np.mean(g[0])
        cells = [f"{np.mean(g[r])-r0_mean:+.3f}" if g[r] else "—" for r in [1,2,3]]
        print(f"{label:<10} | {' | '.join(cells)}")


if __name__ == "__main__":
    main()
