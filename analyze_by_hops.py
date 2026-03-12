"""Analyze reasoning improvement by number of hops.

Run after evaluate_musique_reasoning.py completes.
Usage: .venv/bin/python analyze_by_hops.py
"""

import json
import numpy as np


def main():
    # Load MuSiQue data for hop info
    data = json.load(open("musique.json"))
    hop_map = {}  # idx -> num_hops
    for idx, sample in enumerate(data):
        n_hops = len(sample.get("question_decomposition", []))
        hop_map[idx] = n_hops

    # Load evaluation results
    results_file = "outputs/musique_reasoning_eval/comparison_results.json"
    with open(results_file) as f:
        all_results = json.load(f)

    results = all_results["results"]
    print(f"Total samples: {len(results)}")

    # Group by hops
    by_hops = {}
    for r in results:
        idx = r["idx"]
        n_hops = hop_map.get(idx, 0)
        if n_hops not in by_hops:
            by_hops[n_hops] = []
        by_hops[n_hops].append(r)

    print(f"\n{'Hops':>4s} {'N':>4s} | {'Base R@5':>8s} {'Reas R@5':>8s} {'Δ R@5':>8s} | {'Base EM':>7s} {'Reas EM':>7s} {'Δ EM':>7s} | {'Avg Rounds':>10s}")
    print("-" * 85)

    for n_hops in sorted(by_hops.keys()):
        group = by_hops[n_hops]
        n = len(group)

        base_r5 = np.mean([r["baseline_retrieval"].get("Recall@5", 0) for r in group])
        reas_r5 = np.mean([r["reasoning_retrieval"].get("final", {}).get("Recall@5", 0) for r in group])

        base_em = np.mean([
            1 if r.get("baseline_answer", "").strip().lower() == r["gold_answer"].strip().lower() else 0
            for r in group
        ])
        reas_em = np.mean([
            1 if r.get("reasoning_answer", "").strip().lower() == r["gold_answer"].strip().lower() else 0
            for r in group
        ])

        avg_rounds = np.mean([
            r["reasoning_retrieval"].get("avg_rounds_used", 1) for r in group
        ])

        print(f"{n_hops:4d} {n:4d} | {base_r5:8.3f} {reas_r5:8.3f} {reas_r5-base_r5:+8.3f} | {base_em:7.3f} {reas_em:7.3f} {reas_em-base_em:+7.3f} | {avg_rounds:10.2f}")

    # Overall
    print("-" * 85)
    n = len(results)
    base_r5 = np.mean([r["baseline_retrieval"].get("Recall@5", 0) for r in results])
    reas_r5 = np.mean([r["reasoning_retrieval"].get("final", {}).get("Recall@5", 0) for r in results])
    base_em = np.mean([
        1 if r.get("baseline_answer", "").strip().lower() == r["gold_answer"].strip().lower() else 0
        for r in results
    ])
    reas_em = np.mean([
        1 if r.get("reasoning_answer", "").strip().lower() == r["gold_answer"].strip().lower() else 0
        for r in results
    ])
    avg_rounds = np.mean([r["reasoning_retrieval"].get("avg_rounds_used", 1) for r in results])
    print(f" All {n:4d} | {base_r5:8.3f} {reas_r5:8.3f} {reas_r5-base_r5:+8.3f} | {base_em:7.3f} {reas_em:7.3f} {reas_em-base_em:+7.3f} | {avg_rounds:10.2f}")


if __name__ == "__main__":
    main()
