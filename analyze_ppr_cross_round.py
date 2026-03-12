"""Analyze PPR diagnostics across rounds.

Key question: does reasoning push PPR diffusion to new territory?
- Does query_sim decrease across rounds (reaching further hops)?
- Does the CHANGE in query_sim predict R@5 improvement?

Run: .venv/bin/python evaluate_musique_reasoning.py --sample_limit 200 --max_rounds 3
Then: .venv/bin/python analyze_ppr_cross_round.py
"""

import json
import numpy as np
from scipy import stats


def main():
    results_file = "outputs/musique_reasoning_eval/comparison_results.json"
    with open(results_file) as f:
        all_results = json.load(f)

    # Collect per-round PPR diagnostics from results
    # These are saved in per_round_qa or we need to extract from round_states
    # For now, parse from the detailed results

    multi_round_data = []

    for item in all_results["results"]:
        reasoning = item.get("reasoning_retrieval", {})
        rounds_used = reasoning.get("avg_rounds_used", 1)

        # Check if PPR diagnostics are saved
        ppr_diags = reasoning.get("ppr_diagnostics", [])
        if not ppr_diags:
            continue

        row = {
            "idx": item["idx"],
            "rounds_used": rounds_used,
            "ppr_diags": ppr_diags,
            "baseline_r5": item["baseline_retrieval"].get("Recall@5", 0),
            "reasoning_r5": reasoning.get("final", {}).get("Recall@5", 0),
        }

        gold = item["gold_answer"].strip().lower()
        row["baseline_em"] = 1 if item.get("baseline_answer", "").strip().lower() == gold else 0
        row["reasoning_em"] = 1 if item.get("reasoning_answer", "").strip().lower() == gold else 0

        multi_round_data.append(row)

    if not multi_round_data:
        print("No PPR diagnostics found in results. Need to re-run evaluation with diagnostics enabled.")
        print("The evaluation will now save PPR diagnostics per round.")
        print("Run: .venv/bin/python evaluate_musique_reasoning.py --sample_limit 200 --max_rounds 3")
        return

    print(f"Samples with PPR diagnostics: {len(multi_round_data)}")

    # Analyze cross-round changes
    sim_deltas = []  # change in query_sim from round 0 to last round
    entropy_deltas = []
    r5_improvements = []
    em_improvements = []

    for row in multi_round_data:
        diags = row["ppr_diags"]
        if len(diags) < 2:
            continue

        round0 = diags[0]
        last_round = diags[-1]

        sim_delta = last_round.get("query_sim", 0) - round0.get("query_sim", 0)
        entropy_delta = last_round.get("entropy", 0) - round0.get("entropy", 0)
        r5_imp = row["reasoning_r5"] - row["baseline_r5"]
        em_imp = row["reasoning_em"] - row["baseline_em"]

        sim_deltas.append(sim_delta)
        entropy_deltas.append(entropy_delta)
        r5_improvements.append(r5_imp)
        em_improvements.append(em_imp)

    if not sim_deltas:
        print("No multi-round samples found.")
        return

    sim_deltas = np.array(sim_deltas)
    entropy_deltas = np.array(entropy_deltas)
    r5_improvements = np.array(r5_improvements)
    em_improvements = np.array(em_improvements)

    print(f"\nMulti-round samples: {len(sim_deltas)}")
    print(f"\n=== CROSS-ROUND CHANGES ===")
    print(f"Δ query_sim:  mean={sim_deltas.mean():+.4f}, std={sim_deltas.std():.4f}")
    print(f"Δ entropy:    mean={entropy_deltas.mean():+.4f}, std={entropy_deltas.std():.4f}")

    print(f"\n=== DOES SIM DECREASE PREDICT R@5 IMPROVEMENT? ===")
    r, p = stats.pearsonr(sim_deltas, r5_improvements)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  Δquery_sim vs R@5_improve:  r={r:+.3f}, p={p:.4f} {sig}")

    r, p = stats.pearsonr(entropy_deltas, r5_improvements)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  Δentropy   vs R@5_improve:  r={r:+.3f}, p={p:.4f} {sig}")

    # Group: did sim decrease or increase?
    sim_decreased = sim_deltas < 0
    sim_increased = sim_deltas >= 0

    print(f"\n=== GROUP: SIM DECREASED vs INCREASED ===")
    if sim_decreased.sum() > 0:
        print(f"Sim decreased ({sim_decreased.sum():3d}): R@5_improve={r5_improvements[sim_decreased].mean():+.3f}, EM_improve={em_improvements[sim_decreased].mean():+.3f}")
    if sim_increased.sum() > 0:
        print(f"Sim increased ({sim_increased.sum():3d}): R@5_improve={r5_improvements[sim_increased].mean():+.3f}, EM_improve={em_improvements[sim_increased].mean():+.3f}")


if __name__ == "__main__":
    main()
