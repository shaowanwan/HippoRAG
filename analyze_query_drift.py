"""Query drift analysis across three methods on HotpotQA (q3.6):

- v13.1 NER+reasoning (new prompt, bridge re-seeding)
- gap v2 NER+reasoning (PID 67269)
- HippoRAG+IRCoT (thought-concat)

For each method, compute per-round semantic similarity between the
"effective retrieval query" and the original question. Lower similarity =
more drift.

Effective query definitions:
- Rewrite methods (v13.1, gap v2): rewritten_query at round R
- IRCoT (concat): original + " ".join(thoughts[:R+1])
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

V131 = 'outputs/hotpotqa_ner_pipeline_eval/comparison_results_rounds3_20260515_143655_41908.json'
GAPV2 = 'outputs/hotpotqa_ner_pipeline_eval/comparison_results_rounds3_20260506_165457_67269.json'
HIPIR = 'outputs/hotpotqa_ircot_eval/comparison_results.json'

N = 50  # First 50 samples (all methods have ≥ this)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    v131 = json.load(open(V131))['results'][:N]
    gap = json.load(open(GAPV2))['results'][:N]
    hip = json.load(open(HIPIR))['results'][:N]

    # Verify same questions
    same = sum(1 for i in range(N) if v131[i]['question'] == gap[i]['question'] == hip[i]['question'])
    print(f"Same first {N} questions across all 3: {same}/{N}")
    assert same == N, "Questions don't align!"

    # Per-method per-sample per-round trajectories
    # Round 0 = original (sim = 1.0 by definition)
    # Round R (R≥1) = effective query at the start of round R (after R-1 rewrites/thoughts)

    def get_rewrite_trajectory(results, label):
        """For rewrite methods: trajectory[round r] = rewritten_query produced AT round r."""
        trajectories = []  # list of (original, [rewrite0, rewrite1, rewrite2])
        for r in results:
            orig = r['question']
            rounds = r.get('round_diagnostics', [])
            rewrites = [orig]  # round 0 effective = original
            for rd in rounds:
                rq = rd.get('rewritten_query', '').strip()
                if rq:
                    rewrites.append(rq)
                else:
                    rewrites.append(rewrites[-1])  # no rewrite → carry forward
            trajectories.append(rewrites)
        return trajectories

    def get_ircot_trajectory(results, label):
        """For IRCoT: trajectory[round r] = original + ' '.join(thoughts[:r])."""
        trajectories = []
        for r in results:
            orig = r['question']
            thoughts = r.get('ircot_thoughts', [])
            traj = [orig]
            for i in range(len(thoughts)):
                effective = orig + " " + " ".join(thoughts[:i+1])
                traj.append(effective)
            trajectories.append(traj)
        return trajectories

    v131_traj = get_rewrite_trajectory(v131, 'v13.1')
    gap_traj = get_rewrite_trajectory(gap, 'gap v2')
    hip_traj = get_ircot_trajectory(hip, 'IRCoT')

    # Encode all unique texts
    all_texts = set()
    for trajs in [v131_traj, gap_traj, hip_traj]:
        for t in trajs:
            for q in t:
                all_texts.add(q)
    all_texts = list(all_texts)
    print(f"\nEncoding {len(all_texts)} unique queries with MiniLM...")
    embeddings = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
    emb_map = {t: e for t, e in zip(all_texts, embeddings)}

    def aggregate(trajs, max_rounds=4):
        """Return mean sim and mean drift per round."""
        # round 0 = sim 1.0 always; rounds 1..max_rounds-1
        per_round_sims = [[] for _ in range(max_rounds)]
        for traj in trajs:
            orig_emb = emb_map[traj[0]]
            per_round_sims[0].append(1.0)
            for r in range(1, max_rounds):
                if r < len(traj):
                    sim = cosine(orig_emb, emb_map[traj[r]])
                    per_round_sims[r].append(sim)
        return [np.mean(s) if s else None for s in per_round_sims], [len(s) for s in per_round_sims]

    print(f"\n=== Query similarity to original (higher = less drift) ===")
    print(f"{'Method':<20} | round 0 | round 1 | round 2 | round 3")
    print('-' * 75)
    for label, trajs in [('v13.1 (bridge)', v131_traj),
                         ('gap v2', gap_traj),
                         ('IRCoT (thought-concat)', hip_traj)]:
        sims, counts = aggregate(trajs)
        cells = []
        for r in range(4):
            if sims[r] is not None:
                cells.append(f"{sims[r]:.3f} (n={counts[r]})")
            else:
                cells.append("—")
        print(f"{label:<20} | {' | '.join(cells)}")

    # Also report drift per round (1 - sim)
    print(f"\n=== Drift from original (1 - cosine sim, higher = more drift) ===")
    print(f"{'Method':<20} | round 1 drift | round 2 drift | round 3 drift")
    print('-' * 75)
    for label, trajs in [('v13.1 (bridge)', v131_traj),
                         ('gap v2', gap_traj),
                         ('IRCoT (thought-concat)', hip_traj)]:
        sims, counts = aggregate(trajs)
        cells = [f"{1-sims[r]:.3f}" if sims[r] is not None else "—" for r in [1,2,3]]
        print(f"{label:<20} | {' | '.join(cells)}")


if __name__ == "__main__":
    main()
