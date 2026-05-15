"""Query drift: naive-reasoning vs full method (qwen-plus, MuSiQue, 1000 samples).

Single-variable comparison: only the bridge re-seeding mechanism differs.
Both use:
- qwen-plus + MiniLM
- Same REWRITE_SYSTEM_PROMPT (default, no HOTPOTQA_MODE)
- Same NER cache
- Same data (MuSiQue 1000)
- max_rounds 3

Difference:
- PID 34587 (full): bridge re-seeding + temp edges + RRF + degree-adaptive
- PID 35866 (naive): query rewrite only, NO bridge re-seeding, NO RRF, NO temp edges

Hypothesis: bridge re-seeding ANCHORS rewrites via observed entities,
so PID 34587's rewrites should drift LESS than PID 35866's.
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

FULL = 'outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260402_235319_34587.json'
NAIVE = 'outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260403_010535_35866.json'


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def trajectory(results, n):
    """For each sample, return [original_q, rewrite1, rewrite2, rewrite3, ...]."""
    out = []
    for r in results[:n]:
        orig = r['question']
        rounds = r.get('round_diagnostics', [])
        traj = [orig]
        for rd in rounds:
            rq = rd.get('rewritten_query', '').strip()
            if rq:
                traj.append(rq)
            else:
                traj.append(traj[-1])  # no rewrite this round → carry forward
        out.append(traj)
    return out


def main():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    full = json.load(open(FULL))
    naive = json.load(open(NAIVE))
    n_full = full['summary']['n_completed']
    n_naive = naive['summary']['n_completed']
    n = min(n_full, n_naive)
    print(f"Full PID 34587 n={n_full}, Naive PID 35866 n={n_naive}, comparing first {n}")

    full_traj = trajectory(full['results'], n)
    naive_traj = trajectory(naive['results'], n)

    # Sanity: same first n questions?
    same = sum(1 for i in range(n) if full_traj[i][0] == naive_traj[i][0])
    print(f"Same questions: {same}/{n}")
    if same < n:
        print(f"  ⚠️ Mismatch — first differing idx: {next(i for i in range(n) if full_traj[i][0] != naive_traj[i][0])}")

    # Encode all unique texts
    all_texts = set()
    for trajs in [full_traj, naive_traj]:
        for t in trajs:
            for q in t:
                all_texts.add(q)
    all_texts = sorted(all_texts)
    print(f"\nEncoding {len(all_texts)} unique queries with MiniLM...")
    embs = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
    emb_map = {t: e for t, e in zip(all_texts, embs)}

    def aggregate(trajs, max_rounds=4):
        per_round = [[] for _ in range(max_rounds)]
        for traj in trajs:
            orig_emb = emb_map[traj[0]]
            per_round[0].append(1.0)
            for r in range(1, max_rounds):
                if r < len(traj):
                    per_round[r].append(cosine(orig_emb, emb_map[traj[r]]))
        return [(np.mean(s), len(s)) if s else (None, 0) for s in per_round]

    print(f"\n=== Drift (1 - cosine sim from original) ===\n")
    print(f"{'Method':<22} | round 1 drift  | round 2 drift  | round 3 drift")
    print('-' * 80)
    for label, trajs in [('Full (PID 34587)', full_traj),
                         ('Naive (PID 35866)', naive_traj)]:
        sims = aggregate(trajs)
        cells = []
        for r in [1, 2, 3]:
            s, c = sims[r]
            if s is not None:
                cells.append(f"{1-s:.3f} (n={c})")
            else:
                cells.append("—")
        print(f"{label:<22} | {' | '.join(cells)}")

    # Δ
    full_sims = aggregate(full_traj)
    naive_sims = aggregate(naive_traj)
    print(f"\n=== Δ drift (Naive - Full); positive = naive drifts MORE ===")
    for r in [1, 2, 3]:
        fs, fc = full_sims[r]
        ns, nc = naive_sims[r]
        if fs is not None and ns is not None:
            d = (1-ns) - (1-fs)
            print(f"  round {r}: ΔDrift = {d:+.3f}  (full n={fc}, naive n={nc})")

    # Also: continuation rate (how many samples used round 2/3 vs stopped at round 1)
    print(f"\n=== Round usage (% samples reaching round r) ===")
    for label, trajs in [('Full', full_traj), ('Naive', naive_traj)]:
        usage = []
        for r in [1, 2, 3]:
            reached = sum(1 for t in trajs if len(t) > r) / len(trajs) * 100
            usage.append(f"r{r}: {reached:.1f}%")
        print(f"  {label}: {' '.join(usage)}")


if __name__ == "__main__":
    main()
