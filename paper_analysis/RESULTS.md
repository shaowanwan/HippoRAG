# Paper analysis — bridge composition & BRGD-vs-ITER win/loss

All numbers computed from the canonical NER+BRGD run **41732**
(`outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260421_143341_41732.json`,
MuSiQue 1000) and the canonical NER+ITER-RETGEN runs (MuSiQue 15547, 2Wiki 15542).
Reproduce with the scripts in this directory.

## 1. Bridge composition (Fig. `bridge_composition_4way.pdf`)
Each LLM-discovered bridge entity (`round_diagnostics.new_discovered_entities`, all rounds)
is classified, in priority order:
- **Answer**: matches the MuSiQue final answer (substring).
- **Hop-intermediate**: matches an intermediate `question_decomposition` answer.
- **Query**: appears in the original question (substring).
- **Contextual**: everything else — then split via graph connectivity:
  - **evidence**: the resolved entity node links (entity-passage edge) to a gold supporting passage.
  - **distractor**: it does not.
  Resolution uses `_resolve_entities_in_graph` (exact + embedding, threshold 0.55), matching the pipeline.

| Category | Count | % |
|---|---|---|
| Contextual (distractor) | 1732 | 36.0% |
| Hop-intermediate | 1204 | 25.1% |
| Contextual (evidence) | 1085 | 22.6% |
| Query | 396 | 8.2% |
| Answer | 389 | 8.1% |
| **Total** | **4806** | 100% |

- ~64% of bridges are on the answer path; ~36% are topically-related but off-path distractors.
- Distractors are real entities (100% appear in some passage), NOT hallucinations; they diffuse away in PPR.
- Grounding by role: Answer 96%, Hop 91%, Query 72%, Contextual 38.5% grounded in gold.

### Validation (contextual evidence/distractor)
Two independent methods agree:
- **Graph connectivity** (1000 q): 38.5% evidence / 61.5% distractor.
- **LLM judge** (qwen-plus, 162 q sample, 564 bridges): 35.3% relevant / 64.7% unrelated.
→ The ~⅓ relevant / ~⅔ distractor split for contextual is robust.

## 2. BRGD vs ITER-RETGEN win/loss (Fig. `winloss_decomposition.pdf`)
Per-question EM (`ner_em`), one method correct = win. NER backbone.

| | Both correct | BRGD win | ITER win | Both wrong | net |
|---|---|---|---|---|---|
| MuSiQue (41732 vs 15547) | 285 | 113 | 67 | 535 | +46 |
| 2Wiki (28808 vs 15542) | 501 | 148 | 54 | 297 | +94 |

NOTE: supersedes the older memory figure (124/61/+63 was wrong); use these recomputed numbers.

## 3. Win-case sub-classification — why ITER-RETGEN loses (MuSiQue, 113 BRGD wins)
Of the 113 questions where BRGD is correct and ITER-RETGEN is wrong, the reason ITER failed:

| Category | Count | % | Description |
|---|---|---|---|
| A. ITER never produced the bridge entity, refused to answer | 10 | 8.8% | |
| B. ITER mentioned the bridge entity, but retrieval was weak | 30 | 26.5% | dense encoder failed to route bridge entities sharply to PPR seeds; sparse graph weakens diffusion |
| C. ITER had bridge and retrieval; final answer incorrect | 66 | 58.4% | (QA failure, sub-classified below) |
| &nbsp;&nbsp;C1. Hallucination (wrong content) | 22 | 19.5% | |
| &nbsp;&nbsp;C2. Refusal ("not specified", "unknown") | 17 | 15.0% | |
| &nbsp;&nbsp;C3. Partial — under-specified (e.g., year only) | 11 | 9.7% | |
| &nbsp;&nbsp;C4. Near-paraphrase — semantically same, EM=0 | 6 | 5.3% | |
| &nbsp;&nbsp;C5. Paraphrase — alternative phrasing | 4 | 3.5% | |
| &nbsp;&nbsp;C6. Verbose — correct answer wrapped in extra text | 4 | 3.5% | |
| &nbsp;&nbsp;C7. Empty / malformed | 2 | 1.8% | |
| D. ITER drifted in round 1 and did not recover | 6 | 5.3% | |

Key point for the paper: A+B (35.3%) are retrieval/bridge failures — esp. **B (26.5%) is direct
evidence for signal dilution** (ITER had the right entity but its dense query could not route it).
C (58.4%) are QA failures consistent with MuSiQue being answerable without full retrieval.

NOTE: this table was produced by the author's own (LLM-assisted) classification of the 113 win
cases — no reproducible script is checked in. Content preserved here from the thesis slide.

## 4. Component ablations (MuSiQue 1000, NER backbone)
Canonical reference = 41732 (EM .396 / R@5 .623, MMR on, RRF on).

| Ablation | EM | F1 | R@5 | vs reference | run |
|---|---|---|---|---|---|
| **canonical** (MMR on, RRF on) | .396 | — | .623 | — | 41732 |
| **− RRF** (HISTORY_WEIGHT=0) | .370 | — | .582 | −1.4 EM / −4.1 R@5 (vs 25208 .384/.610) | 93436 (2026-06-18) |
| **− MMR** (mmr_lambda=1.0) | .395 | .493 | .618 | −0.1 EM / −0.5 R@5 (within noise) | 96939 (2026-06-18) |

- **RRF helps** (removing it drops EM and R@5) → keep as a component.
- **MMR is neutral at 1000** (.395 vs .396) → NOT a contributing component; downgraded to a
  parenthetical in the method (§3.4) and retained only as a default. (Single-round MMR-off was
  better, .548 vs .506 R@5, but the effect washes out across rounds.)

## Files
- `bridge_composition.py` / `bridge_composition_2wiki.py` — composition (5-category) + figure.
- `winloss_decomposition.py` — win/loss + figure.
- `wincase_atomic.py` / `wincase_atomic_2wiki.py` — why ITER loses, per win case (A/B/C/D).
- `failure_breakdown_2wiki.py` — ITER-fail vs BRGD-fail breakdown (Table in §5.4).
- `contextual_llm_judge.py` — LLM-judge validation of contextual evidence/distractor.

## 5. Echo / cross-round repetition (MuSiQue 1000, BRGD vs NER+ITER)
Cross-round similarity of consecutive `reasoning_traces` (actual logged text, no embedding model,
no reconstruction). ITER-RETGEN repeats near-verbatim; BRGD turns over each round.

| metric (consecutive rounds) | BRGD | NER+ITER |
|---|---|---|
| Jaccard unigram | 0.363 | 0.496 |
| Jaccard bigram | 0.143 | 0.325 |
| ROUGE-L | 0.297 | 0.500 |
| difflib sequence-ratio | 0.110 | 0.385 |
| embedding cosine (MiniLM) | 0.849 | 0.873 |

- Phrase/sequence metrics show the gap (ITER 1.7–3.5× more repetitive); unigram understates it.
- Embedding similarity is comparable (both stay on the question topic) → the echo is *lexical/verbatim*, not semantic.
- NER+ITER per-round seeds were never logged (round_diagnostics empty in all iterretgen runs);
  seed-level numbers (0.72 cross-round overlap) are from a `query_ner` reconstruction (proxy) — use text metrics above instead.

### 2WikiMultiHopQA (same metrics)
| metric (consecutive rounds) | BRGD | NER+ITER |
|---|---|---|
| Jaccard unigram | 0.384 | 0.621 |
| Jaccard bigram | 0.158 | 0.478 |
| ROUGE-L | 0.358 | 0.641 |
| difflib sequence-ratio | 0.205 | 0.582 |

Echo is even stronger on 2Wiki (ITER up to 2.8x more repetitive). Consistent both datasets.
