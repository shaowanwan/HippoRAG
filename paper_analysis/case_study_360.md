# Case Study — idx=360 (2WikiMultiHopQA, compositional 2-hop) — FIGURE CASE

Chosen for the thesis Case Study figure (supersedes idx=535, which was dropped because ITER's flat
refusals read as "ITER looks dumb" and BRGD's trace leaked the answer via its hypothesised-context
field). idx=360 is fairer: ITER makes a *reasonable but wrong* inference, and BRGD's answer is fully
grounded in a retrieved document.

Runs: BRGD=28808, NER+ITER-RETGEN=15542. All quotes verbatim from `reasoning_traces`.

## Question
**Q:** Which country is the director of film *Get Carter* (2000) from?
**Gold chain:** *Get Carter* (2000) --(director)--> **Stephen Kay** --(born in)--> **New Zealand**
**Gold answer:** New Zealand

## NER+ITER-RETGEN — WRONG ("United States", R@5 = 0.5)
All three rounds return the **identical** output:
> "The director of the 2000 film 'Get Carter' is **Stephen Kay**, and the text states it is an
> '**American** action thriller film', indicating the director is from the **United States**." → *United States*

- Retrieves the film page (hop-1) but never Stephen Kay's biography (hop-2).
- Reader is doc-grounded: with only the film page, it infers the director's nationality from the
  film's "American" label. The inference is reasonable but wrong (Kay is New Zealand-born).

## BRGD — CORRECT ("New Zealand", R@5: 0.5 → 1.0), fully grounded
- **R0** (base R@5 0.5): trace = "The 2000 film 'Get Carter' is identified as an American production
  directed by **Stephen Kay** (from Doc 1)... Stephen Kay is the key bridge entity." →
  bridge **Stephen Kay**; rewritten query "where was Stephen Kay born?"
- **R1** (base R@5 1.0): trace = "Doc 2 states he is '**New Zealand**-born American', directly
  answering the question." → answer **New Zealand**.
- BRGD's R0 trace does NOT pre-state the answer (no hypothesised-context leak, unlike 535); the
  answer is read from the retrieved biography in R1.

## Why ITER does not use world knowledge (both use qwen-plus)
Both methods call the same LLM, so the difference is not what the model knows but how it is used:
- ITER-RETGEN's **reader** is prompted to answer from the retrieved passages (doc-grounded QA), so it
  does not draw on parametric memory to fill the gap; given only the film page it infers from the
  "American" label.
- BRGD's **bridge-prediction** step is allowed to reason/hypothesise, so the model can name a
  retrieval target (Stephen Kay) — but the final answer is still read from the retrieved biography.

This is systematic, not a bad case: on 2WikiMultiHopQA, 86.5% of ITER-RETGEN's head-to-head losses
are "produced the bridge entity but retrieval did not fetch the gold passage" (failure breakdown).
