# Case Study — idx=535 (2WikiMultiHopQA, compositional 2-hop)

All numbers verified from the canonical runs (BRGD=28808, NER+ITER-RETGEN=15542) and a
**faithful** seed reconstruction (deterministic `NERIndex.retrieve` on the exact logged queries
= question + previous generation). Reconstruction is exact, not a proxy, because seeding is
deterministic given the query.

## Question
**Q:** Who is the spouse of the director of film *West of Shanghai*?
**Gold chain:** *West of Shanghai* --(directed by)--> **John Farrow** --(spouse)--> **Maureen O'Sullivan**
**Gold answer:** Maureen O'Sullivan

## NER+ITER-RETGEN — FAILS (answer "Not specified", R@5 = 0.0)
Per-round PPR seeds (embedding-matched, top by weight) + the generation that drives the next query.

| Round | Top seeds (weight) | Generation → answer |
|---|---|---|
| R1 | **daughter of shanghai 0.97** · west of shanghai 0.95 · december 6 1911 0.95 · charlie chan in shanghai 0.49 | "director is **John Farrow**... spouse not mentioned" → *Not specified* |
| R2 | april 23 1969 0.98 · daughter of shanghai 0.94 · guo liang 0.94 · **john farrow 0.25** | "none mention *West of Shanghai*" → *Unknown* |
| R3 | **daughter of shanghai 0.97** · december 6 1911 0.94 · charlie chan in shanghai 0.46 | drifts to other "Shanghai" films → *Not specified* |

- Seeds are dominated by **"Shanghai"-titled films**; the WRONG film (*Daughter of Shanghai*) outweighs the right one in every round.
- John Farrow appears only in R2 (because R1's text named him) and only at **weight 0.25** — drowned out; vanishes again in R3.
- Never routes to John Farrow's page → never reaches Maureen O'Sullivan.

## BRGD — SUCCEEDS (answer "Maureen O'Sullivan", R@5: 0.5 → 1.0)
| Round | base R@5 | retrieved (hop) | action |
|---|---|---|---|
| R0 | 0.5 | [West of Shanghai] (hop-1) | reads film page → extracts bridge **John Farrow**; rewrites query to **"Who is the spouse of John Farrow?"** (drops "Shanghai"); injects `john farrow` as a HARD seed |
| R1 | 1.0 | [John Farrow] (hop-2) | discovers **Maureen O'Sullivan** → answer |

## Why ITER's seeds miss (verified embedding analysis)
Raw MiniLM cosine, query vs entity:

| entity | cosine |
|---|---|
| daughter of shanghai (wrong film) | **0.637** |
| west of shanghai (right film) | 0.591 |
| charlie chan in shanghai | 0.560 |
| **john farrow (the bridge)** | **0.240** |

Relation-word effect: `cos(spouse, daughter) = 0.525` vs `cos(spouse, west) = 0.135`.
→ The query asks for a *spouse*, so MiniLM pulls in the family-relation word *Daughter* and ranks the
WRONG film highest; the actual bridge *John Farrow* scores only 0.240, so embedding seeding never
prioritises it. BRGD bypasses this by **naming** the bridge and injecting it as a hard seed.

## Figure layout
```
+----------------------------------------------------------------------+
|  Q: spouse of the director of "West of Shanghai"?   A: Maureen O'Sullivan |
|  chain:  West of Shanghai --dir--> John Farrow --spouse--> M. O'Sullivan  |
+-------------------------------+--------------------------------------+
|  NER+ITER-RETGEN  (FAILS)     |   BRGD  (SUCCEEDS)                    |
|                               |                                      |
|  R1 seeds: [daughter 0.97]    |   R0: read West of Shanghai page      |
|            [west .95][charlie]|       -> bridge: JOHN FARROW          |
|     gen: "John Farrow, but    |       q' = "spouse of John Farrow?"   |
|           spouse not in text" |       inject john farrow (hard seed)  |
|                               |          base R@5 0.5                  |
|  R2 seeds: [daughter .94]     |                                      |
|            [john farrow 0.25] |   R1: seed -> [John Farrow page]      |
|     gen: "none mention film"  |       -> Maureen O'Sullivan           |
|                               |          R@5 1.0  ✓                    |
|  R3 seeds: back to [daughter] |                                      |
|     -> "Not specified"  ✗     |                                      |
|     R@5 = 0.0                 |                                      |
+-------------------------------+--------------------------------------+
|  Embedding trap: cos(q, daughter-of-shanghai)=0.637 > cos(q, west-of- |
|  shanghai)=0.591;  cos(q, john-farrow)=0.240.  Because cos(spouse,    |
|  daughter)=0.525 >> cos(spouse, west)=0.135.                          |
+----------------------------------------------------------------------+
```
Colour key: blue = query entities ("West of Shanghai"); orange = bridge ("John Farrow");
green = answer ("Maureen O'Sullivan"); red = embedding distractors ("Daughter of Shanghai").
