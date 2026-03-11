# Experiment Log

## Common Setup
- Dataset: MuSiQue (200 samples, seed=42)
- LLM: qwen-plus (Aliyun DashScope)
- Embedding: all-MiniLM-L6-v2 (384-dim, local)
- Reranker: Disabled
- Graph: facts_and_sim_passage_node_unidirectional
- PPR damping: 0.5, linking_top_k: 5, retrieval_top_k: 200, qa_top_k: 5

---

## Exp 1: RRF Query Rewrite (2026-03-07)
- **Branch**: main (commit 1c30424)
- **Method**: Multi-round retrieval with LLM query rewriting + RRF score fusion
- **Max rounds**: 3

### Results
| Metric | Baseline | Reasoning | Delta |
|--------|----------|-----------|-------|
| EM     | 0.4150   | 0.4550    | +4.0% |
| F1     | 0.4941   | 0.5450    | +5.1% |
| R@5    | 0.6617   | 0.7735    | +11.2% |

### Per-Round Breakdown
| Round    | N   | EM     | F1     | R@1    | R@2    | R@5    | R@10   | R@20   |
|----------|-----|--------|--------|--------|--------|--------|--------|--------|
| Baseline | 200 | 0.4150 | 0.4941 | 0.2604 | 0.3917 | 0.6617 | 0.8550 | 1.0000 |
| Round 0  | 199 | 0.4121 | 0.4916 | 0.2617 | 0.3920 | 0.6616 | 0.8543 | 1.0000 |
| Final    | 200 | 0.4550 | 0.5450 | 0.3032 | 0.4908 | 0.7735 | 0.9116 | 1.0000 |

---

## Exp 2: Graph Reshape + Entity Seed Expansion (2026-03-08)
- **Branch**: feature/graph-reshape
- **Method**: Round 0 full retrieval, Round 1+ pure PPR with LLM-discovered bridge entities + GraphOverlay bridge edges + RRF
- **Max rounds**: 3

### Results
| Metric | Baseline | Reasoning | Delta |
|--------|----------|-----------|-------|
| EM     | 0.4150   | 0.4450    | +3.0% |
| F1     | 0.5072   | 0.5366    | +2.9% |
| R@5    | 0.6675   | 0.7092    | +4.2% |
| R@10   | 0.8504   | 0.8996    | +4.9% |

### Per-Round Breakdown
| Round    | N   | EM     | F1     | R@1    | R@2    | R@5    | R@10   | R@20   |
|----------|-----|--------|--------|--------|--------|--------|--------|--------|
| Baseline | 200 | 0.4150 | 0.5072 | 0.2621 | 0.3950 | 0.6675 | 0.8504 | 1.0000 |
| Round 0  | 200 | 0.4150 | 0.5072 | 0.2621 | 0.3950 | 0.6675 | 0.8504 | 1.0000 |
| Round 1  |  78 | 0.2308 | 0.2918 | 0.1902 | 0.2714 | 0.5171 | 0.8066 | 1.0000 |
| Round 2  |  67 | 0.3433 | 0.3868 | 0.1841 | 0.2699 | 0.5336 | 0.8122 | 1.0000 |
| Final    | 200 | 0.4450 | 0.5366 | 0.2654 | 0.4062 | 0.7092 | 0.8996 | 1.0000 |

Avg reasoning rounds: 1.73, Total time: 184.8 min

### Analysis
- **Worse than Exp 1 (RRF query rewrite)** across all metrics
- Round 1+ pure PPR (no fact matching) loses the benefit of rewritten query finding new facts
- Round 1 EM drops to 0.23 — pure graph modification alone is insufficient
- **Conclusion**: Should not skip fact matching. Next step: combine full pipeline rewrite (Exp 1) with entity seed expansion (Exp 2)

---

## Exp 3: Combined Full Pipeline + Entity Seed + Bridge Edges (2026-03-09)
- **Branch**: feature/graph-reshape
- **Method**: Every round runs full pipeline (query rewrite + fact matching + DPR + PPR), with entity seed injection (weight=0.5) + bridge edges (weight=1.0) in Round 1+. Doc-level RRF across rounds.
- **Max rounds**: 3

### Results
| Metric | Baseline | Reasoning | Delta |
|--------|----------|-----------|-------|
| EM     | 0.4200   | 0.4750    | +5.5% |
| F1     | 0.5123   | 0.5653    | +5.3% |
| R@5    | 0.6700   | 0.7760    | +10.6% |
| R@10   | 0.8504   | 0.9108    | +6.0% |

### Per-Round Breakdown
| Round    | N   | EM     | F1     | R@1    | R@2    | R@5    | R@10   | R@20   |
|----------|-----|--------|--------|--------|--------|--------|--------|--------|
| Baseline | 200 | 0.4200 | 0.5123 | 0.2621 | 0.3979 | 0.6700 | 0.8504 | 1.0000 |
| Round 0  | 199 | 0.4171 | 0.5098 | 0.2634 | 0.3982 | 0.6700 | 0.8497 | 1.0000 |
| Round 1  |  89 | 0.3596 | 0.4239 | 0.2125 | 0.3455 | 0.6264 | 0.8446 | 1.0000 |
| Round 2  |  61 | 0.3607 | 0.4008 | 0.2117 | 0.3374 | 0.6107 | 0.8128 | 1.0000 |
| Final    | 200 | 0.4750 | 0.5653 | 0.2831 | 0.4451 | 0.7760 | 0.9108 | 1.0000 |

Avg reasoning rounds: 1.75

### Analysis
- **Best EM/F1 so far**: EM +5.5% (vs Exp 1 +4.0%), F1 +5.3% (vs Exp 1 +5.1%)
- R@5 comparable to Exp 1 (+10.6% vs +11.2%), R@10 better (+6.0% vs Exp 1 not recorded)
- Entity seed + bridge edges provide marginal improvement on top of query rewrite
- Entity seed weight (fixed 0.5) and bridge edge weight (fixed 1.0) are not calibrated — may introduce noise
- **Note**: This experiment's code was never committed. The results came from uncommitted code combining Exp 1 + Exp 2.
- **Next step**: Properly implement and commit the combined approach → Exp 4.

---

## Exp 4: Combined Query Rewrite + Entity Seed Expansion (2026-03-10)
- **Branch**: feature/graph-reshape
- **Method**: Exp 1 (query rewrite + full pipeline + RRF) + Exp 2 (entity seed expansion + bridge edges + extra PPR)
- **Max rounds**: 3

### Architecture

Each round executes the following steps:

1. **Full pipeline retrieval** with the current (possibly rewritten) query:
   - Fact matching → entity linking → DPR → PPR → ranked docs
   - Results accumulated into doc-level RRF: `score[doc] += 1/(60 + rank + 1)`

2. **Entity seed expansion** (Round 1+ only, if bridge entities discovered in previous rounds):
   - Copy current round's `node_weights` (from fact matching + DPR)
   - Add discovered entity seeds: `weight[vid] += 0.5` for each resolved entity
   - Build bridge edges (weight=1.0) via GraphOverlay:
     - discovered entities ↔ existing seed entities (full bipartite connection)
     - discovered entities ↔ discovered entities (full clique)
   - Run extra PPR on modified graph → results also accumulated into RRF

3. **LLM reasoning** (not on last round):
   - Input: original query, current query, top-10 retrieved docs, previous traces
   - Output: analysis, rewritten_query, discovered_entities (1-5), should_stop
   - Rewritten query used for next round's full pipeline
   - Discovered entities resolved to graph vertices (exact hash → embedding fallback, threshold=0.6)

4. **Early stop**: if Recall@5 ≥ 1.0 or LLM says should_stop=true

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| max_rounds | 3 | Maximum reasoning rounds |
| rrf_k | 60 | RRF constant: `1/(k + rank + 1)` |
| entity_seed_weight | 0.5 | Weight added for each discovered entity seed |
| bridge_edge_weight | 1.0 | Weight of temporary bridge edges |
| embedding_threshold | 0.6 | Cosine similarity threshold for entity resolution fallback |
| linking_top_k | 5 | Max entity seeds from fact matching |
| damping | 0.5 | PPR damping factor |

### LLM Prompt Design

**System prompt** asks LLM to perform 4 tasks simultaneously:
1. Analyze what info is found vs missing
2. Identify bridge entities (in docs but NOT in original query, 1-5 max, lowercase)
3. Rewrite query to target missing info (incorporate bridge entities)
4. Decide whether to stop

**User prompt** includes:
- Original query + current query (round N)
- Top-10 retrieved docs (truncated to 500 chars each)
- Previous reasoning traces (up to 3)

**Key difference from Exp 1/2**: Prompt requests BOTH `rewritten_query` and `discovered_entities` in a single LLM call. Exp 1 only had `rewritten_query`, Exp 2 only had `discovered_entities`.

### Results
| Metric | Baseline | Reasoning | Delta |
|--------|----------|-----------|-------|
| EM     | 0.4200   | 0.5000    | **+8.0%** |
| F1     | 0.5123   | 0.5792    | **+6.69%** |
| R@5    | 0.6717   | 0.8054    | **+13.37%** |

### Statistics
- Avg reasoning rounds: 1.83
- Entity discovery: 52.0% of queries found bridge entities in graph
- Avg entities discovered per query: 3.04
- Total time: 63.7 min

### Comparison with Previous Experiments
| Exp | Method | EM Δ | F1 Δ | R@5 Δ |
|-----|--------|------|------|-------|
| 1 | Query rewrite only | +4.0% | +5.1% | +11.2% |
| 2 | Seed expansion only (pure PPR) | +3.0% | +2.9% | +4.2% |
| 3 | Combined (uncommitted) | +5.5% | +5.3% | +10.6% |
| **4** | **Combined (committed)** | **+8.0%** | **+6.69%** | **+13.37%** |

### Analysis
- **Best results across all experiments** — EM +8.0%, R@5 +13.37%
- Significantly outperforms Exp 3 backup (+2.5% EM, +2.8% R@5), likely due to cleaner code and proper prompt design
- Query rewrite provides retrieval diversity (new facts each round), entity seed expansion reinforces graph connectivity — both are essential
- 52% entity resolve rate suggests room for improvement (better entity matching or LLM entity extraction)
- Current limitations:
  - Discovered entities accumulate across rounds without pruning (wrong entities persist)
  - Fixed seed weight (0.5) regardless of LLM confidence
  - No mechanism to detect and recover from bad reasoning rounds

---

## Exp 5: Virtual Entity Nodes (2026-03-10)
- **Branch**: feature/graph-reshape
- **Method**: Exp 4 + add unresolved entities as virtual graph nodes (connected to passages by text match)
- **Result**: No improvement over Exp 4 — only 15% of queries triggered virtual nodes (avg 0.23/query), negligible impact
- **Conclusion**: Abandoned. Most LLM-discovered entities either exist in graph or don't appear in passage text.

---

## Exp 6: Weighted RRF (2026-03-11)
- **Branch**: feature/graph-reshape
- **Method**: Exp 4 + round-weighted RRF. Later rounds contribute more: `round_weight = 1.0 + round_i * 0.5`
  - Round 0: weight 1.0, Round 1: weight 1.5, Round 2: weight 2.0
  - Seed expansion PPR uses same round weight
  - Entity seed weight remains fixed at 0.5 (round-based entity weight tested but no improvement)

### Results
| Metric | Baseline | Reasoning | Delta |
|--------|----------|-----------|-------|
| EM     | 0.4150   | 0.5100    | **+9.5%** |
| F1     | 0.5073   | 0.5996    | **+9.23%** |
| R@5    | 0.6717   | 0.8121    | **+14.04%** |

### Statistics
- Avg reasoning rounds: 1.80
- R@5=1.0 queries: 127/200 (Exp 4: 122/200)

### Comparison with Previous Experiments
| Exp | Method | EM Δ | F1 Δ | R@5 Δ |
|-----|--------|------|------|-------|
| 1 | Query rewrite only | +4.0% | +5.1% | +11.2% |
| 2 | Seed expansion only | +3.0% | +2.9% | +4.2% |
| 3 | Combined (uncommitted) | +5.5% | +5.3% | +10.6% |
| 4 | Combined (committed) | +8.0% | +6.69% | +13.37% |
| 5 | + Virtual entity nodes | +8.0% | +6.69% | +13.54% |
| **6** | **+ Weighted RRF** | **+9.5%** | **+9.23%** | **+14.04%** |

### Bad Case Analysis (Exp 4, 200 samples)

| Category | Count | % | Description |
|----------|-------|---|-------------|
| R@5=1 & EM=1 | 78 | 39% | Fully correct |
| R@5=1 & EM=0 | 44 | 22% | QA bottleneck — docs found but answer wrong |
| R@5<1 & EM=0 | 56 | 28% | Retrieval bottleneck — missing docs |
| R@5<1 & EM=1 | 22 | 11% | Partial retrieval but answer correct |
| Regression (B→R) | 6 | 3% | Baseline correct, reasoning wrong |

**QA bottleneck (44 cases)**: 13 format mismatch (F1≥0.5), 8 verbose, 23 wrong reasoning.
**Retrieval bottleneck (56 cases)**: All missing docs found in top-20, just ranked too low. 23 in top-10, 33 in top-20.

### Analysis
- Weighted RRF gives +1.5% EM over equal-weight, by promoting later rounds' more targeted results
- Round-based entity seed weight tested (weight = 0.5 × round_weight) but no improvement — PPR normalizes reset_prob, diluting the difference
- Main remaining bottlenecks: QA model quality (22%) and retrieval ranking (28%)

---

## Exp 7: Higher Expansion Damping (2026-03-11)
- **Branch**: feature/graph-reshape
- **Method**: Exp 6 + higher damping (0.7 vs 0.5) for seed expansion PPR
  - Base PPR (full pipeline): damping=0.5 (unchanged)
  - Seed expansion PPR: damping=0.7 (bridge entities spread further along graph)
  - Rationale: higher damping = more walking along edges = bridge entity influence propagates further, less reset to start

### Key Parameters
| Parameter | Base PPR | Expansion PPR |
|-----------|----------|---------------|
| damping | 0.5 | **0.7** |
| RRF round weight | 1.0 + i×0.5 | 1.0 + i×0.5 |
| entity seed weight | — | 0.5 (fixed) |

### Results
| Metric | Baseline | Reasoning | Delta |
|--------|----------|-----------|-------|
| EM     | 0.4200   | 0.5150    | **+9.5%** |
| F1     | 0.5089   | 0.6025    | **+9.37%** |
| R@5    | 0.6692   | 0.8208    | **+15.17%** |

### Statistics
- Avg reasoning rounds: 1.80
- R@5=1.0 queries: 128/200

### Comparison with All Experiments
| Exp | Method | EM Δ | F1 Δ | R@5 Δ |
|-----|--------|------|------|-------|
| 1 | Query rewrite only | +4.0% | +5.1% | +11.2% |
| 2 | Seed expansion only | +3.0% | +2.9% | +4.2% |
| 3 | Combined (uncommitted) | +5.5% | +5.3% | +10.6% |
| 4 | Combined (committed) | +8.0% | +6.69% | +13.37% |
| 5 | + Virtual entity nodes | +8.0% | +6.69% | +13.54% |
| 6 | + Weighted RRF | +9.5% | +9.23% | +14.04% |
| **7** | **+ Expansion damping 0.7** | **+9.5%** | **+9.37%** | **+15.17%** |

### Analysis
- **Best R@5 so far** — +15.17%, up from +14.04% in Exp 6
- Higher damping lets bridge entities propagate further through graph, finding more relevant passages
- EM unchanged (+9.5%) because the newly found passages hit QA bottleneck cases (docs found but QA model fails)
- Potential next steps: increase qa_top_k, optimize QA prompt, or use stronger LLM

---

## Exp 8: Degree-Adaptive Weights (2026-03-11)
- **Branch**: feature/graph-reshape
- **Method**: Exp 7 + degree-adaptive weights for both bridge edges and entity seeds
  - Weight formula: `weight = base × (1 + log(degree + 1))`
  - High-degree hub nodes get proportionally heavier weights to resist PPR probability dilution
  - Applied to both bridge edge weights and entity seed weights

### Code Changes vs Exp 7
The only change is in `controller.py`:

1. **New method `_degree_adaptive_weight(vid, base_weight)`**:
   - Returns `base_weight × (1 + log(degree + 1))`
   - Degree=3 (median entity) → multiplier ≈ 2.4×
   - Degree=50 → multiplier ≈ 4.9×
   - Degree=145 (max observed) → multiplier ≈ 6.0×

2. **Bridge edge weights** (previously fixed 1.0):
   - Now: `bridge_weight = degree_adaptive_weight(vid, 1.0)` per discovered entity
   - Inter-discovered edges: `max(weight_d1, weight_d2)`

3. **Entity seed weights** (previously fixed 0.5):
   - Now: `seed_weight = degree_adaptive_weight(vid, 0.5)`
   - Note: only relative ratios matter (PPR normalizes reset_prob)

### Key Parameters (cumulative from Exp 6+7+8)
| Parameter | Value | Description |
|-----------|-------|-------------|
| RRF_ROUND_BOOST | 0.5 | Round i weight = 1.0 + i × 0.5 (Exp 6) |
| EXPANSION_DAMPING | 0.7 | Seed expansion PPR damping (Exp 7) |
| DEFAULT_ENTITY_SEED_WEIGHT | 0.5 | Base weight before degree scaling (Exp 8) |
| bridge_edge_base | 1.0 | Base weight before degree scaling (Exp 8) |
| degree_adaptive | log(deg+1) | Scaling factor for weights (Exp 8) |

### Results (2 runs for robustness)
| Metric | Baseline | Run 1 | Run 2 | Avg Delta |
|--------|----------|-------|-------|-----------|
| EM     | 0.420    | 0.525 | 0.520 | **+10.3%** |
| F1     | 0.512    | 0.614 | 0.609 | **+9.9%** |
| R@5    | 0.672    | 0.818 | 0.817 | **+14.8%** |

### Statistics
- Avg reasoning rounds: 1.80
- Total time: ~6-9 min

### Comparison with All Experiments
| Exp | Method | EM Δ | F1 Δ | R@5 Δ |
|-----|--------|------|------|-------|
| 1 | Query rewrite only | +4.0% | +5.1% | +11.2% |
| 2 | Seed expansion only | +3.0% | +2.9% | +4.2% |
| 3 | Combined (uncommitted) | +5.5% | +5.3% | +10.6% |
| 4 | Combined (committed) | +8.0% | +6.69% | +13.37% |
| 5 | + Virtual entity nodes | +8.0% | +6.69% | +13.54% |
| 6 | + Weighted RRF | +9.5% | +9.23% | +14.04% |
| 7 | + Expansion damping 0.7 | +9.5% | +9.37% | +15.17% |
| **8** | **+ Degree-adaptive (log)** | **+10.3%** | **+9.9%** | **+14.8%** |

### Failed Experiments (after Exp 8)
| Exp | Method | EM Δ | Result |
|-----|--------|------|--------|
| 9 | Degree×rarity (TF-IDF style) | +9.0% | Worse — rarity penalized good bridge entities |
| 10 | sqrt + per-round seed scaling | +9.5% | Worse — sqrt over-compensated, seed scaling diluted by normalization |
| 11 | log + per-round bridge scaling | +9.5% | Worse — stronger bridges over-concentrated PPR probability |

### Analysis
- **Best EM/F1 across all experiments** — degree-adaptive log weights provide the best balance
- Log compression (vs sqrt or linear) avoids high-degree hubs dominating PPR diffusion while still compensating for dilution
- Per-round weight scaling (seed or bridge) does not help:
  - Seed weights: PPR normalizes reset_prob → absolute values don't matter, only ratios
  - Bridge weights: stronger edges over-concentrate probability at bridge-connected nodes
- Results are robust across 2 independent runs (±0.5% variance)
