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
