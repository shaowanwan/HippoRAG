# Experiment Results — MuSiQue, 1000 samples

Full-corpus (11656 passages), QA top-5

## ★ Main Experiments

### Overall (1000 samples)

| Method | EM | F1 | R@5 |
|--------|-----|-----|-----|
| NER GTE baseline (Round 0) | 0.347 | 0.446 | 0.639 |
| **NER GTE + reasoning + hub** | **0.421** | **0.523** | **0.692** |
| NER MiniLM baseline (Round 0) | 0.255 | 0.340 | 0.466 |
| **NER MiniLM + reasoning + hub** | **0.391** | **0.495** | **0.635** |
| NER MiniLM + naive reasoning | 0.360 | 0.447 | 0.516 |
| HippoRAG OpenIE baseline | 0.381 | 0.479 | 0.677 |
| **HippoRAG OpenIE + v2 reasoning** | **0.453** | **0.547** | **0.711** |

### vs Baseline

| Method | Δ EM | Δ F1 | Δ R@5 |
|--------|------|------|-------|
| NER GTE | +7.4% | +7.7% | +5.3% |
| NER MiniLM | +13.6% | +15.5% | +16.9% |
| HippoRAG OpenIE | +7.2% | +6.8% | +3.4% |

### 2-hop (518 samples)

| Method | EM | F1 | R@5 |
|--------|-----|-----|-----|
| NER GTE baseline | 0.448 | 0.539 | 0.758 |
| **NER GTE + reasoning + hub** | **0.529** | **0.629** | **0.844** |
| NER MiniLM baseline | 0.309 | 0.385 | 0.590 |
| **NER MiniLM + reasoning + hub** | **0.486** | **0.586** | **0.783** |
| NER MiniLM + naive reasoning | 0.465 | 0.548 | 0.682 |
| HippoRAG baseline | 0.488 | 0.580 | 0.798 |
| **HippoRAG + v2** | **0.542** | **0.643** | **0.868** |

### 3-hop (316 samples)

| Method | EM | F1 | R@5 |
|--------|-----|-----|-----|
| NER GTE baseline | 0.282 | 0.392 | 0.577 |
| **NER GTE + reasoning + hub** | **0.367** | **0.479** | **0.600** |
| NER MiniLM baseline | 0.225 | 0.320 | 0.384 |
| **NER MiniLM + reasoning + hub** | **0.342** | **0.455** | **0.532** |
| NER MiniLM + naive reasoning | 0.304 | 0.393 | 0.391 |
| HippoRAG baseline | 0.320 | 0.436 | 0.619 |
| **HippoRAG + v2** | **0.415** | **0.506** | **0.610** |

### 4-hop (166 samples)

| Method | EM | F1 | R@5 |
|--------|-----|-----|-----|
| NER GTE baseline | 0.157 | 0.261 | 0.387 |
| **NER GTE + reasoning + hub** | **0.187** | **0.276** | **0.396** |
| NER MiniLM baseline | 0.145 | 0.240 | 0.236 |
| **NER MiniLM + reasoning + hub** | **0.187** | **0.288** | **0.373** |
| NER MiniLM + naive reasoning | 0.139 | 0.234 | 0.236 |
| HippoRAG baseline | 0.163 | 0.246 | 0.407 |
| **HippoRAG + v2** | **0.247** | **0.325** | **0.417** |

## Experiment Traces

### NER GTE + reasoning + hub (1000 samples)
- **Result file**: `outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260402_223825_3102030.json`
- **QA replay (no reranker)**: `outputs/musique_ner_pipeline_eval/gte_hub_no_rerank_1000_progress.json`
- **Branch**: `experiment/v2-cross` (commit `f315649`)
- **Code commit**: `5865cd5` (removed reranker)
- **Server**: LIACS calcium, Job 7125 (GTE hub 1000)
- **Embedding**: GTE-Qwen2-7B-instruct, **LLM**: qwen-plus

### NER MiniLM + reasoning + hub (1000 samples)
- **Result file**: `outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260402_235319_34587.json`
- **QA replay (no reranker)**: `/tmp/hub_no_rerank_1000_progress.json`
- **Branch**: `experiment/v2-cross` (commit `f315649`)
- **Code commit**: `5865cd5` (removed reranker)
- **Run**: local macOS
- **Embedding**: all-MiniLM-L6-v2, **LLM**: qwen-plus

### NER MiniLM + naive reasoning (1000 samples)
- **Result file**: `outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260403_010535_35866.json`
- **Branch**: `experiment/naive-reasoning`
- **Run**: local macOS
- **Embedding**: all-MiniLM-L6-v2, **LLM**: qwen-plus
- **Note**: query rewrite only, no bridge entities, no RRF accumulation

### HippoRAG OpenIE + v2 reasoning (1000 samples)
- **Result file**: `outputs/musique_reasoning_eval/qwen-plus__gte-Qwen2-7B-instruct/comparison_results_fullcorpus_v2.json`
- **Branch**: `experiment/hipporag-v2-reasoning` (commit `20deedd`)
- **Server**: LIACS calcium, Job 7216
- **Embedding**: GTE-Qwen2-7B-instruct, **LLM**: qwen-plus
- **OpenIE cache**: `outputs/musique/openie_results_ner_qwen-plus.json`

## Method Components
- **Baseline (Round 0)**: NER co-occurrence graph + sentence/pair matching → PPR
- **Reasoning**: Multi-round LLM reasoning with seed decay, same-round bridge edges, two-way RRF
- **Hub augmentation**: Cross-sentence coreference + causal relation edges (hub-filtered)
- **Naive reasoning**: Query rewrite only, no bridge entities, no RRF accumulation

## Reasoning Strategy
1. Seed weight decay (0.5^age per round)
2. Same-round inter-discovered edges only
3. Two-way RRF (0.3 × history_norm + 1.0 × current)
4. Re-discovered entities average round

## Cross-sentence Hub Augmentation
- Coref: entities in 2+ passages, get_eid merge (weight+1)
- Cross-sentence: both entities in 2+ passages, get_eid merge (weight+1)
- Pair embeddings for augmented pairs only
- Augmented index cached as pkl

## Git Branches

| Branch | Commit | Description |
|--------|--------|-------------|
| `experiment/v2-cross` | `f315649` | NER reasoning + hub (main NER experiments) |
| `feature/full-corpus-reasoning` | `5865cd5` | Removed reranker + results |
| `experiment/hipporag-v2-reasoning` | `20deedd` | HippoRAG OpenIE + v2 reasoning |
| `experiment/naive-reasoning` | — | Naive reasoning ablation |

## Shared Resources
- `outputs/musique_ner_pipeline_eval/ner_cache.json` — NER cache (11656 passages)
- `outputs/musique_ner_cross_sentence_eval/cross_sentence_cache.json` — Cross-sentence relations
- `outputs/musique_ner_pipeline_eval/global_ner_index.pkl` — Base NER index (MiniLM)
- `outputs/musique_ner_cross_sentence_eval/cross_sentence_cache_augmented_hub_index.pkl` — Hub augmented index
- Server: same files at `/data/s4303873/HippoRAG/outputs/`
