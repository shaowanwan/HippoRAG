# Experiment Results — MuSiQue, 1000 samples

Full-corpus (11656 passages), no reranker, QA top-5

## Overall

| Method | EM | F1 | R@5 |
|--------|-----|-----|-----|
| GTE baseline (Round 0) | 0.347 | 0.446 | 0.639 |
| **GTE hub** | **0.421** | **0.523** | **0.692** |
| MiniLM baseline (Round 0) | 0.255 | 0.340 | 0.466 |
| **MiniLM hub** | **0.391** | **0.495** | **0.635** |
| Naive reasoning (MiniLM) | 0.360 | 0.447 | 0.516 |

### vs Baseline

| Embedding | Δ EM | Δ F1 | Δ R@5 |
|-----------|------|------|-------|
| GTE | +7.4% | +7.7% | +5.3% |
| MiniLM | +13.6% | +15.5% | +16.9% |

## By Hop

### 2-hop (518 samples)

| Method | EM | F1 | R@5 |
|--------|-----|-----|-----|
| GTE baseline | 0.448 | 0.539 | 0.758 |
| **GTE hub** | **0.529** | **0.629** | **0.844** |
| MiniLM baseline | 0.309 | 0.385 | 0.590 |
| **MiniLM hub** | **0.486** | **0.586** | **0.783** |
| Naive (MiniLM) | 0.465 | 0.548 | 0.682 |

### 3-hop (316 samples)

| Method | EM | F1 | R@5 |
|--------|-----|-----|-----|
| GTE baseline | 0.282 | 0.392 | 0.577 |
| **GTE hub** | **0.367** | **0.479** | **0.600** |
| MiniLM baseline | 0.225 | 0.320 | 0.384 |
| **MiniLM hub** | **0.342** | **0.455** | **0.532** |
| Naive (MiniLM) | 0.304 | 0.393 | 0.391 |

### 4-hop (166 samples)

| Method | EM | F1 | R@5 |
|--------|-----|-----|-----|
| GTE baseline | 0.157 | 0.261 | 0.387 |
| **GTE hub** | **0.187** | **0.276** | **0.396** |
| MiniLM baseline | 0.145 | 0.240 | 0.236 |
| **MiniLM hub** | **0.187** | **0.288** | **0.373** |
| Naive (MiniLM) | 0.139 | 0.234 | 0.236 |

## Reasoning v2 Changes (from v1)
1. Seed weight decay (0.5^age)
2. Same-round inter-discovered edges only
3. Two-way RRF (0.3 × history_norm + 1.0 × current)
4. Re-discovered entities average round
5. No reranker

## Cross-sentence Hub Augmentation
- Coref: entities in 2+ passages, get_eid merge (weight+1)
- Cross-sentence: both entities in 2+ passages, get_eid merge (weight+1)
- No synthetic sentence embeddings
- Only pair embeddings for augmented pairs
- Augmented index cached as pkl

## Git Branches

| Branch | Commit | Description |
|--------|--------|-------------|
| `experiment/v2-cross` | `f315649` | v2+hub — hub filter + get_eid merge + augmented pkl |
| NER v2 code | `c9b7751` | Decay + two-way RRF |

## Shared Resources
- `outputs/musique_ner_pipeline_eval/ner_cache.json` — NER cache (11656 passages)
- `outputs/musique_ner_cross_sentence_eval/cross_sentence_cache.json` — Cross-sentence relations
- `outputs/musique_ner_pipeline_eval/global_ner_index.pkl` — Base NER index (MiniLM)
- `outputs/musique_ner_cross_sentence_eval/cross_sentence_cache_augmented_hub_index.pkl` — Hub augmented index
- Server: same files at `/data/s4303873/HippoRAG/outputs/`
