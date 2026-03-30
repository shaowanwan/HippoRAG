# Project: Reason-Shaped Memory for HippoRAG 2

## Goal
- Target: EMNLP 2026
- Core contribution: LLM reasoning-guided iterative retrieval over knowledge graph

## Architecture Principles
- **HippoRAG uses OpenIE for entity extraction** — no strict entity schema, entities are free-form phrases from open information extraction. This means entity names are noisy, overlapping, and not canonicalized.
- **PPR scores are probability distributions, NOT comparable across queries** — must use RRF (Reciprocal Rank Fusion) for score fusion, never raw PPR scores.
- **Entity seed expansion is more powerful than query rewrite alone** for multi-hop retrieval — directly modifies PPR random walk starting points on the knowledge graph.
- **Round 0: full HippoRAG pipeline** (entity linking → fact matching → PPR). **Round 1+: expansion PPR only** with LLM-discovered bridge entities as new seeds.
- **Degree-adaptive weights** `base × (1 + log(degree + 1))` compensate for hub node probability dilution in PPR.
- **Mini-PPR on 5-hop subgraph** selects which bridge edges to build (threshold 0.0001).
- **Weighted RRF fusion**: pipeline RRF weight 0.5, expansion RRF weight 1.0 — expansion results are more targeted and should dominate.
- **LLM reasoning reranker** reorders top docs based on reasoning trajectory before QA.

## Current Best Results (MuSiQue, 1000 samples running)
- EM +10~12%, F1 +11~13%, R@5 +20~22% over HippoRAG baseline

# Project Rules

## Experiments
- Every experiment MUST save full reasoning trajectories per query per round: analysis, rewritten_query, discovered_entities, should_stop, retrieved doc IDs
- Save experiment results and setup to `experiment_log.md` after each run
- Output files should be named/organized by parameter config (e.g., `comparison_results_top5_oneshot.json`), so different configs don't overwrite each other. Same-config reruns can overwrite.
- **汇报实验指标时至少包含三个：EM, F1, Recall@5**（不要只报EM）
- **每天实验改的重要地方和实验结果（EM, F1, Recall, efficiency）都要记录到Obsidian笔记里**
- **所有实验必须使用同一个 entity cache**，确保 baseline 一致。有两套 cache，分别对应两套 pipeline：
  - HippoRAG OpenIE pipeline：`--openie_cache outputs/musique/openie_results_ner_qwen-plus.json`
  - NER pipeline：`--ner_cache outputs/musique_ner_pipeline_eval/ner_cache.json`
  - 不同 cache 会产生不同知识图谱，导致 baseline R@5 差异高达 0.03。
- **严禁在实验运行中途修改代码后立刻启动新实验**：必须先确认旧实验已结束或已手动 kill，再改代码，再启动新实验。同时运行多个实验且代码版本不同，会导致结果混乱、无法溯源。

## Server Jobs
- 提交新 job 前，主动删除之前失败 job 产生的无效样本结果文件，避免 resume 时加载坏数据

## Code Changes
- Modifying HippoRAG's original source code (anything outside `src/hipporag/reasoning/`) is allowed, but MUST notify the user before doing so
- Algorithm-level changes (new scoring, new graph operations, new prompt strategies) must be confirmed with the user before implementation

## Environment
- Use `.venv/bin/python` (not system python)
- LLM: qwen-plus via Aliyun DashScope API
- Embedding: all-MiniLM-L6-v2 (local)

## LIACS Server
- **SSH**: `ssh calcium`（通过跳板机 `ssh.liacs.nl` → `calcium.liacs.nl`，配置在 `~/.ssh/config`）
- **不要直接连** `liacs.leidenuniv.nl` 或 `calcium.liacs.nl`，会超时
- 用户: `s4303873`
- 工作目录: `/data/s4303873/HippoRAG`
- Conda 环境: `hipporag`（`source /software/anaconda3/etc/profile.d/conda.sh && conda activate hipporag`）
