"""
Evaluate HippoRAG OpenIE graph + canonical bridge reasoning on MuSiQue.

Combines:
  - HippoRAG OpenIE graph + retrieval pipeline (scaffolding from evaluate_musique_ircot.py)
  - Canonical bridge reasoning algorithm (verbatim from evaluate_musique_ner_pipeline.py
    @ experiment/only-right-baseline HEAD 8e98b23, matches PID 41732 EM=0.396)

Adaptations:
  - HippoRAG has no `pair_embeddings`. NO_SIM_FACTOR is forced to 1 so
    `_get_bridge_query_sims` returns {vid: 1.0} via its existing fallback path.
  - HippoRAG retrieve interface is wrapped via `HippoRAGIndexAdapter` so that
    canonical functions can call `index.retrieve(...)` unchanged.

Usage:
    .venv/bin/python evaluate_musique_hipporag_bridge.py \
        --data_path musique.json --sample_limit 5 --max_rounds 3
"""
import json
import os
import sys
import argparse
import logging
import signal
import time
import traceback
import gc
import math
import re as _re
from hashlib import md5
from typing import Dict, List, Optional, Tuple

import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Force NO_SIM_FACTOR=1: HippoRAG has no pair_embeddings, so query-pair sim is unavailable.
# The canonical _degree_adaptive_weight respects this env to bypass the sim factor.
os.environ.setdefault("NO_SIM_FACTOR", "1")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Section 1: Scaffolding utilities (from evaluate_musique_ircot.py)
# ============================================================================

def load_musique_data(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer(s):
    s = s.lower().strip()
    s = _re.sub(r'[^\w\s]', '', s)
    s = _re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def em_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)


def check_em(pred, gold_answer, gold_aliases):
    all_golds = [gold_answer] + (gold_aliases if gold_aliases else [])
    return any(em_match(pred, g) for g in all_golds)


def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _lazy_imports():
    """Import HippoRAG modules inside function to avoid multiprocessing spawn issue on macOS."""
    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.misc_utils import QuerySolution, compute_mdhash_id
    from src.hipporag.embedding_model import _get_embedding_model_class
    from src.hipporag.prompts.prompt_template_manager import PromptTemplateManager
    from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
    return (
        HippoRAG, QuerySolution, compute_mdhash_id, _get_embedding_model_class,
        PromptTemplateManager, RetrievalRecall,
    )


# ============================================================================
# Section 2: Canonical NER reasoning — VERBATIM copy from
#   evaluate_musique_ner_pipeline.py @ experiment/only-right-baseline (HEAD 8e98b23)
# Do not modify function bodies. Line numbers in comments refer to the donor file.
# ============================================================================

# ── Utility (donor line 38) ─────────────────────────────────────────────────
def compute_hash(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


# ── Query NER prompt + helper (donor lines 147-180) ─────────────────────────
QUERY_NER_SYSTEM = """Your task is to extract the 2 most important named entities from the given question.
These should be the key entities needed to answer the question.
Respond with a JSON object like {"named_entities": ["entity1", "entity2"]}."""

QUERY_NER_ONE_SHOT_INPUT = """When was Neville A. Stanton's employer founded?"""
QUERY_NER_ONE_SHOT_OUTPUT = """{"named_entities": ["Neville A. Stanton", "employer"]}"""


def query_ner(question: str, llm_client) -> List[str]:
    """Extract top-2 key entities from a question using LLM."""
    messages = [
        {"role": "system", "content": QUERY_NER_SYSTEM},
        {"role": "user", "content": QUERY_NER_ONE_SHOT_INPUT},
        {"role": "assistant", "content": QUERY_NER_ONE_SHOT_OUTPUT},
        {"role": "user", "content": question},
    ]
    try:
        result = llm_client.infer(messages=messages)
        response = result[0] if isinstance(result, tuple) else result
        if not isinstance(response, str):
            response = response[0]["content"]
        match = _re.search(r'\{.*\}', response, _re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("named_entities", [])[:2]
        return []
    except Exception as e:
        logger.warning(f"Query NER failed: {e}")
        return []


# ── Recall@k (donor line 791) ───────────────────────────────────────────────
def recall_at_k(retrieved_docs: List[str], gold_docs: List[str], k: int) -> float:
    """Compute Recall@k."""
    if not gold_docs:
        return 0.0
    retrieved_set = set(retrieved_docs[:k])
    gold_set = set(gold_docs)
    return len(retrieved_set & gold_set) / len(gold_set)


# ── Constants (donor lines 805-810) ─────────────────────────────────────────
DEFAULT_ENTITY_SEED_WEIGHT = 0.5
RRF_ROUND_BOOST = 0.5
EXPANSION_DAMPING = 0.7
MINI_PPR_THRESHOLD = 0.0001
PIPELINE_RRF_WEIGHT = 1.0
EXPANSION_RRF_WEIGHT = 1.0


# ── Reasoning prompts (donor lines 813-878) ─────────────────────────────────
REWRITE_SYSTEM_PROMPT = """You are a retrieval reasoning assistant inspired by how human memory uses both observed cues and hypothetical context. Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

1. Analyze what information has been found and what is still missing.
2. Identify CONCRETE bridge entities you SEE in retrieved documents (these will guide structured retrieval).
3. Hypothesize a CONTEXT SCENARIO — natural language describing the likely time period, location, domain, or related concepts (this is your reasoning context, NOT the retrieval query).
4. Rewrite the query using ONLY bridge entities and the original question — keep it focused and short. Do NOT include the context_scenario in the rewritten_query.
5. Decide whether to continue retrieval or stop.

Respond in JSON format:
{
    "analysis": "Brief analysis of what's found vs missing",
    "discovered_entities": ["entity1", "entity2"],
    "context_scenario": "Hypothesized context: time period, location, domain, related concepts",
    "rewritten_query": "Focused natural language query with bridge entities only",
    "should_stop": false
}

Rules:
- "discovered_entities": ONLY entities you can directly SEE in retrieved documents. 1-5 entities max. Use lowercase. DO NOT guess.
- "context_scenario": Short natural language description of the hypothesized context. Stored as your reasoning trace and seen by future rounds, but NOT used directly for retrieval. Examples:
  * "Likely 16th century European religious reform, possibly Lutheran/Calvinist figures"
  * "North Carolina state administration history, 18-19th century capital designation"
  * "Modern NBA basketball event, post-2000 era"
- "rewritten_query": Focused query using ONLY bridge entities + reformulation of the original question. DO NOT add context_scenario, time periods, or hypothesized terms here. Keep it short and concrete.
- "should_stop": true ONLY if retrieved documents already contain a complete answer.

Example:
Q: "When did the city where Curry's college is located become NC's capital?"
After round 0 finding Stephen Curry and Davidson College:
{
    "analysis": "Found Curry attended Davidson College in NC. Need the city of Davidson and its history as state capital.",
    "discovered_entities": ["davidson college", "north carolina"],
    "context_scenario": "Likely 18-19th century North Carolina state administration history; possibly involving Raleigh or another NC city becoming the state capital.",
    "rewritten_query": "When did Davidson, North Carolina become the state capital?",
    "should_stop": false
}

Note: rewritten_query is short, only uses bridge entities. The "18-19th century" hint stays in context_scenario, NOT in rewritten_query.
"""


REWRITE_SYSTEM_PROMPT_REWRITE_FIRST = """You are a retrieval reasoning assistant inspired by how human memory uses both observed cues and hypothetical context. Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

1. Analyze what information has been found and what is still missing.
2. Rewrite the query to better target the missing information — keep it focused and short.
3. Hypothesize a CONTEXT SCENARIO — natural language describing the likely time period, location, domain, or related concepts (this is your reasoning context, NOT the retrieval query).
4. Identify CONCRETE bridge entities you SEE in retrieved documents that help answer the rewritten query (these will guide structured retrieval).
5. Decide whether to continue retrieval or stop.

Respond in JSON format:
{
    "analysis": "Brief analysis of what's found vs missing",
    "rewritten_query": "Focused natural language query targeting missing info",
    "context_scenario": "Hypothesized context: time period, location, domain, related concepts",
    "discovered_entities": ["entity1", "entity2"],
    "should_stop": false
}

Rules:
- "rewritten_query": Focused query targeting the missing piece. Short and concrete.
- "context_scenario": Short natural language description of the hypothesized context. Stored as your reasoning trace and seen by future rounds, but NOT used directly for retrieval.
- "discovered_entities": ONLY entities you can directly SEE in retrieved documents that help answer the rewritten query. 1-5 entities max. Use lowercase. DO NOT guess.
- "should_stop": true ONLY if retrieved documents already contain a complete answer.
"""


# ── reason_and_rewrite (donor lines 881-941) ────────────────────────────────
def reason_and_rewrite(original_query: str, current_query: str, retrieved_docs: List[str],
                       round_idx: int, previous_traces: List[str], llm_client) -> dict:
    """LLM reasoning: analyze retrieved docs, rewrite query, discover bridge entities."""
    docs_text = ""
    for i, doc in enumerate(retrieved_docs[:5]):
        docs_text += f"[Doc {i+1}] {doc}\n\n"

    user_content = f"""Original query: {original_query}
Current query (round {round_idx}): {current_query}

Retrieved documents so far:
{docs_text}"""

    if previous_traces:
        user_content += "\nPrevious reasoning traces:\n"
        for t in previous_traces[-3:]:
            user_content += f"- {t}\n"

    user_content += "\nAnalyze and provide your reasoning output in JSON."

    _rewrite_first = os.getenv("REWRITE_FIRST", "0") == "1"
    _sys_prompt = REWRITE_SYSTEM_PROMPT_REWRITE_FIRST if _rewrite_first else REWRITE_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": _sys_prompt},
        {"role": "user", "content": user_content},
    ]

    defaults = {"analysis": "", "discovered_entities": [], "context_scenario": "", "rewritten_query": "", "should_stop": False}
    try:
        response = llm_client.infer(messages=messages)
        if isinstance(response, tuple):
            response = response[0]
        if not isinstance(response, str):
            response = response[0]["content"]
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        parsed = json.loads(text)
        for key in defaults:
            if key not in parsed:
                parsed[key] = defaults[key]
        if not isinstance(parsed.get("discovered_entities", []), list):
            parsed["discovered_entities"] = []
        parsed["discovered_entities"] = [
            str(e).lower().strip() for e in parsed["discovered_entities"] if e
        ][:5]
        ctx = parsed.get("context_scenario", "")
        if not isinstance(ctx, str):
            ctx = ""
        parsed["context_scenario"] = ctx.strip()
        return parsed
    except Exception as e:
        logger.warning(f"Reasoning failed: {e}")
        defaults["analysis"] = str(e)[:200]
        defaults["should_stop"] = True
        return defaults


# ── Entity resolution (donor lines 946-989) ─────────────────────────────────
def _resolve_entities_in_graph(index, entity_names: List[str], threshold: float = 0.55) -> Dict[str, Tuple[int, float]]:
    """Resolve entity names to graph vertex IDs with similarity scores.
    Returns {name: (vid, sim_score)}. Exact match gets sim=1.0."""
    resolved = {}
    unresolved = []

    for name in entity_names:
        ent_key = compute_hash(name.lower(), prefix="entity-")
        vid = index.node_name_to_idx.get(ent_key)
        if vid is not None:
            resolved[name] = (vid, 1.0)
            logger.info(f"  Entity '{name}' found (exact) -> vertex {vid}")
        else:
            unresolved.append(name)

    # Embedding fallback for unresolved entities
    if unresolved and index.entity_embeddings is not None and len(index.entity_embeddings) > 0:
        query_embs = index.embedding_model.batch_encode(unresolved)
        if query_embs.ndim == 1:
            query_embs = query_embs.reshape(1, -1)
        # L2 normalize for cosine similarity
        q_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1, q_norms)
        query_embs = query_embs / q_norms

        e_norms = np.linalg.norm(index.entity_embeddings, axis=1, keepdims=True)
        e_norms = np.where(e_norms == 0, 1, e_norms)
        normed_ents = index.entity_embeddings / e_norms

        similarities = query_embs @ normed_ents.T

        for i, name in enumerate(unresolved):
            best_idx = np.argmax(similarities[i])
            sim = float(similarities[i][best_idx])
            if sim >= threshold:
                ent_key = index.entity_keys[best_idx]
                vid = index.node_name_to_idx.get(ent_key)
                if vid is not None:
                    resolved[name] = (vid, sim)
                    logger.info(f"  Entity '{name}' matched by embedding (sim={sim:.3f}) -> vertex {vid}")
            else:
                logger.info(f"  Entity '{name}' NOT found in graph (best sim={sim:.3f})")

    return resolved


# ── _degree_adaptive_weight (donor line 992) ────────────────────────────────
def _degree_adaptive_weight(index, vid: int, base_weight: float, sim: float = 1.0) -> float:
    """Scale weight by semantic similarity and log(degree) to resist dilution at high-degree nodes."""
    if os.getenv("NO_SIM_FACTOR", "0") == "1":
        sim = 1.0
    deg = index.graph.degree(vid)
    return base_weight * sim * (1.0 + math.log(deg + 1))


# ── _get_bridge_query_sims (donor line 1000) ────────────────────────────────
def _get_bridge_query_sims(index, discovered: Dict[str, Tuple[int, float, float]],
                           query: str) -> Dict[int, float]:
    """For each bridge entity, compute similarity of 'entity | query' against pair embeddings.

    Returns {vid: max_pair_sim}. This measures how relevant the bridge entity is
    to the query in the context of existing entity relationships.
    """
    if index.pair_embeddings is None or len(index.pair_embeddings) == 0:
        return {vid: 1.0 for name, (vid, sim, dr) in discovered.items()}

    bridge_texts = []
    bridge_vids = []
    for name, (vid, sim, disc_round) in discovered.items():
        bridge_texts.append(f"{name} | {query}")
        bridge_vids.append(vid)

    if not bridge_texts:
        return {}

    bridge_embs = index.embedding_model.batch_encode(bridge_texts)
    if bridge_embs.ndim == 1:
        bridge_embs = bridge_embs.reshape(1, -1)

    b_norms = np.linalg.norm(bridge_embs, axis=1, keepdims=True)
    b_norms = np.where(b_norms == 0, 1, b_norms)
    bridge_embs = bridge_embs / b_norms

    p_norms = np.linalg.norm(index.pair_embeddings, axis=1, keepdims=True)
    p_norms = np.where(p_norms == 0, 1, p_norms)
    normed_pairs = index.pair_embeddings / p_norms

    sims = bridge_embs @ normed_pairs.T
    max_sims = sims.max(axis=1)

    result = {}
    for i, vid in enumerate(bridge_vids):
        result[vid] = float(max(max_sims[i], 0.0))

    return result


# ── _get_existing_seed_ids (donor line 1044) ────────────────────────────────
def _get_existing_seed_ids(index, base_node_weights: np.ndarray) -> List[int]:
    """Extract entity vertex IDs that have non-zero weight in base PPR seeds."""
    passage_idx_set = set(index.passage_node_idxs)
    seed_ids = []
    for vid in range(len(base_node_weights)):
        if base_node_weights[vid] > 0 and vid not in passage_idx_set:
            seed_ids.append(vid)
    return seed_ids


# ── _mini_ppr_select_seeds (donor line 1054) ────────────────────────────────
def _mini_ppr_select_seeds(
    index,
    discovered_vertex_ids: Dict[str, int],
    existing_seed_vertex_ids: List[int],
    round0_top_doc_ids: List[int] = None,
    graph=None,
) -> List[Tuple[int, int, float]]:
    """Run mini-PPR from bridge entities on local 5-hop subgraph.

    discovered_vertex_ids: {name: (vid, weight)} where weight incorporates decay.
    Returns list of (bridge_vid, seed_vid, ppr_score) for selected connections.
    """
    graph = graph if graph is not None else index.graph
    passage_idx_set = set(index.passage_node_idxs)

    # Build subgraph from round 0 top docs' entities + bridge + seeds
    core_vids = set(vid for vid, w in discovered_vertex_ids.values()) | set(existing_seed_vertex_ids)
    if round0_top_doc_ids is not None:
        for doc_id in round0_top_doc_ids:
            if doc_id < len(index.passage_node_idxs):
                p_vid = index.passage_node_idxs[doc_id]
                for n in graph.neighbors(p_vid):
                    if n not in passage_idx_set:
                        core_vids.add(n)

    subgraph_vids = set(core_vids)
    for hop in range(5):
        frontier = set()
        for vid in subgraph_vids:
            for n in graph.neighbors(vid):
                if n not in passage_idx_set:
                    frontier.add(n)
        subgraph_vids |= frontier

    subgraph_vids = sorted(subgraph_vids)
    if len(subgraph_vids) < 2:
        return []

    sub = graph.subgraph(subgraph_vids)
    vid_to_sub = {vid: i for i, vid in enumerate(subgraph_vids)}
    n_sub = len(subgraph_vids)

    reset = np.zeros(n_sub)
    for d_vid, d_weight in discovered_vertex_ids.values():
        if d_vid in vid_to_sub:
            reset[vid_to_sub[d_vid]] = d_weight
    if reset.sum() == 0:
        return []
    reset /= reset.sum()

    ppr_scores = sub.personalized_pagerank(
        vertices=range(n_sub),
        damping=EXPANSION_DAMPING,
        directed=False,
        weights='weight' if 'weight' in sub.es.attributes() else None,
        reset=reset,
        implementation='prpack',
    )

    selected = []
    bridge_vids = set(vid for vid, w in discovered_vertex_ids.values())
    seed_scores = []
    for s_vid in existing_seed_vertex_ids:
        if s_vid not in bridge_vids and s_vid in vid_to_sub:
            score = ppr_scores[vid_to_sub[s_vid]]
            seed_scores.append((s_vid, score))
    seed_scores.sort(key=lambda x: x[1], reverse=True)

    log_items = [(index.graph.vs[sv]["content"][:30], f"{sc:.6f}") for sv, sc in seed_scores[:8]]
    logger.info(f"  Mini-PPR(subgraph, joint) -> seeds: {log_items}")

    for s_vid, score in seed_scores:
        if score >= MINI_PPR_THRESHOLD:
            for d_vid in bridge_vids:
                selected.append((d_vid, s_vid, score))

    return selected


# ── _build_overlay_graph (donor line 1178) ──────────────────────────────────
def _build_overlay_graph(index, temp_edges: List[Tuple[int, int, float]], base_graph=None):
    """Create a copy of the graph with temporary bridge edges added."""
    graph = base_graph if base_graph is not None else index.graph
    g = graph.copy()
    if temp_edges:
        edges = [(e[0], e[1]) for e in temp_edges]
        weights = [e[2] for e in temp_edges]
        g.add_edges(edges, attributes={"weight": weights})
    return g


# ── iterative_retrieve (donor line 1253) — MAIN REASONING LOOP ──────────────
def iterative_retrieve(index, question: str, llm_client, max_rounds: int = 3,
                       gold_docs: List[str] = None, query_entities: Optional[List[str]] = None,
                       entity_top_k: int = 5):
    """Multi-round retrieval with one retrieval pass per round.

    Core semantics:
      1. Each round: single PPR on full graph + temp edges from all_discovered
      2. Seed weights decay across rounds (decay=0.5 per round since discovery)
      3. No hop-5 subgraph
      4. RRF accumulates doc scores across rounds
    """
    SEED_DECAY = float(os.getenv("SEED_DECAY", "0.5"))
    num_docs = len(index.passage_keys)
    rrf_k = 60
    rrf_scores = np.zeros(num_docs)
    current_query = question
    reasoning_traces = []
    round_diagnostics = []
    all_discovered = {}
    base_node_weights = None
    round0_top_doc_ids = None
    combined_scores = None

    for round_i in range(max_rounds):
        logger.info(f"  Round {round_i}: query='{current_query[:60]}...'")

        # Round 1+: enrich query embedding with key entities from current query
        if round_i > 0:
            round_entities = query_ner(current_query, llm_client)
            if round_entities:
                retrieve_query = " | ".join(round_entities) + " | " + current_query
                logger.info(f"  Enriched query: '{retrieve_query[:80]}...'")
            else:
                retrieve_query = current_query
        else:
            retrieve_query = current_query

        # Build this-round temp edges from all_discovered (freshly computed, not accumulated)
        temp_edges = []
        extra_node_weights = None
        if all_discovered and base_node_weights is not None:
            query_sims = _get_bridge_query_sims(index, all_discovered, retrieve_query)

            vid_to_info = {}
            discovered_with_decay = {}
            for name, (vid, res_sim, disc_round) in all_discovered.items():
                decay = SEED_DECAY ** (round_i - disc_round)
                q_sim = query_sims.get(vid, 1.0)
                vid_to_info[vid] = (q_sim, decay)
                discovered_with_decay[name] = (vid, decay)

            existing_seeds = _get_existing_seed_ids(index, base_node_weights)
            ppr_connections = _mini_ppr_select_seeds(
                index,
                discovered_vertex_ids=discovered_with_decay,
                existing_seed_vertex_ids=existing_seeds,
                round0_top_doc_ids=round0_top_doc_ids,
            )

            for d_vid, s_vid, _ in ppr_connections:
                d_qsim, d_decay = vid_to_info.get(d_vid, (1.0, 1.0))
                bridge_weight = _degree_adaptive_weight(index, d_vid, 1.0, sim=d_qsim) * d_decay
                temp_edges.append((d_vid, s_vid, bridge_weight))

            by_round = {}
            for name, (vid, res_sim, disc_round) in all_discovered.items():
                q_sim = query_sims.get(vid, 1.0)
                by_round.setdefault(disc_round, []).append((vid, q_sim, disc_round))
            for dr, entities in by_round.items():
                decay = SEED_DECAY ** (round_i - dr)
                for i, (v1, s1, _) in enumerate(entities):
                    for v2, s2, _ in entities[i + 1:]:
                        w = max(_degree_adaptive_weight(index, v1, 1.0, sim=s1),
                                _degree_adaptive_weight(index, v2, 1.0, sim=s2)) * decay
                        temp_edges.append((v1, v2, w))

            extra_node_weights = np.zeros(index.graph.vcount())
            for name, (vid, res_sim, disc_round) in all_discovered.items():
                decay = SEED_DECAY ** (round_i - disc_round)
                q_sim = query_sims.get(vid, 1.0)
                extra_node_weights[vid] += _degree_adaptive_weight(index, vid, DEFAULT_ENTITY_SEED_WEIGHT, sim=q_sim) * decay

            logger.info(f"  Overlay: {len(all_discovered)} bridge entities, {len(temp_edges)} temp edges")

        working_graph = _build_overlay_graph(index, temp_edges) if temp_edges else index.graph

        sorted_doc_ids, sorted_doc_scores, current_node_weights = index.retrieve(
            retrieve_query,
            working_graph=working_graph,
            extra_node_weights=extra_node_weights,
            return_node_weights=True,
            query_entities=None,
        )
        base_top_docs = [index.passages[index.passage_keys[idx]] for idx in sorted_doc_ids[:10]]

        if round_i == 0:
            base_node_weights = current_node_weights
            round0_top_doc_ids = [int(d) for d in sorted_doc_ids[:20]]

        current_rrf = np.zeros(num_docs)
        for rank, doc_id in enumerate(sorted_doc_ids):
            current_rrf[doc_id] = 1.0 / (rrf_k + rank + 1)

        HISTORY_WEIGHT = 0.3
        CURRENT_WEIGHT = 1.0
        history_norm = rrf_scores / max(round_i, 1)
        combined_scores = HISTORY_WEIGHT * history_norm + CURRENT_WEIGHT * current_rrf

        rrf_scores += current_rrf

        final_sorted_ids = np.argsort(combined_scores)[::-1]
        top_docs = [index.passages[index.passage_keys[idx]] for idx in final_sorted_ids[:10]]

        round_diag = {
            "round_idx": round_i,
            "query": current_query,
            "base_top_doc_ids": [int(idx) for idx in sorted_doc_ids[:10]],
            "rrf_top_doc_ids": [int(idx) for idx in final_sorted_ids[:10]],
            "bridge_edges": len(temp_edges),
            "discovered_entities_total": sorted(all_discovered.keys()),
        }
        if gold_docs is not None:
            round_diag["base_recall"] = {
                "R@1": recall_at_k(base_top_docs, gold_docs, 1),
                "R@2": recall_at_k(base_top_docs, gold_docs, 2),
                "R@5": recall_at_k(base_top_docs, gold_docs, 5),
                "R@10": recall_at_k(base_top_docs, gold_docs, 10),
            }
            round_diag["rrf_recall"] = {
                "R@1": recall_at_k(top_docs, gold_docs, 1),
                "R@2": recall_at_k(top_docs, gold_docs, 2),
                "R@5": recall_at_k(top_docs, gold_docs, 5),
                "R@10": recall_at_k(top_docs, gold_docs, 10),
            }

        if round_i == max_rounds - 1:
            round_diag["stop"] = False
            round_diag["rewritten_query"] = ""
            round_diag["new_discovered_entities"] = []
            round_diagnostics.append(round_diag)
            break

        try:
            reasoning_output = reason_and_rewrite(
                original_query=question,
                current_query=current_query,
                retrieved_docs=top_docs,
                round_idx=round_i,
                previous_traces=reasoning_traces,
                llm_client=llm_client,
            )
        except Exception as e:
            logger.warning(f"  Reasoning error at round {round_i}: {e}")
            break

        analysis_text = reasoning_output.get("analysis", "")
        scenario_text = reasoning_output.get("context_scenario", "")
        if scenario_text:
            trace_text = f"{analysis_text} | Hypothesized context: {scenario_text}"
        else:
            trace_text = analysis_text
        reasoning_traces.append(trace_text)
        round_diag["rewritten_query"] = reasoning_output.get("rewritten_query", "")
        round_diag["new_discovered_entities"] = reasoning_output.get("discovered_entities", [])
        round_diag["context_scenario"] = scenario_text
        round_diag["stop"] = reasoning_output.get("should_stop", False)

        if reasoning_output.get("should_stop", False):
            logger.info(f"  Reasoning stop at round {round_i}")
            round_diagnostics.append(round_diag)
            break

        new_query = reasoning_output.get("rewritten_query", "")
        if new_query and new_query != current_query:
            logger.info(f"  Round {round_i} rewrite: '{new_query[:60]}...'")
            current_query = new_query

        discovered_entities = reasoning_output.get("discovered_entities", [])
        if discovered_entities:
            resolved = _resolve_entities_in_graph(index, discovered_entities)
            for name, (vid, sim) in resolved.items():
                if name not in all_discovered:
                    all_discovered[name] = (vid, sim, round_i)
                else:
                    old_vid, old_sim, old_round = all_discovered[name]
                    avg_round = (old_round + round_i) / 2.0
                    all_discovered[name] = (old_vid, max(old_sim, sim), avg_round)
            logger.info(f"  Resolved {len(resolved)}/{len(discovered_entities)} bridge entities in graph")
            round_diag["discovered_entities_total"] = sorted(all_discovered.keys())

        round_diagnostics.append(round_diag)

    final_sorted_ids = np.argsort(combined_scores)[::-1]
    retrieved_docs = [index.passages[index.passage_keys[idx]] for idx in final_sorted_ids]
    final_doc_scores = combined_scores[final_sorted_ids]

    return retrieved_docs, final_doc_scores, reasoning_traces, round_diagnostics


# ============================================================================
# Section 3: HippoRAG → NER-style index adapter (NEW)
# Translates HippoRAG attributes/methods to the interface the canonical
# functions above expect. No canonical code is modified.
# ============================================================================

class _LazyPassageDict:
    """Dict-like access to passage text via hipporag.chunk_embedding_store."""
    def __init__(self, hipporag):
        self.hipporag = hipporag

    def __getitem__(self, key):
        return self.hipporag.chunk_embedding_store.get_row(key)["content"]


class HippoRAGIndexAdapter:
    """Wraps a HippoRAG instance to expose the interface the canonical NER
    reasoning functions expect (`index.graph`, `index.retrieve`, ...).

    Attributes mirrored from hipporag:
      - graph                 ← hipporag.graph
      - passage_node_idxs     ← hipporag.passage_node_idxs
      - passage_keys          ← hipporag.passage_node_keys (rename)
      - passages[key]         ← chunk_embedding_store.get_row(key)["content"]
      - node_name_to_idx      ← hipporag.node_name_to_vertex_idx (rename)
      - embedding_model       ← hipporag.embedding_model
      - entity_embeddings     ← lazy-loaded from entity_embedding_store
      - entity_keys           ← lazy-loaded from entity_embedding_store
      - pair_embeddings = None  (HippoRAG has none → forces sim=1.0 via fallback)

    Method:
      - retrieve(query, working_graph, extra_node_weights, return_node_weights, query_entities)
        wraps hipporag.get_query_embeddings + get_fact_scores + rerank_facts +
        graph_search_with_fact_entities.
    """
    def __init__(self, hipporag):
        self.hipporag = hipporag
        self.graph = hipporag.graph
        self.passage_node_idxs = hipporag.passage_node_idxs
        self.passage_keys = hipporag.passage_node_keys
        self.passages = _LazyPassageDict(hipporag)
        self.node_name_to_idx = hipporag.node_name_to_vertex_idx
        self.embedding_model = hipporag.embedding_model
        # HippoRAG has no pair_embeddings → canonical _get_bridge_query_sims
        # falls back to {vid: 1.0}, equivalent to NO_SIM_FACTOR=1.
        self.pair_embeddings = None
        # Lazy-loaded
        self._entity_keys_cache = None
        self._entity_embeddings_cache = None

    @property
    def entity_keys(self):
        if self._entity_keys_cache is None:
            self._entity_keys_cache = list(
                self.hipporag.entity_embedding_store.get_all_id_to_rows().keys()
            )
        return self._entity_keys_cache

    @property
    def entity_embeddings(self):
        if self._entity_embeddings_cache is None:
            keys = self.entity_keys
            if not keys:
                self._entity_embeddings_cache = np.array([])
            else:
                self._entity_embeddings_cache = self.hipporag.entity_embedding_store.get_embeddings(keys)
        return self._entity_embeddings_cache

    def retrieve(self, query, working_graph=None, extra_node_weights=None,
                 return_node_weights=False, query_entities=None):
        """Mirror of NER index.retrieve(...). query_entities is accepted but ignored
        (HippoRAG drives entity linking through facts, not user-supplied entities).

        Returns (sorted_doc_ids, sorted_doc_scores) or (..., node_weights) if
        return_node_weights=True.
        """
        hip = self.hipporag
        hip.get_query_embeddings([query])
        fact_scores = hip.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, _ = hip.rerank_facts(query, fact_scores)

        if len(top_k_facts) == 0:
            sorted_ids, sorted_scores = hip.dense_passage_retrieval(query)
            if return_node_weights:
                return sorted_ids, sorted_scores, None
            return sorted_ids, sorted_scores

        return hip.graph_search_with_fact_entities(
            query=query,
            link_top_k=hip.global_config.linking_top_k,
            query_fact_scores=fact_scores,
            top_k_facts=top_k_facts,
            top_k_fact_indices=top_k_fact_indices,
            passage_node_weight=hip.global_config.passage_node_weight,
            extra_node_weights=extra_node_weights,
            working_graph=working_graph,
            return_node_weights=return_node_weights,
        )


# ============================================================================
# Section 4: save_results (adapted from ircot)
# ============================================================================

def save_results(output_path, config, all_results):
    n = len(all_results)
    if n == 0:
        return

    baseline_em = sum(
        1 for r in all_results
        if check_em(r["baseline_answer"], r["gold_answer"], r.get("gold_aliases", []))
    ) / n
    bridge_em = sum(
        1 for r in all_results
        if check_em(r["bridge_answer"], r["gold_answer"], r.get("gold_aliases", []))
    ) / n

    baseline_f1_vals = [r.get("baseline_qa", {}).get("F1") for r in all_results]
    bridge_f1_vals = [r.get("bridge_qa", {}).get("F1") for r in all_results]
    baseline_f1_vals = [v for v in baseline_f1_vals if v is not None]
    bridge_f1_vals = [v for v in bridge_f1_vals if v is not None]
    baseline_f1 = round(float(np.mean(baseline_f1_vals)), 4) if baseline_f1_vals else None
    bridge_f1 = round(float(np.mean(bridge_f1_vals)), 4) if bridge_f1_vals else None

    recall_keys = ["Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20"]
    baseline_recall = {}
    bridge_recall = {}
    for key in recall_keys:
        b_vals = [r.get("baseline_retrieval", {}).get(key) for r in all_results]
        i_vals = [r.get("bridge_retrieval", {}).get(key) for r in all_results]
        b_vals = [v for v in b_vals if v is not None]
        i_vals = [v for v in i_vals if v is not None]
        if b_vals:
            baseline_recall[key] = round(float(np.mean(b_vals)), 4)
        if i_vals:
            bridge_recall[key] = round(float(np.mean(i_vals)), 4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": config,
                "summary": {
                    "n_completed": n,
                    "baseline_em": round(baseline_em, 4),
                    "bridge_em": round(bridge_em, 4),
                    "improvement_em": round(bridge_em - baseline_em, 4),
                    "baseline_f1": baseline_f1,
                    "bridge_f1": bridge_f1,
                    "improvement_f1": round(bridge_f1 - baseline_f1, 4) if baseline_f1 and bridge_f1 else None,
                    "baseline_recall": baseline_recall,
                    "bridge_recall": bridge_recall,
                },
                "results": all_results,
            },
            f,
            indent=2,
            default=make_serializable,
        )


# ============================================================================
# Section 5: main (adapted from evaluate_musique_ircot.py)
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="musique.json")
    parser.add_argument("--sample_limit", type=int, default=5)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--openie_cache", type=str, default="outputs/musique/openie_results_ner_qwen-plus.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default per-sample → outputs/musique_hipporag_bridge_eval, "
                             "global → outputs/musique_hipporag_bridge_global_eval")
    parser.add_argument("--global_index", action="store_true",
                        help="Build a SINGLE HippoRAG over the full corpus (paper setup). "
                             "Default is per-sample (~20 docs per query, easier).")
    parser.add_argument("--corpus_path", type=str,
                        default="reproduce/dataset/musique_corpus.json",
                        help="Full corpus JSON. Used when --global_index is set.")
    args = parser.parse_args()

    (HippoRAG, QuerySolution, compute_mdhash_id, _get_embedding_model_class,
     PromptTemplateManager, RetrievalRecall) = _lazy_imports()

    random.seed(42)
    np.random.seed(42)
    import torch
    torch.manual_seed(42)

    data = load_musique_data(args.data_path)
    if args.sample_limit and args.sample_limit < len(data):
        data = data[:args.sample_limit]

    logger.info(f"Loaded {len(data)} samples")

    llm_model_name = os.getenv("LLM_MODEL_NAME", "qwen-plus")
    embedding_model_name = os.getenv(
        "EMBEDDING_MODEL_NAME", "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    )
    aliyun_base_url = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        os.environ["OPENAI_API_KEY"] = "sk-396199ed7af84eff8a0cf7a71b797601"

    emb_key = embedding_model_name.split("/")[-1].replace(" ", "_")
    shared_graph_dir = os.path.join("outputs", f"musique_shared_{emb_key}")
    # save_dir auto-picks based on global flag if user didn't specify
    if args.output_dir:
        save_dir = args.output_dir
    elif args.global_index:
        save_dir = "outputs/musique_hipporag_bridge_global_eval"
    else:
        save_dir = "outputs/musique_hipporag_bridge_eval"
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "method": "hipporag_bridge_reasoning",
        "max_rounds": args.max_rounds,
        "sample_limit": args.sample_limit,
        "llm": llm_model_name,
        "embedding": embedding_model_name,
        "openie_cache": args.openie_cache,
        "NO_SIM_FACTOR": os.getenv("NO_SIM_FACTOR", "1"),
        "SEED_DECAY": os.getenv("SEED_DECAY", "0.5"),
        "REWRITE_FIRST": os.getenv("REWRITE_FIRST", "0"),
        "global_index": bool(args.global_index),
    }

    # Load OpenIE cache
    openie_text_lookup = {}
    if args.openie_cache:
        logger.info(f"Using pre-computed OpenIE cache: {args.openie_cache}")
        cache_data = json.load(open(args.openie_cache))
        for doc in cache_data["docs"]:
            if "text" in doc:
                text_key = doc["text"]
            else:
                parts = doc["passage"].split("\n", 1)
                text_key = parts[1] if len(parts) > 1 else doc["passage"]
            openie_text_lookup[text_key] = doc
        logger.info(f"Loaded {len(openie_text_lookup)} docs from OpenIE cache")

    output_path = os.path.join(save_dir, "comparison_results.json")
    all_results = []

    if os.path.exists(output_path):
        try:
            existing = json.load(open(output_path))
            all_results = existing.get("results", [])
            logger.info(f"Resuming from {len(all_results)} completed samples")
        except Exception:
            all_results = []

    logger.info(f"Pre-loading embedding model: {embedding_model_name}")
    shared_embedding_model = _get_embedding_model_class(
        embedding_model_name=embedding_model_name
    )(embedding_model_name=embedding_model_name)
    logger.info("Embedding model loaded")

    # ── Global HippoRAG (paper setup): build ONCE on the full corpus ─────────
    # Mirrors evaluate_musique_iterretgen.py — uses musique_corpus.json (11656 docs).
    global_hipporag = None
    if args.global_index:
        if not os.path.exists(args.corpus_path):
            raise FileNotFoundError(
                f"--global_index needs corpus at {args.corpus_path} "
                "(or set --corpus_path)."
            )
        corpus_data = json.load(open(args.corpus_path))
        full_corpus_docs = [f"{c['title']}\n{c['text']}" for c in corpus_data]
        logger.info(f"Global mode: loaded {len(full_corpus_docs)} corpus docs from {args.corpus_path}")

        _data_name = os.path.splitext(os.path.basename(args.data_path))[0]
        llm_safe = llm_model_name.replace("/", "_").replace(" ", "_")
        global_hipporag_dir = os.path.join(
            "outputs", f"{_data_name}_hipporag_fullcorpus_{llm_safe}_{emb_key}"
        )
        os.makedirs(global_hipporag_dir, exist_ok=True)
        logger.info(f"Global HippoRAG save_dir: {global_hipporag_dir}")

        global_hipporag = HippoRAG(
            save_dir=global_hipporag_dir,
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            llm_base_url=aliyun_base_url,
            embedding_model=shared_embedding_model,
        )

        # Inject OpenIE cache for the full corpus (match by exact text + whitespace fallback)
        if openie_text_lookup:
            cache_dest = global_hipporag.openie_results_path
            matched_docs = []
            openie_nows_lookup = {}
            import re as _re_local
            for k, v in openie_text_lookup.items():
                openie_nows_lookup[_re_local.sub(r"\s+", "", k)] = v
            exact_hits = 0
            nows_hits = 0
            for doc_text in full_corpus_docs:
                cached = openie_text_lookup.get(doc_text)
                if cached is not None:
                    exact_hits += 1
                else:
                    cached = openie_nows_lookup.get(_re_local.sub(r"\s+", "", doc_text))
                    if cached is not None:
                        nows_hits += 1
                if cached is not None:
                    matched_docs.append({
                        "idx": compute_mdhash_id(doc_text, prefix="chunk-"),
                        "passage": doc_text,
                        "extracted_entities": cached["extracted_entities"],
                        "extracted_triples": cached.get("extracted_triples", []),
                    })
            logger.info(
                f"  OpenIE match: {len(matched_docs)}/{len(full_corpus_docs)} "
                f"(exact={exact_hits}, nows={nows_hits})"
            )
            if matched_docs:
                with open(cache_dest, "w") as f:
                    json.dump({"docs": matched_docs}, f)
                graph_pickle = global_hipporag._graph_pickle_filename
                if os.path.exists(graph_pickle):
                    if os.path.getmtime(cache_dest) > os.path.getmtime(graph_pickle):
                        os.remove(graph_pickle)
                        logger.info("  Removed stale graph.pickle (older than OpenIE cache)")

        logger.info(f"Indexing global HippoRAG ({len(full_corpus_docs)} docs) — once...")
        global_hipporag.index(docs=full_corpus_docs)
        logger.info("Global HippoRAG ready.")

    SAMPLE_TIMEOUT = 300

    def _timeout_handler(signum, frame):
        raise TimeoutError("Sample timed out")

    total_start = time.time()

    for idx, sample in enumerate(data):
        if idx < len(all_results):
            continue

        question = sample.get("question", "")
        paragraphs = sample.get("paragraphs", [])
        answer = sample.get("answer", "")
        answer_aliases = sample.get("answer_aliases", [])
        docs = [para.get("paragraph_text", "") for para in paragraphs]
        gold_docs_list = [
            para.get("paragraph_text", "")
            for para in paragraphs
            if para.get("is_supporting", False)
        ]
        gold_answers = [answer] + answer_aliases

        per_sample_dir = os.path.join(shared_graph_dir, f"sample_{idx:06d}")
        logger.info(f"[{idx+1}/{len(data)}] {question[:80]}...")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(SAMPLE_TIMEOUT)

        try:
            if global_hipporag is not None:
                # Reuse the global HippoRAG built once before the loop.
                # Skip per-sample indexing entirely.
                hipporag = global_hipporag
            else:
                hipporag = HippoRAG(
                    save_dir=per_sample_dir,
                    llm_model_name=llm_model_name,
                    embedding_model_name=embedding_model_name,
                    llm_base_url=aliyun_base_url,
                    embedding_model=shared_embedding_model,
                )
            if global_hipporag is None and openie_text_lookup:
                os.makedirs(per_sample_dir, exist_ok=True)
                cache_dest = hipporag.openie_results_path
                matched_docs = []
                for doc_text in docs:
                    if doc_text in openie_text_lookup:
                        cached = openie_text_lookup[doc_text]
                        new_idx = compute_mdhash_id(doc_text, prefix="chunk-")
                        matched_docs.append({
                            "idx": new_idx,
                            "passage": doc_text,
                            "extracted_entities": cached["extracted_entities"],
                            "extracted_triples": cached.get("extracted_triples", []),
                        })
                if matched_docs:
                    with open(cache_dest, "w") as f:
                        json.dump({"docs": matched_docs}, f)
                    graph_pickle = hipporag._graph_pickle_filename
                    if os.path.exists(graph_pickle):
                        if os.path.getmtime(cache_dest) > os.path.getmtime(graph_pickle):
                            os.remove(graph_pickle)
                            logger.debug(f"  Removed stale graph.pickle (older than OpenIE cache)")
            if global_hipporag is None:
                hipporag.index(docs=docs)
        except TimeoutError:
            logger.error(f"  Sample {idx} timed out during indexing")
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            all_results.append({
                "idx": idx, "question": question, "gold_answer": answer,
                "gold_aliases": answer_aliases,
                "baseline_answer": "Error", "bridge_answer": "Error",
                "baseline_retrieval": {}, "bridge_retrieval": {},
                "baseline_qa": {}, "bridge_qa": {},
                "error": "Timeout during indexing",
            })
            gc.collect()
            continue
        except Exception as e:
            logger.error(f"  Index failed: {e}")
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            all_results.append({
                "idx": idx, "question": question, "gold_answer": answer,
                "gold_aliases": answer_aliases,
                "baseline_answer": "Error", "bridge_answer": "Error",
                "baseline_retrieval": {}, "bridge_retrieval": {},
                "baseline_qa": {}, "bridge_qa": {},
                "error": str(e),
            })
            gc.collect()
            continue

        # --- Baseline: single-pass HippoRAG ---
        t0 = time.time()
        try:
            baseline = hipporag.rag_qa(
                queries=[question],
                gold_docs=[gold_docs_list],
                gold_answers=[gold_answers],
            )
            baseline_answer = baseline[0][0].answer if baseline[0] else "Unknown"
            baseline_retrieval = baseline[3] if len(baseline) > 3 else {}
            baseline_qa = baseline[4] if len(baseline) > 4 else {}
        except Exception as e:
            logger.error(f"  Baseline failed: {e}")
            baseline_answer = "Error"
            baseline_retrieval = {}
            baseline_qa = {}
        baseline_time = time.time() - t0

        signal.alarm(SAMPLE_TIMEOUT)

        # --- HippoRAG + canonical bridge reasoning ---
        t0 = time.time()
        try:
            adapter = HippoRAGIndexAdapter(hipporag)
            retrieved_docs, doc_scores, reasoning_traces, round_diagnostics = iterative_retrieve(
                index=adapter,
                question=question,
                llm_client=hipporag.llm_model,
                max_rounds=args.max_rounds,
                gold_docs=gold_docs_list,
            )

            bridge_solution = QuerySolution(
                question=question,
                docs=retrieved_docs,
                doc_scores=np.array(doc_scores),
            )
            bridge_qa_result = hipporag.rag_qa(
                queries=[bridge_solution],
                gold_docs=[gold_docs_list],
                gold_answers=[gold_answers],
            )
            bridge_answer = bridge_qa_result[0][0].answer if bridge_qa_result[0] else "Unknown"
            bridge_qa_eval = bridge_qa_result[4] if len(bridge_qa_result) > 4 else {}

            recall_evaluator = RetrievalRecall(global_config=hipporag.global_config)
            bridge_recall, _ = recall_evaluator.calculate_metric_scores(
                gold_docs=[gold_docs_list],
                retrieved_docs=[retrieved_docs],
                k_list=[1, 2, 5, 10, 20],
            )
        except TimeoutError:
            logger.error(f"  Bridge reasoning timed out for sample {idx}")
            bridge_answer = "Error"
            bridge_qa_eval = {}
            bridge_recall = {}
            reasoning_traces = []
            round_diagnostics = []
        except Exception as e:
            logger.error(f"  Bridge reasoning failed: {e}")
            logger.error(traceback.format_exc())
            bridge_answer = "Error"
            bridge_qa_eval = {}
            bridge_recall = {}
            reasoning_traces = []
            round_diagnostics = []
        bridge_time = time.time() - t0

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        result = {
            "idx": idx,
            "question": question,
            "gold_answer": answer,
            "gold_aliases": answer_aliases,
            "baseline_answer": baseline_answer,
            "bridge_answer": bridge_answer,
            "baseline_retrieval": baseline_retrieval,
            "bridge_retrieval": bridge_recall,
            "baseline_qa": baseline_qa,
            "bridge_qa": bridge_qa_eval,
            "reasoning_traces": reasoning_traces,
            "round_diagnostics": round_diagnostics,
            "baseline_time": round(baseline_time, 2),
            "bridge_time": round(bridge_time, 2),
        }
        all_results.append(result)

        if global_hipporag is None:
            del hipporag
        gc.collect()

        b_match = "Y" if check_em(baseline_answer, answer, answer_aliases) else "N"
        i_match = "Y" if check_em(bridge_answer, answer, answer_aliases) else "N"
        logger.info(
            f"  B={b_match} '{baseline_answer[:40]}' | Bridge={i_match} '{bridge_answer[:40]}' | "
            f"Gold='{answer}' | t={baseline_time:.1f}s/{bridge_time:.1f}s"
        )

        if (idx + 1) % 5 == 0:
            save_results(output_path, config, all_results)
            n = len(all_results)
            b_em = sum(1 for r in all_results if check_em(r["baseline_answer"], r["gold_answer"], r.get("gold_aliases", []))) / n
            i_em = sum(1 for r in all_results if check_em(r["bridge_answer"], r["gold_answer"], r.get("gold_aliases", []))) / n
            elapsed = time.time() - total_start
            eta = elapsed / n * (len(data) - n)
            logger.info(
                f"  >>> Progress: {n}/{len(data)} | Baseline EM={b_em:.3f} | "
                f"Bridge EM={i_em:.3f} | ETA={eta/60:.0f}min"
            )

    save_results(output_path, config, all_results)

    n = len(all_results)
    if n > 0:
        b_em = sum(1 for r in all_results if check_em(r["baseline_answer"], r["gold_answer"], r.get("gold_aliases", []))) / n
        i_em = sum(1 for r in all_results if check_em(r["bridge_answer"], r["gold_answer"], r.get("gold_aliases", []))) / n
        print(f"\n{'='*60}")
        print(f"HippoRAG + Bridge Reasoning: {n} samples")
        print(f"  Baseline EM: {b_em:.4f}")
        print(f"  Bridge EM:   {i_em:.4f}")
        print(f"  Delta:       {i_em - b_em:+.4f}")
        print(f"  Results: {output_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
