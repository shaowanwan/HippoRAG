import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .round_state import RetrievalRoundState, RoundMetrics
from .query_rewriter import QueryRewriter
from .graph_overlay import GraphOverlay
from ..utils.misc_utils import QuerySolution, compute_mdhash_id
from ..evaluation.retrieval_eval import RetrievalRecall

logger = logging.getLogger(__name__)

# Default weight for discovered entity seeds
DEFAULT_ENTITY_SEED_WEIGHT = 0.5

# RRF round weight: later rounds contribute more
# Round i weight = 1.0 + i * RRF_ROUND_BOOST
RRF_ROUND_BOOST = 0.5

# Damping for seed expansion PPR (higher = bridge entities spread further)
EXPANSION_DAMPING = 0.7

# Edge attention mode: None, "query_conditioned", "graph_propagation"
EDGE_ATTENTION_MODE = "query_conditioned"
# Minimum gate value to prevent completely killing edges
EDGE_GATE_MIN = 0.1

# Per-node damping: True to enable adaptive damping based on reasoning direction
PER_NODE_DAMPING = False
# Base damping and max adjustment range
PER_NODE_DAMPING_BASE = 0.5
PER_NODE_DAMPING_DELTA = 0.2  # nodes aligned with direction: damping -= delta


class ReasoningController:
    """Orchestrates multi-round reasoning-guided retrieval.

    Each round:
      1. Run full HippoRAG retrieval with the current (rewritten) query → RRF
      2. LLM reasons → rewrite query + discover bridge entities
      3. Resolve entities → seed expansion + bridge edges → extra PPR → RRF
    """

    def __init__(self, hipporag, max_rounds: int = 3):
        self.hipporag = hipporag
        self.max_rounds = max_rounds
        self.query_rewriter = QueryRewriter(hipporag.llm_model)

    def iterative_retrieve(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None,
    ) -> Tuple[List[QuerySolution], Dict]:
        if num_to_retrieve is None:
            num_to_retrieve = self.hipporag.global_config.retrieval_top_k

        if not self.hipporag.ready_to_retrieve:
            self.hipporag.prepare_retrieval_objects()

        all_results = []
        all_round_states = []
        total_start = time.time()

        for q_idx, query in enumerate(queries):
            logger.info(f"[ReasoningController] Query {q_idx}: {query[:80]}...")
            state = RetrievalRoundState(
                original_query=query,
                current_query=query,
            )

            q_gold = gold_docs[q_idx] if gold_docs else None
            result = self._iterative_retrieve_single(
                state=state,
                num_to_retrieve=num_to_retrieve,
                gold_docs_for_query=q_gold,
            )
            all_results.append(result)
            all_round_states.append(state)

        total_time = time.time() - total_start

        eval_results = self._aggregate_eval(all_round_states, all_results, gold_docs)
        eval_results["total_time"] = round(total_time, 2)

        return all_results, eval_results, all_round_states

    # ── Entity resolution (from Exp 2) ─────────────────────────────────

    def _find_entity_in_graph(self, entity_name: str) -> Optional[int]:
        """Find an entity in the graph by exact hash match. Returns vertex index or None."""
        hip = self.hipporag
        entity_key = compute_mdhash_id(content=entity_name.lower(), prefix="entity-")
        return hip.node_name_to_vertex_idx.get(entity_key, None)

    def _find_entities_by_embedding(self, entity_names: List[str], threshold: float = 0.6) -> Dict[str, int]:
        """Find entities in the graph via embedding similarity fallback."""
        hip = self.hipporag
        if len(hip.entity_embeddings) == 0 or not entity_names:
            return {}

        query_embeddings = hip.embedding_model.batch_encode(entity_names, norm=True)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        similarities = np.dot(query_embeddings, hip.entity_embeddings.T)

        found = {}
        for i, name in enumerate(entity_names):
            best_idx = np.argmax(similarities[i])
            if similarities[i][best_idx] >= threshold:
                entity_key = hip.entity_node_keys[best_idx]
                vertex_id = hip.node_name_to_vertex_idx.get(entity_key)
                if vertex_id is not None:
                    found[name] = vertex_id
                    logger.debug(f"  Entity '{name}' matched by embedding (sim={similarities[i][best_idx]:.3f})")

        return found

    def _resolve_entities(self, entity_names: List[str]) -> Dict[str, int]:
        """Resolve entity names to graph vertex IDs. Exact match first, then embedding fallback."""
        resolved = {}
        unresolved = []

        for name in entity_names:
            vid = self._find_entity_in_graph(name)
            if vid is not None:
                resolved[name] = vid
                logger.info(f"  Entity '{name}' found (exact) -> vertex {vid}")
            else:
                unresolved.append(name)

        if unresolved:
            embedding_matches = self._find_entities_by_embedding(unresolved)
            for name, vid in embedding_matches.items():
                resolved[name] = vid
                logger.info(f"  Entity '{name}' found (embedding) -> vertex {vid}")
            for name in unresolved:
                if name not in embedding_matches:
                    logger.info(f"  Entity '{name}' NOT found in graph")

        return resolved

    # ── Graph reshape (from Exp 2) ─────────────────────────────────────

    def _get_existing_seed_ids(self, base_node_weights: np.ndarray) -> List[int]:
        """Extract entity vertex IDs that have non-zero weight in base PPR seeds."""
        passage_idx_set = set(self.hipporag.passage_node_idxs)
        seed_ids = []
        for vid in range(len(base_node_weights)):
            if base_node_weights[vid] > 0 and vid not in passage_idx_set:
                seed_ids.append(vid)
        return seed_ids

    def _degree_adaptive_weight(self, vid: int, base_weight: float) -> float:
        """Scale weight by log(degree) to resist dilution at high-degree nodes."""
        deg = self.hipporag.graph.degree(vid)
        return base_weight * (1.0 + math.log(deg + 1))

    def _build_overlay(
        self,
        discovered_vertex_ids: List[int],
        existing_seed_vertex_ids: List[int],
    ) -> Optional[GraphOverlay]:
        """Build GraphOverlay with bridge edges between discovered and existing seed entities."""
        if not discovered_vertex_ids:
            return None

        overlay = GraphOverlay(self.hipporag.graph)

        for d_vid in discovered_vertex_ids:
            bridge_weight = self._degree_adaptive_weight(d_vid, 1.0)
            for s_vid in existing_seed_vertex_ids:
                if d_vid != s_vid:
                    overlay.add_reasoning_edge(d_vid, s_vid, bridge_weight)

        for i, d1 in enumerate(discovered_vertex_ids):
            for d2 in discovered_vertex_ids[i+1:]:
                w = max(self._degree_adaptive_weight(d1, 1.0),
                        self._degree_adaptive_weight(d2, 1.0))
                overlay.add_reasoning_edge(d1, d2, w)

        if overlay.num_temp_edges > 0:
            logger.info(f"  GraphOverlay: {overlay.summary()}")
            return overlay
        return None

    # ── PPR diagnostics ────────────────────────────────────────────────

    def _compute_ppr_diagnostics(self, node_weights: np.ndarray, query: str) -> dict:
        """Compute PPR distribution diagnostics: entropy, concentration, query-subgraph similarity."""
        hip = self.hipporag
        entity_idxs = hip.entity_node_idxs

        # Run PPR to get full score distribution
        ppr_scores = np.array(hip.graph.personalized_pagerank(
            vertices=range(len(hip.node_name_to_vertex_idx)),
            damping=0.5,
            directed=False,
            weights='weight',
            reset=node_weights.tolist(),
            implementation='prpack',
        ))

        # Entropy over entity nodes
        entity_scores = ppr_scores[entity_idxs]
        pos_scores = entity_scores[entity_scores > 0]
        if len(pos_scores) > 0:
            p = pos_scores / pos_scores.sum()
            entropy = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            entropy = 0.0

        # Concentration: top-10 entity share
        total = entity_scores.sum()
        if total > 0:
            concentration = float(np.sort(entity_scores)[-10:].sum() / total)
        else:
            concentration = 0.0

        # Query-subgraph similarity
        query_emb = hip.embedding_model.batch_encode([query], norm=True)[0]
        entity_embs = hip.entity_embeddings

        # Use all entities with non-zero PPR score, weighted by PPR score
        active_mask = entity_scores > 0
        active_idx = np.where(active_mask)[0]

        if len(active_idx) > 0 and len(entity_embs) > 0:
            active_embs = entity_embs[active_idx]
            active_weights = entity_scores[active_idx]
            active_weights = active_weights / active_weights.sum()

            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
            emb_norms = active_embs / (np.linalg.norm(active_embs, axis=1, keepdims=True) + 1e-12)
            sims = np.dot(emb_norms, query_norm)

            # PPR-weighted avg similarity
            avg_sim = float(np.dot(active_weights, sims))

            # Coverage gap: PPR-weighted centroid vs query
            centroid = np.average(active_embs, axis=0, weights=active_weights)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
            coverage_gap = 1.0 - float(np.dot(query_norm, centroid_norm))
        else:
            avg_sim = 0.0
            coverage_gap = 1.0

        return {
            "entropy": entropy,
            "concentration": concentration,
            "query_sim": avg_sim,
            "coverage_gap": coverage_gap,
        }

    # ── Edge attention ────────────────────────────────────────────────

    def _apply_query_conditioned_edge_attention(
        self,
        original_query_emb: np.ndarray,
        rewritten_query_emb: np.ndarray,
        discovered_entity_vids: List[int] = None,
        graph=None,
    ) -> 'ig.Graph':
        """Apply tri-signal edge attention: original query + rewritten query + discovered entities.

        Three signals:
        1. Reasoning direction (rewritten - original): what's NEW to explore
        2. Rewritten query: overall relevance to current retrieval goal
        3. Discovered entity centroid: proximity to bridge anchors

        gate = 1.0 + α * max(0, cos(direction, edge))
                    + β * max(0, cos(rewritten, edge))
                    + γ * max(0, cos(entity_centroid, edge))

        Only boosts, never suppresses (gate >= 1.0).
        Returns a copy of the graph with modified edge weights.
        """
        hip = self.hipporag
        g = (graph if graph is not None else hip.graph).copy()
        entity_embs = hip.entity_embeddings
        entity_keys = hip.entity_node_keys

        # Build vertex_id -> entity embedding index mapping
        vid_to_emb_idx = {}
        for emb_idx, node_key in enumerate(entity_keys):
            vid = hip.node_name_to_vertex_idx.get(node_key)
            if vid is not None:
                vid_to_emb_idx[vid] = emb_idx

        # Signal 1: Reasoning direction via orthogonal decomposition
        # Remove the original query's component from the rewritten query,
        # leaving only the purely "new" semantic direction
        orig_norm = original_query_emb / (np.linalg.norm(original_query_emb) + 1e-12)
        proj = np.dot(rewritten_query_emb, orig_norm) * orig_norm
        orthogonal = rewritten_query_emb - proj  # component orthogonal to original query
        dir_norm_val = np.linalg.norm(orthogonal)
        has_direction = dir_norm_val > 1e-8
        direction_norm = orthogonal / (dir_norm_val + 1e-12) if has_direction else None

        # Signal 2: Rewritten query
        rewritten_norm = rewritten_query_emb / (np.linalg.norm(rewritten_query_emb) + 1e-12)

        # Signal 3: Discovered entity embeddings (for max-pooling similarity)
        discovered_emb_norms = []
        if discovered_entity_vids:
            for vid in discovered_entity_vids:
                emb_idx = vid_to_emb_idx.get(vid)
                if emb_idx is not None:
                    e = entity_embs[emb_idx]
                    discovered_emb_norms.append(e / (np.linalg.norm(e) + 1e-12))

        if not has_direction and not discovered_emb_norms:
            # No signal to gate with
            return g

        for eid in range(g.ecount()):
            edge = g.es[eid]
            src, tgt = edge.source, edge.target

            src_emb_idx = vid_to_emb_idx.get(src)
            tgt_emb_idx = vid_to_emb_idx.get(tgt)

            if src_emb_idx is None and tgt_emb_idx is None:
                continue

            # Compute edge representation
            if src_emb_idx is not None and tgt_emb_idx is not None:
                edge_repr = (entity_embs[src_emb_idx] + entity_embs[tgt_emb_idx]) / 2
            elif src_emb_idx is not None:
                edge_repr = entity_embs[src_emb_idx]
            else:
                edge_repr = entity_embs[tgt_emb_idx]

            edge_repr_norm = edge_repr / (np.linalg.norm(edge_repr) + 1e-12)

            gate = 1.0

            # α: reasoning direction alignment (orthogonal component)
            # Temperature-scaled sigmoid: sharpen cosine differences
            if direction_norm is not None:
                dir_score = float(np.dot(direction_norm, edge_repr_norm))
                dir_gate = 1.0 / (1.0 + math.exp(-dir_score / 0.15))  # τ=0.15
                gate += max(0, dir_gate - 0.5) * 1.0  # only boost, max +0.5

            # β: rewritten query relevance
            rel_score = float(np.dot(rewritten_norm, edge_repr_norm))
            rel_gate = 1.0 / (1.0 + math.exp(-rel_score / 0.15))
            gate += max(0, rel_gate - 0.5) * 0.4  # only boost, max +0.2

            # γ: max similarity to any discovered entity
            if discovered_emb_norms:
                max_ent_sim = max(float(np.dot(e_norm, edge_repr_norm)) for e_norm in discovered_emb_norms)
                ent_gate = 1.0 / (1.0 + math.exp(-max_ent_sim / 0.15))
                gate += max(0, ent_gate - 0.5) * 0.4

            edge["weight"] = edge["weight"] * gate

        return g

    def _apply_graph_propagation_attention(self, query_emb: np.ndarray) -> np.ndarray:
        """Graph embedding propagation: aggregate neighbor embeddings weighted by query attention.

        For each entity node, compute a propagated embedding by attending over its neighbors.
        Then re-compute seed weights as cos(query, propagated_emb).

        Returns new node_weights array for PPR seeds.
        """
        hip = self.hipporag
        entity_embs = hip.entity_embeddings
        entity_keys = hip.entity_node_keys
        entity_node_idxs = hip.entity_node_idxs
        n_nodes = len(hip.node_name_to_vertex_idx)

        # Build vertex_id -> entity embedding index mapping
        vid_to_emb_idx = {}
        for emb_idx, node_key in enumerate(entity_keys):
            vid = hip.node_name_to_vertex_idx.get(node_key)
            if vid is not None:
                vid_to_emb_idx[vid] = emb_idx

        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        emb_dim = entity_embs.shape[1] if len(entity_embs) > 0 else 0

        propagated_embs = {}  # vid -> propagated embedding

        for vid in entity_node_idxs:
            emb_idx = vid_to_emb_idx.get(vid)
            if emb_idx is None:
                continue

            neighbors = hip.graph.neighbors(vid)
            neighbor_embs = []
            for n_vid in neighbors:
                n_emb_idx = vid_to_emb_idx.get(n_vid)
                if n_emb_idx is not None:
                    neighbor_embs.append(entity_embs[n_emb_idx])

            if not neighbor_embs:
                propagated_embs[vid] = entity_embs[emb_idx]
                continue

            neighbor_embs = np.array(neighbor_embs)
            neighbor_norms = neighbor_embs / (np.linalg.norm(neighbor_embs, axis=1, keepdims=True) + 1e-12)

            # Attention weights: softmax of cos(query, neighbor)
            attn_scores = np.dot(neighbor_norms, query_norm)
            # Softmax with temperature
            attn_scores = np.exp(attn_scores - attn_scores.max())
            attn_weights = attn_scores / (attn_scores.sum() + 1e-12)

            # Propagated = weighted sum of neighbor embeddings + self
            propagated = 0.5 * entity_embs[emb_idx] + 0.5 * np.dot(attn_weights, neighbor_embs)
            propagated_embs[vid] = propagated

        # Compute new seed weights
        node_weights = np.zeros(n_nodes)
        for vid, prop_emb in propagated_embs.items():
            prop_norm = prop_emb / (np.linalg.norm(prop_emb) + 1e-12)
            sim = float(np.dot(query_norm, prop_norm))
            node_weights[vid] = max(sim, 0.0)

        return node_weights

    # ── Per-node damping PPR ──────────────────────────────────────────

    def _compute_per_node_damping(
        self,
        original_query_emb: np.ndarray,
        rewritten_query_emb: np.ndarray,
        discovered_vids: List[int] = None,
    ) -> np.ndarray:
        """Compute per-node damping with tri-signal attention.

        Three signals (same as edge attention):
        1. Orthogonal direction alignment → lower damping (spread further in reasoning direction)
        2. Rewritten query relevance → lower damping (relevant nodes propagate more)
        3. Max discovered entity similarity → lower damping (bridge neighborhood spreads)

        All signals use temperature-scaled sigmoid for sharper discrimination.
        Damping is only LOWERED (never raised above base), so non-relevant nodes are unaffected.

        Returns array of damping values per node.
        """
        hip = self.hipporag
        n_nodes = len(hip.node_name_to_vertex_idx)
        base = PER_NODE_DAMPING_BASE
        delta = PER_NODE_DAMPING_DELTA
        damping = np.full(n_nodes, base)

        entity_embs = hip.entity_embeddings
        entity_keys = hip.entity_node_keys

        # Build vertex_id -> entity embedding index mapping
        vid_to_emb_idx = {}
        for emb_idx, node_key in enumerate(entity_keys):
            vid = hip.node_name_to_vertex_idx.get(node_key)
            if vid is not None:
                vid_to_emb_idx[vid] = emb_idx

        # Signal 1: Orthogonal direction
        orig_norm = original_query_emb / (np.linalg.norm(original_query_emb) + 1e-12)
        proj = np.dot(rewritten_query_emb, orig_norm) * orig_norm
        orthogonal = rewritten_query_emb - proj
        dir_norm_val = np.linalg.norm(orthogonal)
        has_direction = dir_norm_val > 1e-8
        direction_norm = orthogonal / (dir_norm_val + 1e-12) if has_direction else None

        # Signal 2: Rewritten query
        rewritten_norm = rewritten_query_emb / (np.linalg.norm(rewritten_query_emb) + 1e-12)

        # Signal 3: Discovered entity embeddings
        discovered_emb_norms = []
        if discovered_vids:
            for vid in discovered_vids:
                emb_idx = vid_to_emb_idx.get(vid)
                if emb_idx is not None:
                    e = entity_embs[emb_idx]
                    discovered_emb_norms.append(e / (np.linalg.norm(e) + 1e-12))

        # Compute per-node damping adjustment
        for emb_idx, node_key in enumerate(entity_keys):
            vid = hip.node_name_to_vertex_idx.get(node_key)
            if vid is None:
                continue

            emb_norm = entity_embs[emb_idx] / (np.linalg.norm(entity_embs[emb_idx]) + 1e-12)
            adjustment = 0.0

            # α: direction alignment
            if direction_norm is not None:
                dir_score = float(np.dot(direction_norm, emb_norm))
                dir_gate = 1.0 / (1.0 + math.exp(-dir_score / 0.15))
                adjustment += max(0, dir_gate - 0.5) * 0.4

            # β: rewritten query relevance
            rel_score = float(np.dot(rewritten_norm, emb_norm))
            rel_gate = 1.0 / (1.0 + math.exp(-rel_score / 0.15))
            adjustment += max(0, rel_gate - 0.5) * 0.3

            # γ: max discovered entity similarity
            if discovered_emb_norms:
                max_sim = max(float(np.dot(e_norm, emb_norm)) for e_norm in discovered_emb_norms)
                ent_gate = 1.0 / (1.0 + math.exp(-max_sim / 0.15))
                adjustment += max(0, ent_gate - 0.5) * 0.5

            # Lower damping = spread further. Adjustment is [0, ~0.6]
            # Scale to damping reduction: max delta reduction
            damping[vid] = base - delta * min(adjustment / 0.6, 1.0)

        return damping

    def _run_ppr_per_node_damping(
        self,
        reset_prob: np.ndarray,
        damping: np.ndarray,
        graph=None,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run PPR with per-node damping via power iteration (sparse matrix).

        Standard PPR: score = α * reset + (1-α) * M @ score
        Per-node:     score[v] = α[v] * reset[v] + (1-α[v]) * Σ_u P(u→v) * score[u]

        Returns (sorted_doc_ids, sorted_doc_scores) same as hip.run_ppr.
        """
        from scipy import sparse

        hip = self.hipporag
        g = graph if graph is not None else hip.graph
        n_nodes = g.vcount()

        # Normalize reset_prob
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        total = reset_prob.sum()
        if total > 0:
            reset_norm = reset_prob / total
        else:
            reset_norm = np.ones(n_nodes) / n_nodes

        # Build sparse transition matrix M where M[j,i] = w(i,j) / out_weight(i)
        has_weight = "weight" in g.es.attributes()
        rows, cols, vals = [], [], []
        out_weights = np.zeros(n_nodes)

        for e in g.es:
            w = e["weight"] if has_weight else 1.0
            s, t = e.source, e.target
            # Undirected: both directions
            out_weights[s] += w
            out_weights[t] += w
            rows.extend([t, s])
            cols.extend([s, t])
            vals.extend([w, w])

        # Normalize by out_weights
        for k in range(len(vals)):
            src = cols[k]
            if out_weights[src] > 0:
                vals[k] /= out_weights[src]

        M = sparse.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))

        # Power iteration
        scores = reset_norm.copy()
        alpha = damping          # per-node damping array
        one_minus_alpha = 1.0 - damping

        for iteration in range(max_iter):
            propagated = M.dot(scores)
            new_scores = alpha * reset_norm + one_minus_alpha * propagated

            diff = np.abs(new_scores - scores).sum()
            scores = new_scores
            if diff < tol:
                logger.debug(f"Per-node damping PPR converged in {iteration+1} iterations")
                break

        # Extract passage scores
        doc_scores = np.array([scores[idx] for idx in hip.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

    # ── Retrieval ──────────────────────────────────────────────────────

    def _retrieve_single_query(self, query: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Run full HippoRAG retrieval for a single query.

        Returns:
            (sorted_doc_ids, sorted_doc_scores, node_weights)
        """
        hip = self.hipporag

        hip.get_query_embeddings([query])

        query_fact_scores = hip.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, rerank_log = hip.rerank_facts(query, query_fact_scores)

        if len(top_k_facts) == 0:
            logger.info("  No facts found, falling back to DPR.")
            sorted_doc_ids, sorted_doc_scores = hip.dense_passage_retrieval(query)
            return sorted_doc_ids, sorted_doc_scores, None
        else:
            sorted_doc_ids, sorted_doc_scores, node_weights = hip.graph_search_with_fact_entities(
                query=query,
                link_top_k=hip.global_config.linking_top_k,
                query_fact_scores=query_fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=hip.global_config.passage_node_weight,
                return_node_weights=True,
            )
            return sorted_doc_ids, sorted_doc_scores, node_weights

    def _iterative_retrieve_single(
        self,
        state: RetrievalRoundState,
        num_to_retrieve: int,
        gold_docs_for_query: Optional[List[str]] = None,
    ) -> QuerySolution:
        """Run iterative retrieval: full pipeline + query rewrite + entity seed expansion + RRF."""
        hip = self.hipporag
        num_docs = len(hip.passage_node_keys)
        rrf_k = 60

        rrf_scores = np.zeros(num_docs)
        base_node_weights = None
        all_discovered = {}  # name -> vertex_id, accumulated across rounds

        for round_i in range(self.max_rounds):
            state.round_idx = round_i
            round_start = time.time()

            logger.info(f"  Round {round_i}: query='{state.current_query[:60]}...'")

            # Step 1: Full retrieval with current query
            sorted_doc_ids, sorted_doc_scores, node_weights = self._retrieve_single_query(state.current_query)
            if node_weights is not None:
                base_node_weights = node_weights

            # Step 2: RRF accumulate (later rounds weighted higher)
            round_weight = 1.0 + round_i * RRF_ROUND_BOOST
            for rank, doc_id in enumerate(sorted_doc_ids):
                rrf_scores[doc_id] += round_weight / (rrf_k + rank + 1)

            # Step 3: If we have discovered entities from previous rounds, run expanded PPR
            if all_discovered and base_node_weights is not None:
                modified_weights = base_node_weights.copy()
                for name, vid in all_discovered.items():
                    modified_weights[vid] += self._degree_adaptive_weight(vid, DEFAULT_ENTITY_SEED_WEIGHT)

                existing_seeds = self._get_existing_seed_ids(base_node_weights)
                overlay = self._build_overlay(
                    discovered_vertex_ids=list(all_discovered.values()),
                    existing_seed_vertex_ids=existing_seeds,
                )

                working_graph = overlay.get_working_graph() if overlay else None

                # Apply edge attention if configured
                if EDGE_ATTENTION_MODE == "query_conditioned":
                    orig_emb = hip.embedding_model.batch_encode([state.original_query], norm=True)[0]
                    rewrite_emb = hip.embedding_model.batch_encode([state.current_query], norm=True)[0]
                    discovered_vids = list(all_discovered.values())
                    working_graph = self._apply_query_conditioned_edge_attention(
                        orig_emb, rewrite_emb,
                        discovered_entity_vids=discovered_vids,
                        graph=working_graph,
                    )
                    logger.info(f"  Applied tri-signal edge attention (dir+rewrite+entities)")
                elif EDGE_ATTENTION_MODE == "graph_propagation":
                    query_emb = hip.embedding_model.batch_encode([state.current_query], norm=True)[0]
                    prop_weights = self._apply_graph_propagation_attention(query_emb)
                    # Merge: keep original seeds + add propagation-derived seeds
                    alpha = 0.3  # blend factor for propagation
                    modified_weights = modified_weights + alpha * prop_weights
                    logger.info(f"  Applied graph propagation attention (alpha={alpha})")

                # Run PPR: per-node damping or standard
                if PER_NODE_DAMPING:
                    orig_emb = hip.embedding_model.batch_encode([state.original_query], norm=True)[0]
                    rewrite_emb = hip.embedding_model.batch_encode([state.current_query], norm=True)[0]
                    node_damping = self._compute_per_node_damping(
                        orig_emb, rewrite_emb,
                        discovered_vids=list(all_discovered.values()),
                    )
                    boosted_doc_ids, boosted_doc_scores = self._run_ppr_per_node_damping(
                        modified_weights,
                        damping=node_damping,
                        graph=working_graph,
                    )
                    logger.info(f"  Per-node damping PPR: damping range [{node_damping.min():.2f}, {node_damping.max():.2f}]")
                else:
                    boosted_doc_ids, boosted_doc_scores = hip.run_ppr(
                        modified_weights,
                        damping=EXPANSION_DAMPING,
                        graph=working_graph,
                    )

                for rank, doc_id in enumerate(boosted_doc_ids):
                    rrf_scores[doc_id] += round_weight / (rrf_k + rank + 1)

                logger.info(
                    f"  Seed expansion PPR: {len(all_discovered)} entities, "
                    f"bridge edges: {overlay.num_temp_edges if overlay else 0}, "
                    f"round_weight: {round_weight:.1f}, "
                    f"per_node_damping: {PER_NODE_DAMPING}"
                )

            # Build current result from RRF scores
            final_sorted_ids = np.argsort(rrf_scores)[::-1]
            top_k_docs = [
                hip.chunk_embedding_store.get_row(hip.passage_node_keys[idx])["content"]
                for idx in final_sorted_ids[:num_to_retrieve]
            ]

            # PPR diagnostics (always against original query for cross-round comparison)
            ppr_diag = {}
            if base_node_weights is not None:
                ppr_diag = self._compute_ppr_diagnostics(base_node_weights, state.original_query)
                logger.info(
                    f"  PPR diag: entropy={ppr_diag['entropy']:.3f}, "
                    f"concentration={ppr_diag['concentration']:.3f}, "
                    f"query_sim={ppr_diag['query_sim']:.3f}, "
                    f"coverage_gap={ppr_diag['coverage_gap']:.3f}"
                )

            # Evaluate
            round_metrics = {}
            if gold_docs_for_query is not None:
                round_metrics = self._evaluate_round(top_k_docs, gold_docs_for_query)

            state.record_round(
                docs=top_k_docs,
                metrics=round_metrics,
                query_used=state.current_query,
            )
            # Attach PPR diagnostics to the round metrics
            if ppr_diag and state.round_metrics:
                rm = state.round_metrics[-1]
                rm.ppr_entropy = ppr_diag.get("entropy", 0.0)
                rm.ppr_concentration = ppr_diag.get("concentration", 0.0)
                rm.ppr_query_sim = ppr_diag.get("query_sim", 0.0)
                rm.ppr_coverage_gap = ppr_diag.get("coverage_gap", 0.0)

            round_time = time.time() - round_start
            logger.info(
                f"  Round {round_i} done ({round_time:.2f}s). "
                f"Metrics: {round_metrics}"
            )

            # Early stop
            if round_metrics.get("Recall@5", 0) >= 1.0:
                logger.info(f"  Recall@5=1.0 at round {round_i}, early stop.")
                break

            if round_i == self.max_rounds - 1:
                break

            # Step 4: LLM reasoning -> rewrite query + discover entities
            reasoning_output = self.query_rewriter.reason_and_rewrite(
                original_query=state.original_query,
                current_query=state.current_query,
                retrieved_docs=top_k_docs[:10],
                round_idx=round_i,
                previous_traces=state.reasoning_traces,
            )

            state.reasoning_traces.append(reasoning_output.get("analysis", ""))

            if reasoning_output.get("should_stop", False):
                logger.info(f"  Reasoning says stop at round {round_i}.")
                break

            # Query rewrite (from Exp 1)
            new_query = reasoning_output.get("rewritten_query", "")
            if new_query and new_query != state.current_query:
                logger.info(f"  Query rewritten: '{new_query[:60]}...'")
                state.current_query = new_query

            # Entity seed expansion (from Exp 2)
            discovered_entities = reasoning_output.get("discovered_entities", [])
            if discovered_entities:
                logger.info(f"  Discovered entities: {discovered_entities}")
                resolved = self._resolve_entities(discovered_entities)
                all_discovered.update(resolved)
                state.discovered_entities.extend(
                    [(name, vid) for name, vid in resolved.items()]
                )
                logger.info(f"  Total resolved entities: {len(all_discovered)}")

        # Final result from RRF scores
        final_sorted_ids = np.argsort(rrf_scores)[::-1]
        final_docs = [
            hip.chunk_embedding_store.get_row(hip.passage_node_keys[idx])["content"]
            for idx in final_sorted_ids[:num_to_retrieve]
        ]
        final_scores = rrf_scores[final_sorted_ids[:num_to_retrieve]].tolist()

        return QuerySolution(
            question=state.original_query,
            docs=final_docs,
            doc_scores=final_scores,
            answer=None,
        )

    def _evaluate_round(
        self, retrieved_docs: List[str], gold_docs: List[str]
    ) -> Dict[str, float]:
        k_list = [1, 2, 5, 10, 20]
        evaluator = RetrievalRecall(global_config=self.hipporag.global_config)
        pooled, _ = evaluator.calculate_metric_scores(
            gold_docs=[gold_docs],
            retrieved_docs=[retrieved_docs],
            k_list=k_list,
        )
        return pooled

    def _aggregate_eval(
        self,
        all_round_states: List[RetrievalRoundState],
        all_results: List[QuerySolution],
        gold_docs: Optional[List[List[str]]],
    ) -> Dict:
        eval_results = {}

        if gold_docs is None:
            eval_results["note"] = "No gold docs provided, skipping eval."
            return eval_results

        max_rounds_used = max(len(s.round_metrics) for s in all_round_states)
        for r in range(max_rounds_used):
            round_metrics_list = []
            for state in all_round_states:
                if r < len(state.round_metrics):
                    round_metrics_list.append(state.round_metrics[r].recall_at_k)
            if round_metrics_list:
                avg_metrics = {}
                for key in round_metrics_list[0]:
                    avg_metrics[key] = round(
                        np.mean([m.get(key, 0) for m in round_metrics_list]), 4
                    )
                eval_results[f"round_{r}"] = avg_metrics

        evaluator = RetrievalRecall(global_config=self.hipporag.global_config)
        k_list = [1, 2, 5, 10, 20, 50, 100]
        final_pooled, _ = evaluator.calculate_metric_scores(
            gold_docs=gold_docs,
            retrieved_docs=[r.docs for r in all_results],
            k_list=k_list,
        )
        eval_results["final"] = final_pooled

        eval_results["avg_rounds_used"] = round(
            np.mean([len(s.round_metrics) for s in all_round_states]), 2
        )

        entities_per_query = [len(s.discovered_entities) for s in all_round_states]
        eval_results["avg_entities_discovered"] = round(np.mean(entities_per_query), 2)

        # PPR diagnostics per query (for cross-round analysis)
        # When running 1 query at a time, this has 1 entry
        for state in all_round_states:
            query_diags = []
            for rm in state.round_metrics:
                query_diags.append({
                    "round": rm.round_idx,
                    "entropy": round(rm.ppr_entropy, 4),
                    "concentration": round(rm.ppr_concentration, 4),
                    "query_sim": round(rm.ppr_query_sim, 4),
                    "coverage_gap": round(rm.ppr_coverage_gap, 4),
                })
            eval_results["ppr_diagnostics"] = query_diags  # per-round list for this query

        return eval_results
