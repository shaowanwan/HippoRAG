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
                    f"round_weight: {round_weight:.1f}"
                )

            # Build current result from RRF scores
            final_sorted_ids = np.argsort(rrf_scores)[::-1]
            top_k_docs = [
                hip.chunk_embedding_store.get_row(hip.passage_node_keys[idx])["content"]
                for idx in final_sorted_ids[:num_to_retrieve]
            ]

            # Evaluate
            round_metrics = {}
            if gold_docs_for_query is not None:
                round_metrics = self._evaluate_round(top_k_docs, gold_docs_for_query)

            state.record_round(
                docs=top_k_docs,
                metrics=round_metrics,
                query_used=state.current_query,
            )

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

        return eval_results
