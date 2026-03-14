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

# LP boost: multiply bridge edge weight when LLM predicted the link
LP_BOOST = 1.5

# Mini-PPR threshold: only build bridge edge if PPR score > this (covers ~6 hops)
PPR_FILTER_THRESHOLD = 0.000001

# RRF weight: scale factor for pipeline RRF relative to PPR (PPR is primary)
RRF_WEIGHT = 0.1





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

    def _build_reverse_index(self):
        """Build vertex_id -> entity_key reverse index (cached)."""
        if not hasattr(self, '_vid_to_key'):
            self._vid_to_key = {}
            self._key_to_emb_idx = {}
            hip = self.hipporag
            for key, idx in hip.node_name_to_vertex_idx.items():
                self._vid_to_key[idx] = key
            for i, key in enumerate(hip.entity_node_keys):
                self._key_to_emb_idx[key] = i

    def _vertex_display_name(self, vid: int) -> str:
        """Get human-readable name for a vertex ID."""
        self._build_reverse_index()
        key = self._vid_to_key.get(vid)
        if key is None:
            return f"entity_{vid}"
        return self.hipporag.entity_embedding_store.hash_id_to_text.get(key, key)

    def _get_entity_embedding(self, vid: int) -> Optional[np.ndarray]:
        """Get the embedding for an entity vertex."""
        self._build_reverse_index()
        key = self._vid_to_key.get(vid)
        if key is None:
            return None
        emb_idx = self._key_to_emb_idx.get(key)
        if emb_idx is None:
            return None
        return self.hipporag.entity_embeddings[emb_idx]

    def _get_doc_facts(self, doc_content: str) -> List[Tuple[str, str]]:
        """Get entity pairs from a document's knowledge graph triples."""
        hip = self.hipporag
        doc_key = compute_mdhash_id(content=doc_content, prefix="passage-")
        doc_vid = hip.node_name_to_vertex_idx.get(doc_key)
        if doc_vid is None:
            return []

        pairs = []
        neighbors = list(hip.graph.neighbors(doc_vid))
        entity_neighbors = []
        passage_idx_set = set(hip.passage_node_idxs)
        for n in neighbors:
            if n not in passage_idx_set:
                entity_neighbors.append(n)

        # Find pairs of entities that are connected in the graph
        for i, e1 in enumerate(entity_neighbors):
            for e2 in entity_neighbors[i+1:]:
                if hip.graph.has_edge(e1, e2):
                    name1 = self._vertex_display_name(e1)
                    name2 = self._vertex_display_name(e2)
                    pairs.append((name1, name2))
                    if len(pairs) >= 10:
                        return pairs
        return pairs

    def _find_important_edges(
        self,
        discovered_vertex_ids: Dict[str, int],
        existing_seed_vertex_ids: List[int],
    ) -> List[Tuple[int, int, float]]:
        """Find important existing edges by diffusing from bridge entities in local subgraph.

        1. Collect bridge + seed entities and their k-hop neighbors → subgraph
        2. Run PPR from bridge entities on subgraph
        3. Edges with high flow (both endpoints have high PPR) are important
        Returns list of (src_vid, dst_vid, importance_score) in the original graph.
        """
        hip = self.hipporag
        graph = hip.graph
        passage_idx_set = set(hip.passage_node_idxs)

        # Step 1: Collect subgraph nodes — bridge + seed + their 2-hop entity neighbors
        core_vids = set(discovered_vertex_ids.values()) | set(existing_seed_vertex_ids)
        subgraph_vids = set(core_vids)

        for hop in range(2):
            frontier = set()
            for vid in subgraph_vids:
                for n in graph.neighbors(vid):
                    if n not in passage_idx_set:  # only entity nodes
                        frontier.add(n)
            subgraph_vids |= frontier

        subgraph_vids = sorted(subgraph_vids)
        if len(subgraph_vids) < 2:
            return []

        # Build local subgraph
        sub = graph.subgraph(subgraph_vids)
        # Map original vid -> subgraph index
        vid_to_sub = {vid: i for i, vid in enumerate(subgraph_vids)}
        sub_to_vid = {i: vid for vid, i in vid_to_sub.items()}

        # Step 2: PPR from bridge entities on subgraph
        n_sub = len(subgraph_vids)
        reset = np.zeros(n_sub)
        for d_vid in discovered_vertex_ids.values():
            if d_vid in vid_to_sub:
                reset[vid_to_sub[d_vid]] = 1.0
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
        ppr_scores = np.array(ppr_scores)

        # Step 3: Score each edge by flow = min(ppr[src], ppr[dst])
        # (both endpoints must have high probability for the edge to be important)
        seen_pairs = set()
        important_edges = []
        for edge in sub.es:
            src_sub, dst_sub = edge.source, edge.target
            src_vid = sub_to_vid[src_sub]
            dst_vid = sub_to_vid[dst_sub]
            pair = (min(src_vid, dst_vid), max(src_vid, dst_vid))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            flow = min(ppr_scores[src_sub], ppr_scores[dst_sub])
            if flow > 1e-5:
                important_edges.append((src_vid, dst_vid, flow))

        important_edges.sort(key=lambda x: x[2], reverse=True)

        # Log
        top_edges_log = []
        for src, dst, flow in important_edges[:10]:
            src_name = self._vertex_display_name(src)
            dst_name = self._vertex_display_name(dst)
            top_edges_log.append(f"({src_name} -- {dst_name}: {flow:.6f})")
        logger.info(f"  Important edges (top 10): {', '.join(top_edges_log)}")

        return important_edges

    def _mini_ppr_select_seeds(
        self,
        discovered_vertex_ids: Dict[str, int],
        existing_seed_vertex_ids: List[int],
    ) -> List[Tuple[int, int, float]]:
        """Run mini-PPR from all bridge entities together on local subgraph.

        Uses subgraph (bridge + seeds + 3-hop neighbors) instead of full graph for speed.
        All bridge entities as seeds in one PPR run. Select seed connections by threshold.

        Returns list of (bridge_vid, seed_vid, ppr_score) for selected connections.
        """
        hip = self.hipporag
        graph = hip.graph
        passage_idx_set = set(hip.passage_node_idxs)

        # Build subgraph: bridge entities + seeds + 3-hop entity neighbors
        core_vids = set(discovered_vertex_ids.values()) | set(existing_seed_vertex_ids)
        subgraph_vids = set(core_vids)
        for hop in range(3):
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

        # All bridge entities together as seeds
        reset = np.zeros(n_sub)
        for d_vid in discovered_vertex_ids.values():
            if d_vid in vid_to_sub:
                reset[vid_to_sub[d_vid]] = 1.0
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

        # Collect seed scores above threshold
        selected = []
        bridge_vids = set(discovered_vertex_ids.values())
        seed_scores = []
        for s_vid in existing_seed_vertex_ids:
            if s_vid not in bridge_vids and s_vid in vid_to_sub:
                score = ppr_scores[vid_to_sub[s_vid]]
                seed_scores.append((s_vid, score))
        seed_scores.sort(key=lambda x: x[1], reverse=True)

        log_items = [(self._vertex_display_name(sv), f"{sc:.6f}") for sv, sc in seed_scores[:8]]
        logger.info(f"  Mini-PPR(subgraph, joint) -> seeds: {log_items}")

        # Connect all seeds with non-zero score to all bridge entities
        for s_vid, score in seed_scores:
            if score > 0:
                for d_vid in bridge_vids:
                    selected.append((d_vid, s_vid, score))

        return selected

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
        expansion_ppr_scores = np.zeros(num_docs)
        base_node_weights = None
        all_discovered = {}  # name -> vertex_id, accumulated across rounds
        doc_first_round = {}  # doc_id -> first round it appeared in top-K

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
                existing_seeds = self._get_existing_seed_ids(base_node_weights)

                # Mini-PPR filtered bridge edges
                ppr_connections = self._mini_ppr_select_seeds(
                    discovered_vertex_ids=all_discovered,
                    existing_seed_vertex_ids=existing_seeds,
                )

                modified_weights = base_node_weights.copy()
                for name, vid in all_discovered.items():
                    modified_weights[vid] += self._degree_adaptive_weight(vid, DEFAULT_ENTITY_SEED_WEIGHT)

                overlay = GraphOverlay(self.hipporag.graph)
                # Only add edges for PPR-selected bridge→seed pairs
                selected_pairs = set()
                for d_vid, s_vid, score in ppr_connections:
                    if score >= PPR_FILTER_THRESHOLD:
                        bridge_weight = self._degree_adaptive_weight(d_vid, 1.0)
                        overlay.add_reasoning_edge(d_vid, s_vid, bridge_weight)
                        selected_pairs.add((d_vid, s_vid))

                # Inter-discovered edges (always connect)
                d_list = list(all_discovered.values())
                for i, v1 in enumerate(d_list):
                    for v2 in d_list[i+1:]:
                        w = max(self._degree_adaptive_weight(v1, 1.0),
                                self._degree_adaptive_weight(v2, 1.0))
                        overlay.add_reasoning_edge(v1, v2, w)

                working_graph = overlay.get_working_graph() if overlay else None
                boosted_doc_ids, boosted_doc_scores = hip.run_ppr(
                    modified_weights,
                    damping=EXPANSION_DAMPING,
                    graph=working_graph,
                )

                # Expansion PPR: use raw PPR scores, overwrite (not accumulate)
                expansion_ppr_scores = np.zeros(num_docs)
                for doc_id, score in zip(boosted_doc_ids, boosted_doc_scores):
                    expansion_ppr_scores[doc_id] = score

                logger.info(
                    f"  PPR-filtered expansion: {len(all_discovered)} entities, "
                    f"selected edges: {len(selected_pairs)}/{len(ppr_connections)}, "
                    f"round_weight: {round_weight:.1f}"
                )

            combined_scores = expansion_ppr_scores + RRF_WEIGHT * rrf_scores
            final_sorted_ids = np.argsort(combined_scores)[::-1]
            top_k_docs = [
                hip.chunk_embedding_store.get_row(hip.passage_node_keys[idx])["content"]
                for idx in final_sorted_ids[:num_to_retrieve]
            ]

            # Track first appearance round for each doc in top-K
            for doc_id in final_sorted_ids[:num_to_retrieve]:
                did = int(doc_id)
                if did not in doc_first_round:
                    doc_first_round[did] = round_i

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

            # Entity seed expansion
            discovered_entities = reasoning_output.get("discovered_entities", [])
            if discovered_entities:
                logger.info(f"  Discovered entities: {discovered_entities}")
                resolved = self._resolve_entities(discovered_entities)
                all_discovered.update(resolved)
                state.discovered_entities.extend(
                    [(name, vid) for name, vid in resolved.items()]
                )
                logger.info(f"  Total resolved entities: {len(all_discovered)}")

        combined_scores = expansion_ppr_scores + RRF_WEIGHT * rrf_scores
        final_sorted_ids = np.argsort(combined_scores)[::-1]

        final_docs = [
            hip.chunk_embedding_store.get_row(hip.passage_node_keys[idx])["content"]
            for idx in final_sorted_ids[:num_to_retrieve]
        ]
        final_scores = [combined_scores[idx] for idx in final_sorted_ids[:num_to_retrieve]]

        # LLM reranker: reorder docs based on reasoning trajectory
        if state.reasoning_traces:
            reranked_docs = self._llm_reasoning_rerank(
                query=state.original_query,
                docs=final_docs,
                reasoning_traces=state.reasoning_traces,
            )
            if reranked_docs is not None:
                final_docs = reranked_docs

        return QuerySolution(
            question=state.original_query,
            docs=final_docs,
            doc_scores=final_scores,
            answer=None,
        )

    def _llm_reasoning_rerank(
        self,
        query: str,
        docs: List[str],
        reasoning_traces: List[str],
    ) -> Optional[List[str]]:
        """Use LLM to reorder documents based on reasoning trajectory.

        The LLM sees the reasoning steps and reorders docs so that
        the QA model reads them in the logical order of the reasoning chain.
        """
        import json as _json

        traces_text = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(reasoning_traces) if t)
        if not traces_text.strip():
            return None

        docs_text = ""
        for i, doc in enumerate(docs):
            docs_text += f"[Doc {i+1}] {doc[:300]}\n\n"

        prompt = f"""Given a multi-hop question and the reasoning steps used to find information, reorder the documents so they follow the logical reasoning chain. Documents supporting earlier reasoning steps should come first.

Question: {query}

Reasoning steps:
{traces_text}

Documents:
{docs_text}

Return a JSON list of document numbers in the recommended reading order. Only include the numbers, e.g. [3, 1, 5, 2, 4].
Important: include ALL document numbers exactly once."""

        try:
            messages = [
                {"role": "system", "content": "You are a document ordering assistant. Return only a JSON list of integers."},
                {"role": "user", "content": prompt},
            ]
            response, _, _ = self.hipporag.llm_model.infer(messages)

            # Parse response
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            order = _json.loads(text)
            if not isinstance(order, list) or len(order) != len(docs):
                logger.warning(f"  LLM rerank returned invalid order: {order}")
                return None

            # Convert 1-indexed to 0-indexed
            reranked = []
            for idx in order:
                i = int(idx) - 1
                if 0 <= i < len(docs):
                    reranked.append(docs[i])

            if len(reranked) != len(docs):
                logger.warning(f"  LLM rerank: missing docs after reorder")
                return None

            logger.info(f"  LLM reasoning rerank: {order}")
            return reranked

        except Exception as e:
            logger.warning(f"  LLM reasoning rerank failed: {e}")
            return None

    def _subgraph_rerank(
        self,
        doc_ids: np.ndarray,
        base_node_weights: Optional[np.ndarray],
        discovered_vertex_ids: Dict[str, int],
    ) -> Optional[np.ndarray]:
        """Re-rank top-N docs by running PPR on their local subgraph.

        1. Collect entity neighbors of top-N docs → small subgraph
        2. Run PPR from query seeds + bridge entities on subgraph
        3. Rank docs directly by subgraph PPR score
        """
        hip = self.hipporag
        graph = hip.graph
        passage_idx_set = set(hip.passage_node_idxs)

        # Step 1: Collect subgraph — doc passage nodes + their entity neighbors + 1-hop
        subgraph_vids = set()
        doc_vid_map = {}  # doc_id -> passage vertex id
        for doc_id in doc_ids:
            p_vid = hip.passage_node_idxs[doc_id]
            doc_vid_map[doc_id] = p_vid
            subgraph_vids.add(p_vid)
            for n in graph.neighbors(p_vid):
                subgraph_vids.add(n)
                if n not in passage_idx_set:
                    for nn in graph.neighbors(n):
                        subgraph_vids.add(nn)

        # Add bridge entity vertices
        for vid in discovered_vertex_ids.values():
            subgraph_vids.add(vid)

        # Add original seed vertices
        if base_node_weights is not None:
            for vid in range(len(base_node_weights)):
                if base_node_weights[vid] > 0 and vid not in passage_idx_set:
                    subgraph_vids.add(vid)

        subgraph_vids = sorted(subgraph_vids)
        if len(subgraph_vids) < 2:
            return None

        sub = graph.subgraph(subgraph_vids)
        vid_to_sub = {vid: i for i, vid in enumerate(subgraph_vids)}
        n_sub = len(subgraph_vids)

        # Step 2: Build reset vector — query seeds + bridge entities
        reset = np.zeros(n_sub)
        if base_node_weights is not None:
            for vid in range(len(base_node_weights)):
                if base_node_weights[vid] > 0 and vid in vid_to_sub:
                    reset[vid_to_sub[vid]] = base_node_weights[vid]
        for vid in discovered_vertex_ids.values():
            if vid in vid_to_sub:
                reset[vid_to_sub[vid]] += self._degree_adaptive_weight(vid, DEFAULT_ENTITY_SEED_WEIGHT)
        if reset.sum() == 0:
            return None
        reset /= reset.sum()

        # Step 3: PPR on subgraph
        ppr_scores = sub.personalized_pagerank(
            vertices=range(n_sub),
            damping=EXPANSION_DAMPING,
            directed=False,
            weights='weight' if 'weight' in sub.es.attributes() else None,
            reset=reset,
            implementation='prpack',
        )

        # Step 4: Rank docs directly by subgraph PPR score
        doc_scores = []
        for doc_id in doc_ids:
            p_vid = doc_vid_map[doc_id]
            sub_idx = vid_to_sub.get(p_vid)
            score = ppr_scores[sub_idx] if sub_idx is not None else 0.0
            doc_scores.append((doc_id, score))

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        new_order = [d[0] for d in doc_scores]

        # Log
        original_order = list(doc_ids)
        changes = sum(1 for i, (o, n) in enumerate(zip(original_order[:5], new_order[:5])) if o != n)
        logger.info(
            f"  Subgraph rerank: {n_sub} nodes, {sub.ecount()} edges, "
            f"top-5 changes: {changes}/5"
        )

        return np.array(new_order)

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
