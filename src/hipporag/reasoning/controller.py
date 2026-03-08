import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .round_state import RetrievalRoundState, RoundMetrics
from .query_rewriter import QueryRewriter
from ..utils.misc_utils import QuerySolution
from ..evaluation.retrieval_eval import RetrievalRecall

logger = logging.getLogger(__name__)


class ReasoningController:
    """Orchestrates multi-round reasoning-guided retrieval.

    Each round:
      1. Run full HippoRAG retrieval with the current query
      2. Max-pool document scores across rounds
      3. LLM reasons about results -> rewrite query or stop
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

    def _retrieve_single_query(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run full HippoRAG retrieval for a single query.

        Returns:
            (sorted_doc_ids, sorted_doc_scores) — arrays covering all documents.
        """
        hip = self.hipporag

        hip.get_query_embeddings([query])

        query_fact_scores = hip.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, rerank_log = hip.rerank_facts(query, query_fact_scores)

        if len(top_k_facts) == 0:
            logger.info("  No facts found, falling back to DPR.")
            sorted_doc_ids, sorted_doc_scores = hip.dense_passage_retrieval(query)
        else:
            sorted_doc_ids, sorted_doc_scores = hip.graph_search_with_fact_entities(
                query=query,
                link_top_k=hip.global_config.linking_top_k,
                query_fact_scores=query_fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=hip.global_config.passage_node_weight,
            )

        return sorted_doc_ids, sorted_doc_scores

    def _iterative_retrieve_single(
        self,
        state: RetrievalRoundState,
        num_to_retrieve: int,
        gold_docs_for_query: Optional[List[str]] = None,
    ) -> QuerySolution:
        """Run iterative retrieval for a single query with RRF across rounds."""
        hip = self.hipporag
        num_docs = len(hip.passage_node_keys)
        rrf_k = 60  # RRF constant

        # Accumulated RRF scores across all rounds
        rrf_scores = np.zeros(num_docs)

        for round_i in range(self.max_rounds):
            state.round_idx = round_i
            round_start = time.time()

            logger.info(f"  Round {round_i}: query='{state.current_query[:60]}...'")

            # Step 1: Full retrieval with current query
            sorted_doc_ids, sorted_doc_scores = self._retrieve_single_query(state.current_query)

            # Step 2: RRF — accumulate 1/(k + rank) for each doc
            for rank, doc_id in enumerate(sorted_doc_ids):
                rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)

            # Build current result from RRF scores
            final_sorted_ids = np.argsort(rrf_scores)[::-1]
            top_k_docs = [
                hip.chunk_embedding_store.get_row(hip.passage_node_keys[idx])["content"]
                for idx in final_sorted_ids[:num_to_retrieve]
            ]

            # Step 3: Evaluate this round
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

            # Early stop: retrieval already perfect
            if round_metrics.get("Recall@5", 0) >= 1.0:
                logger.info(f"  Recall@5=1.0 at round {round_i}, early stop.")
                break

            # Last round: skip reasoning
            if round_i == self.max_rounds - 1:
                break

            # Step 4: LLM reasoning -> rewrite query or stop
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

            # Update query for next round
            new_query = reasoning_output.get("rewritten_query", "")
            if new_query and new_query != state.current_query:
                logger.info(f"  Query rewritten: '{new_query[:60]}...'")
                state.current_query = new_query

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

        # Per-round metrics (averaged across queries)
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

        # Final evaluation
        evaluator = RetrievalRecall(global_config=self.hipporag.global_config)
        k_list = [1, 2, 5, 10, 20, 50, 100]
        final_pooled, _ = evaluator.calculate_metric_scores(
            gold_docs=gold_docs,
            retrieved_docs=[r.docs for r in all_results],
            k_list=k_list,
        )
        eval_results["final"] = final_pooled

        # Average rounds used
        eval_results["avg_rounds_used"] = round(
            np.mean([len(s.round_metrics) for s in all_round_states]), 2
        )

        return eval_results
