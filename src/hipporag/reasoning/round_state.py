import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Metrics for a single retrieval round."""
    round_idx: int = 0
    recall_at_k: Dict[str, float] = field(default_factory=dict)
    num_retrieved: int = 0
    query_used: str = ""
    # PPR diffusion diagnostics
    ppr_entropy: float = 0.0
    ppr_concentration: float = 0.0
    ppr_query_sim: float = 0.0  # avg similarity between query and top activated entities
    ppr_coverage_gap: float = 0.0


@dataclass
class RetrievalRoundState:
    """Mutable state that persists across retrieval rounds for a single query.

    Created once per query, carried through all rounds, discarded after the query.
    """
    original_query: str = ""
    current_query: str = ""
    round_idx: int = 0

    # History
    retrieved_docs_per_round: List[List[str]] = field(default_factory=list)
    round_metrics: List[RoundMetrics] = field(default_factory=list)
    reasoning_traces: List[str] = field(default_factory=list)
    discovered_entities: list = field(default_factory=list)  # [(name, vertex_id), ...]

    def record_round(self, docs: List[str], metrics: Dict[str, float], query_used: str):
        self.retrieved_docs_per_round.append(docs)
        rm = RoundMetrics(
            round_idx=self.round_idx,
            recall_at_k=metrics,
            num_retrieved=len(docs),
            query_used=query_used,
        )
        self.round_metrics.append(rm)

    def get_all_retrieved_docs(self) -> List[str]:
        """Union of all docs retrieved across rounds (deduped, order preserved)."""
        seen = set()
        result = []
        for docs in self.retrieved_docs_per_round:
            for d in docs:
                if d not in seen:
                    seen.add(d)
                    result.append(d)
        return result

    def summary(self) -> str:
        lines = [f"Round {self.round_idx}, query='{self.current_query[:80]}...'"]
        for rm in self.round_metrics:
            lines.append(f"  round {rm.round_idx}: {rm.recall_at_k}")
        return "\n".join(lines)
