import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GraphOverlay:
    """Per-query temporary graph modifications that persist across retrieval rounds.

    This overlay sits on top of the base igraph Graph without mutating it.
    Each query creates its own overlay; it is discarded after the query finishes.
    """

    def __init__(self, base_graph):
        self.base_graph = base_graph
        # Temporary edges: list of (src_vertex_id, dst_vertex_id, weight)
        self.temp_edges: List[Tuple[int, int, float]] = []
        # Multipliers on existing edge weights: (src_id, dst_id) -> multiplier
        self.edge_multipliers: Dict[Tuple[int, int], float] = {}

    def add_reasoning_edge(self, src_id: int, dst_id: int, weight: float):
        """Add a temporary edge inferred by reasoning."""
        if src_id == dst_id:
            return
        # Avoid duplicate temp edges between same pair
        for i, (s, d, w) in enumerate(self.temp_edges):
            if (s == src_id and d == dst_id) or (s == dst_id and d == src_id):
                # Update weight to the max
                self.temp_edges[i] = (s, d, max(w, weight))
                return
        self.temp_edges.append((src_id, dst_id, weight))
        logger.debug(f"GraphOverlay: added temp edge {src_id} <-> {dst_id}, weight={weight:.4f}")

    def boost_edge(self, src_id: int, dst_id: int, multiplier: float):
        """Boost an existing edge weight by a multiplier."""
        key = (src_id, dst_id)
        key_rev = (dst_id, src_id)
        # Store for both directions (undirected graph)
        self.edge_multipliers[key] = self.edge_multipliers.get(key, 1.0) * multiplier
        self.edge_multipliers[key_rev] = self.edge_multipliers.get(key_rev, 1.0) * multiplier

    def get_working_graph(self):
        """Return a copy of the base graph with overlay applied.

        This copies the graph so the base graph is never mutated.
        """
        g = self.base_graph.copy()

        # Apply edge multipliers to existing edges
        if self.edge_multipliers:
            name_to_idx = {v["name"]: v.index for v in g.vs}
            for (src_id, dst_id), mult in self.edge_multipliers.items():
                eid = g.get_eid(src_id, dst_id, error=False)
                if eid != -1:
                    g.es[eid]["weight"] = g.es[eid]["weight"] * mult

        # Add temporary edges
        if self.temp_edges:
            edges = [(e[0], e[1]) for e in self.temp_edges]
            weights = [e[2] for e in self.temp_edges]
            g.add_edges(edges, attributes={"weight": weights})

        return g

    @property
    def num_temp_edges(self) -> int:
        return len(self.temp_edges)

    @property
    def num_boosted_edges(self) -> int:
        # Each undirected boost creates 2 entries; count unique pairs
        return len(self.edge_multipliers) // 2

    def summary(self) -> str:
        return f"GraphOverlay(temp_edges={self.num_temp_edges}, boosted_edges={self.num_boosted_edges})"
