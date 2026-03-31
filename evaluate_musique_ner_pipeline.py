"""
Evaluate NER-based pipeline vs HippoRAG OpenIE pipeline on MuSiQue.

Key differences from HippoRAG:
  1. LLM NER (entity extraction only, no triples)
  2. Sentence-level matching for seed selection (not triple fact matching)
  3. Entity-passage bipartite graph with sentence co-occurrence weights

Usage:
    .venv/bin/python evaluate_musique_ner_pipeline.py --data_path musique.json --sample_limit 30
"""
import json
import os
import sys
import random
import argparse
import logging
import time
import gc
import re as _re
from datetime import datetime
from itertools import combinations
from collections import defaultdict
from hashlib import md5
from typing import List, Dict, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# ── Utilities ──────────────────────────────────────────────────────

def compute_hash(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def _file_content_fingerprint(path: Optional[str]) -> str:
    if not path:
        return "none"
    if not os.path.exists(path):
        return "missing"
    with open(path, "rb") as f:
        return md5(f.read()).hexdigest()


def _index_cache_dir(base_dir: str, embedding_model_name: str, ner_cache_path: Optional[str], docs: List[str]) -> str:
    docs_fingerprint = md5(json.dumps(list(docs), ensure_ascii=False).encode()).hexdigest()
    ner_info = _file_content_fingerprint(ner_cache_path)
    cache_id = md5(f"{embedding_model_name}|docs={len(docs)}|{docs_fingerprint}|{ner_info}".encode()).hexdigest()
    return os.path.join(base_dir, "ner_index_cache", cache_id)


def _make_results_output_path(save_dir: str, mode: str, max_rounds: int, run_id: Optional[str] = None) -> str:
    if mode == "ircot":
        output_tag = f"_ircot_rounds{max_rounds}"
    elif max_rounds > 1:
        output_tag = f"_rounds{max_rounds}"
    else:
        output_tag = ""
    if run_id is None:
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    return os.path.join(save_dir, f"comparison_results{output_tag}_{run_id}.json")


def normalize_answer(s):
    s = s.lower().strip()
    s = _re.sub(r'[^\w\s]', '', s)
    s = _re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def em_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)


def compute_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    from collections import Counter
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def check_em(pred, gold_answer, gold_aliases):
    all_golds = [gold_answer] + (gold_aliases if gold_aliases else [])
    return any(em_match(pred, g) for g in all_golds)


def check_f1(pred, gold_answer, gold_aliases):
    all_golds = [gold_answer] + (gold_aliases if gold_aliases else [])
    return max(compute_f1(pred, g) for g in all_golds)


def min_max_normalize(x):
    mn, mx = np.min(x), np.max(x)
    r = mx - mn
    if r == 0:
        return np.ones_like(x)
    return (x - mn) / r


def l2_normalize(x):
    """L2 normalize embeddings (row-wise)."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return x / norms


# Query instruction prefixes
QUERY_INSTRUCTION_SENTENCE = 'Given a question, retrieve relevant sentences that best answer the question.'
QUERY_INSTRUCTION_PASSAGE = 'Given a question, retrieve relevant documents that best answer the question.'


def sentence_split(text: str) -> List[str]:
    """Simple sentence splitting using regex. Handles Mr./Mrs./Dr. etc."""
    # Split on period/question/exclamation followed by space+uppercase or end
    sentences = _re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Filter empty
    return [s.strip() for s in sentences if s.strip()]


# ── LLM NER ──────────────────────────────────────────────────────

NER_SYSTEM = """Your task is to extract named entities from the given paragraph.
Respond with a JSON list of entities."""

NER_ONE_SHOT_INPUT = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

NER_ONE_SHOT_OUTPUT = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}"""

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


def llm_ner(text: str, llm_client) -> List[str]:
    """Extract named entities from text using LLM."""
    messages = [
        {"role": "system", "content": NER_SYSTEM},
        {"role": "user", "content": NER_ONE_SHOT_INPUT},
        {"role": "assistant", "content": NER_ONE_SHOT_OUTPUT},
        {"role": "user", "content": text},
    ]
    try:
        result = llm_client.infer(messages=messages)
        response = result[0] if isinstance(result, tuple) else result
        if not isinstance(response, str):
            response = response[0]["content"]
        # Parse JSON from response
        # Try to find JSON object in response
        match = _re.search(r'\{.*\}', response, _re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("named_entities", [])
        return []
    except Exception as e:
        logger.warning(f"NER failed: {e}")
        return []


# ── Indexing ──────────────────────────────────────────────────────

class NERIndex:
    """NER-based index: sentences, entities, entity-passage graph."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        # Sentence store: {sentence_id: {text, passage_id, entities}}
        self.sentences = {}
        # Entity to sentence mapping
        self.entity_to_sentences = defaultdict(list)
        # Entity to passage mapping
        self.entity_to_passages = defaultdict(set)
        # All unique entities
        self.entities = []
        self.entity_keys = []  # hash keys
        # Passage info
        self.passages = {}  # {passage_id: text}
        self.passage_keys = []
        # Embeddings
        self.sentence_embeddings = None  # (n_sentences, dim)
        self.passage_embeddings = None  # (n_passages, dim)
        self.entity_embeddings = None  # (n_entities, dim)
        self.pair_embeddings = None  # (n_pairs, dim)
        # Entity pairs: list of (ent_text1, ent_key1, ent_text2, ent_key2, sent_id)
        self.pairs = []
        # Sentence ID list (ordered)
        self.sentence_ids = []
        # Graph
        self.graph = None
        self.node_name_to_idx = {}
        self.entity_node_idxs = []
        self.passage_node_idxs = []

    def save(self, path: str):
        """Save index to pickle for reuse."""
        import pickle
        # Don't save embedding_model (not picklable)
        emb = self.embedding_model
        self.embedding_model = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.embedding_model = emb
        logger.info(f"Saved NERIndex to {path}")

    @classmethod
    def load(cls, path: str, embedding_model):
        """Load index from pickle."""
        import pickle
        with open(path, 'rb') as f:
            index = pickle.load(f)
        index.embedding_model = embedding_model
        logger.info(f"Loaded NERIndex from {path}: {index.graph.vcount()} nodes, {index.graph.ecount()} edges")
        return index

    def build(self, docs: List[str], ner_results: Dict[str, List[str]]):
        """Build index from docs and pre-computed NER results.

        Args:
            docs: list of passage texts
            ner_results: {passage_text: [entity1, entity2, ...]}
        """
        logger.info(f"Building NER index for {len(docs)} passages...")

        # Step 1: Sentence splitting + entity matching
        sent_counter = 0
        for doc_idx, doc_text in enumerate(docs):
            passage_id = compute_hash(doc_text, prefix="passage-")
            self.passages[passage_id] = doc_text
            self.passage_keys.append(passage_id)

            entities = ner_results.get(doc_text, [])
            sents = sentence_split(doc_text)

            for sent_text in sents:
                sent_id = f"sent-{sent_counter}"
                sent_entities = []
                for ent in entities:
                    if ent.lower() in sent_text.lower():
                        ent_key = compute_hash(ent.lower(), prefix="entity-")
                        sent_entities.append((ent.lower(), ent_key))
                        self.entity_to_sentences[ent_key].append(sent_id)
                        self.entity_to_passages[ent_key].add(passage_id)

                self.sentences[sent_id] = {
                    "text": sent_text,
                    "passage_id": passage_id,
                    "entities": sent_entities,
                }
                self.sentence_ids.append(sent_id)
                sent_counter += 1

        # Collect unique entities
        seen = set()
        for ent_key, passage_set in self.entity_to_passages.items():
            if ent_key not in seen:
                seen.add(ent_key)
                self.entity_keys.append(ent_key)
        # Get entity text from any sentence
        ent_key_to_text = {}
        for sent_id, sent_data in self.sentences.items():
            for ent_text, ent_key in sent_data["entities"]:
                if ent_key not in ent_key_to_text:
                    ent_key_to_text[ent_key] = ent_text
        self.entities = [ent_key_to_text.get(k, k) for k in self.entity_keys]

        logger.info(f"  {len(self.sentences)} sentences, {len(self.entity_keys)} unique entities, "
                     f"{len(self.passage_keys)} passages")

        # Step 2: Generate entity pairs from sentences
        for sent_id in self.sentence_ids:
            sent_data = self.sentences[sent_id]
            ents = sent_data["entities"]
            if len(ents) >= 2:
                for (t1, k1), (t2, k2) in combinations(ents, 2):
                    self.pairs.append((t1, k1, t2, k2, sent_id))

        logger.info(f"  {len(self.pairs)} entity pairs from sentences")

        # Step 3: Compute embeddings
        logger.info("  Computing sentence embeddings...")
        sent_texts = [self.sentences[sid]["text"] for sid in self.sentence_ids]
        if sent_texts:
            self.sentence_embeddings = self.embedding_model.batch_encode(sent_texts)
        else:
            self.sentence_embeddings = np.array([])

        logger.info("  Computing passage embeddings...")
        passage_texts = [self.passages[pid] for pid in self.passage_keys]
        if passage_texts:
            self.passage_embeddings = self.embedding_model.batch_encode(passage_texts)
        else:
            self.passage_embeddings = np.array([])

        logger.info("  Computing entity embeddings...")
        if self.entities:
            self.entity_embeddings = self.embedding_model.batch_encode(self.entities)
        else:
            self.entity_embeddings = np.array([])

        logger.info("  Computing pair embeddings...")
        if self.pairs:
            pair_texts = [f"{p[0]} | {p[2]} | {self.sentences[p[4]]['text']}" for p in self.pairs]
            self.pair_embeddings = self.embedding_model.batch_encode(pair_texts)
        else:
            self.pair_embeddings = np.array([])

        # Step 3: Build graph
        self._build_graph()

        logger.info(f"  Graph: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")

    def _build_graph(self):
        """Build graph with 3 edge types: entity-entity co-occurrence, entity-passage, synonymy."""
        import igraph as ig

        # Nodes: entities + passages
        node_names = self.entity_keys + self.passage_keys
        node_contents = self.entities + [self.passages[pid] for pid in self.passage_keys]
        node_types = ["entity"] * len(self.entity_keys) + ["passage"] * len(self.passage_keys)

        self.graph = ig.Graph(directed=False)
        self.graph.add_vertices(len(node_names))
        self.graph.vs["name"] = node_names
        self.graph.vs["content"] = node_contents
        self.graph.vs["type"] = node_types

        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}
        self.entity_node_idxs = list(range(len(self.entity_keys)))
        self.passage_node_idxs = list(range(len(self.entity_keys),
                                            len(self.entity_keys) + len(self.passage_keys)))

        edges = []
        weights = []

        # 1) Entity ↔ Entity co-occurrence edges (from pairs, weight = co-occurrence count)
        cooccur_count = defaultdict(int)
        for t1, k1, t2, k2, sent_id in self.pairs:
            pair = (k1, k2) if k1 < k2 else (k2, k1)
            cooccur_count[pair] += 1
        for (k1, k2), count in cooccur_count.items():
            i1 = self.node_name_to_idx.get(k1)
            i2 = self.node_name_to_idx.get(k2)
            if i1 is not None and i2 is not None:
                edges.append((i1, i2))
                weights.append(float(count))
        n_cooccur = len(edges)
        logger.info(f"  {n_cooccur} entity-entity co-occurrence edges")

        # 2) Entity ↔ Passage edges (weight = 1.0, like HippoRAG)
        for ent_key in self.entity_keys:
            ent_idx = self.node_name_to_idx[ent_key]
            for passage_id in self.entity_to_passages[ent_key]:
                p_idx = self.node_name_to_idx.get(passage_id)
                if p_idx is not None:
                    edges.append((ent_idx, p_idx))
                    weights.append(1.0)
        n_ep = len(edges) - n_cooccur
        logger.info(f"  {n_ep} entity-passage edges")

        # 3) Synonymy edges (entity-entity, batched KNN like HippoRAG)
        syn_threshold = float(os.environ.get("SYNONYMY_THRESHOLD", "0.8"))
        syn_topk = min(2047, len(self.entity_keys))
        if len(self.entity_keys) > 1 and self.entity_embeddings is not None and len(self.entity_embeddings) > 0:
            logger.info(f"  Computing synonymy edges (threshold={syn_threshold}, topk={syn_topk})...")
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            emb_tensor = torch.tensor(self.entity_embeddings, dtype=torch.float32)
            emb_tensor = torch.nn.functional.normalize(emb_tensor, dim=1)

            n_before = len(edges)
            batch_size = 1000
            k = min(syn_topk + 1, len(self.entity_keys))  # +1 to skip self

            for start in range(0, len(self.entity_keys), batch_size):
                end = min(start + batch_size, len(self.entity_keys))
                query_batch = emb_tensor[start:end].to(device)
                sims = query_batch @ emb_tensor.T.to(device)  # (batch, n_entities)
                topk_scores, topk_idxs = torch.topk(sims, k=k, dim=1)

                for bi in range(end - start):
                    i = start + bi
                    for ki in range(k):
                        j = int(topk_idxs[bi, ki])
                        score = float(topk_scores[bi, ki])
                        if j > i and score >= syn_threshold:  # j > i to avoid duplicates
                            edges.append((i, j))
                            weights.append(score)

            logger.info(f"  {len(edges) - n_before} synonymy edges")

        if edges:
            self.graph.add_edges(edges)
            self.graph.es["weight"] = weights


    def _compute_node_weights(self, query: str, passage_weight_scale: float = 0.05,
                              link_top_k: int = 5, pair_alpha: float = 0.5,
                              sent_alpha: float = 0.5, entity_top_k: int = 5,
                              mmr_lambda: float = 0.7,
                              query_entities: Optional[List[str]] = None) -> np.ndarray:
        use_instruction = getattr(self.embedding_model, 'supports_instruction', False)
        sent_instr = QUERY_INSTRUCTION_SENTENCE if use_instruction else ''
        pass_instr = QUERY_INSTRUCTION_PASSAGE if use_instruction else ''
        query_emb_sentence = l2_normalize(self.embedding_model.batch_encode(
            [query], instruction=sent_instr))
        query_emb_passage = l2_normalize(self.embedding_model.batch_encode(
            [query], instruction=pass_instr))

        node_weights = np.zeros(self.graph.vcount())
        sent_id_to_idx = {sid: i for i, sid in enumerate(self.sentence_ids)}
        sent_scores_all = None
        if self.sentence_embeddings is not None and len(self.sentence_embeddings) > 0:
            sent_scores_all = (self.sentence_embeddings @ query_emb_sentence.T).flatten()
            sent_scores_all = min_max_normalize(sent_scores_all)
        passage_scores = None
        if self.passage_embeddings is not None and len(self.passage_embeddings) > 0:
            passage_scores = (self.passage_embeddings @ query_emb_passage.T).flatten()
            passage_scores = min_max_normalize(passage_scores)

        if self.pairs and self.pair_embeddings is not None and len(self.pair_embeddings) > 0:
            top_pair_idxs = []
            combined_scores_lookup = {}

            # Full matrix multiply — same approach as HippoRAG fact matching
            pair_scores = (self.pair_embeddings @ query_emb_sentence.T).flatten()
            pair_scores = min_max_normalize(pair_scores)

            combined_scores = np.zeros(len(self.pairs))
            for pi, (_, _, _, _, sent_id) in enumerate(self.pairs):
                s_score = 0.0
                if sent_scores_all is not None:
                    si = sent_id_to_idx.get(sent_id, -1)
                    if si >= 0:
                        s_score = sent_scores_all[si]
                combined_scores[pi] = pair_alpha * pair_scores[pi] + sent_alpha * s_score

            pair_norms = np.linalg.norm(self.pair_embeddings, axis=1, keepdims=True)
            pair_norms = np.where(pair_norms == 0, 1, pair_norms)
            normed_pairs = self.pair_embeddings / pair_norms

            candidate_size = min(30, len(self.pairs))
            candidate_idxs = np.argsort(combined_scores)[::-1][:candidate_size].tolist()

            selected_embs = []
            for _ in range(link_top_k):
                if not candidate_idxs:
                    break
                if not selected_embs:
                    best_pi = candidate_idxs[0]
                else:
                    selected_matrix = np.array(selected_embs)
                    best_mmr = -float("inf")
                    best_pi = candidate_idxs[0]
                    for pi in candidate_idxs:
                        relevance = combined_scores[pi]
                        sim = np.max(normed_pairs[pi] @ selected_matrix.T)
                        mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * sim
                        if mmr_score > best_mmr:
                            best_mmr = mmr_score
                            best_pi = pi
                top_pair_idxs.append(best_pi)
                selected_embs.append(normed_pairs[best_pi])
                candidate_idxs.remove(best_pi)
            combined_scores_lookup = {pi: combined_scores[pi] for pi in top_pair_idxs}

            ent_occur_count = np.zeros(self.graph.vcount())
            ent_max_score = np.zeros(self.graph.vcount())
            for pi in top_pair_idxs:
                t1, k1, t2, k2, sent_id = self.pairs[pi]
                score = combined_scores_lookup.get(pi, 0.0)
                for ent_key in [k1, k2]:
                    ent_idx = self.node_name_to_idx.get(ent_key)
                    if ent_idx is not None:
                        n_passages = max(len(self.entity_to_passages[ent_key]), 1)
                        weighted_score = score / n_passages
                        node_weights[ent_idx] += weighted_score
                        ent_occur_count[ent_idx] += 1
                        if weighted_score > ent_max_score[ent_idx]:
                            ent_max_score[ent_idx] = weighted_score
            for idx in self.entity_node_idxs:
                if ent_occur_count[idx] > 0:
                    avg = node_weights[idx] / ent_occur_count[idx]
                    node_weights[idx] = (ent_max_score[idx] + avg) / 2.0

        entity_weights = [(idx, node_weights[idx]) for idx in self.entity_node_idxs if node_weights[idx] > 0]
        if len(entity_weights) > entity_top_k:
            entity_weights.sort(key=lambda x: x[1], reverse=True)
            keep_idxs = set(idx for idx, _ in entity_weights[:entity_top_k])
            for idx in self.entity_node_idxs:
                if idx not in keep_idxs:
                    node_weights[idx] = 0.0

        if passage_scores is not None:
            passage_scores = passage_scores * passage_weight_scale
            for i, p_idx in enumerate(self.passage_node_idxs):
                node_weights[p_idx] = passage_scores[i]

        # Round 0 query entity seeds: resolve top-2 query entities directly in graph
        if query_entities and self.entity_embeddings is not None and len(self.entity_embeddings) > 0:
            resolved = _resolve_entities_in_graph(self, query_entities)
            for name, (vid, sim) in resolved.items():
                ent_key = self.entity_keys[vid] if vid < len(self.entity_keys) else None
                n_passages = max(len(self.entity_to_passages.get(ent_key, [])), 1) if ent_key else 1
                w = DEFAULT_ENTITY_SEED_WEIGHT * sim / n_passages
                node_weights[vid] = max(node_weights[vid], w)
            logger.info(f"  Query entity seeds: {list(resolved.keys())} → {len(resolved)} nodes")

        return node_weights

    def retrieve(self, query: str, top_k: int = 5, passage_weight_scale: float = 0.05,
                 link_top_k: int = 5, pair_alpha: float = 0.5,
                 sent_alpha: float = 0.5,
                 entity_top_k: int = 5,
                 mmr_lambda: float = 0.7,
                 working_graph=None,
                 extra_node_weights: Optional[np.ndarray] = None,
                 return_node_weights: bool = False,
                 query_entities: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve passages using pair matching + sentence matching → PPR."""
        base_node_weights = self._compute_node_weights(
            query=query,
            passage_weight_scale=passage_weight_scale,
            link_top_k=link_top_k,
            pair_alpha=pair_alpha,
            sent_alpha=sent_alpha,
            entity_top_k=entity_top_k,
            mmr_lambda=mmr_lambda,
            query_entities=query_entities,
        )
        node_weights = base_node_weights.copy()
        if extra_node_weights is not None:
            node_weights = node_weights + extra_node_weights

        if node_weights.sum() == 0:
            logger.warning("No seed weights, falling back to passage embedding")
            use_instruction = getattr(self.embedding_model, 'supports_instruction', False)
            pass_instr = QUERY_INSTRUCTION_PASSAGE if use_instruction else ''
            query_emb_passage = l2_normalize(self.embedding_model.batch_encode(
                [query], instruction=pass_instr))
            passage_scores = (self.passage_embeddings @ query_emb_passage.T).flatten()
            sorted_ids = np.argsort(passage_scores)[::-1]
            if return_node_weights:
                return sorted_ids, passage_scores[sorted_ids], base_node_weights
            return sorted_ids, passage_scores[sorted_ids]

        sorted_doc_ids, sorted_doc_scores = _run_ppr(
            self,
            reset_prob=node_weights,
            damping=0.5,
            graph=working_graph,
        )
        if return_node_weights:
            return sorted_doc_ids, sorted_doc_scores, base_node_weights
        return sorted_doc_ids, sorted_doc_scores


# ── QA (aligned with HippoRAG's rag_qa_musique prompt) ──────

QA_SYSTEM = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
)

QA_ONE_SHOT_DOCS = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
)

QA_ONE_SHOT_INPUT = (
    f"{QA_ONE_SHOT_DOCS}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    '\nThought: '
)

QA_ONE_SHOT_OUTPUT = (
    "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
    "\nAnswer: 1862."
)


def llm_qa(query: str, docs: List[str], llm_client) -> str:
    """QA with HippoRAG-aligned prompt: CoT Thought + Answer extraction."""
    # Build prompt_user in same format as HippoRAG
    prompt_user = ''
    for passage in docs[:5]:
        prompt_user += f'Wikipedia Title: {passage}\n\n'
    prompt_user += 'Question: ' + query + '\nThought: '

    messages = [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": QA_ONE_SHOT_INPUT},
        {"role": "assistant", "content": QA_ONE_SHOT_OUTPUT},
        {"role": "user", "content": prompt_user},
    ]
    try:
        result = llm_client.infer(messages=messages)
        response = result[0] if isinstance(result, tuple) else result
        if not isinstance(response, str):
            response = response[0]["content"]
        # Extract answer after "Answer:" like HippoRAG
        try:
            pred_ans = response.split('Answer:')[1].strip()
        except (IndexError, AttributeError):
            pred_ans = response.strip()
        return pred_ans
    except Exception as e:
        logger.error(f"QA failed: {e}")
        return "Error"


# ── Evaluation ──────────────────────────────────────────────────────

def recall_at_k(retrieved_docs: List[str], gold_docs: List[str], k: int) -> float:
    """Compute Recall@k."""
    if not gold_docs:
        return 0.0
    retrieved_set = set(retrieved_docs[:k])
    gold_set = set(gold_docs)
    return len(retrieved_set & gold_set) / len(gold_set)


# ── Reasoning (ported from feature/graph-reshape reasoning/) ──────

import math

# Constants (from feature/graph-reshape controller.py)
DEFAULT_ENTITY_SEED_WEIGHT = 0.5
RRF_ROUND_BOOST = 0.5
EXPANSION_DAMPING = 0.7
MINI_PPR_THRESHOLD = 0.0001
PIPELINE_RRF_WEIGHT = 1.0
EXPANSION_RRF_WEIGHT = 1.0

# Prompt (from feature/graph-reshape query_rewriter.py — includes discovered_entities)
REWRITE_SYSTEM_PROMPT = """You are a retrieval reasoning assistant. Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

1. Analyze what information has been found and what is still missing.
2. Identify key bridge entities in the retrieved documents that connect to the missing information.
3. Rewrite the query to better target the missing information.
4. Decide whether to continue retrieval or stop.

Respond in JSON format:
{
    "analysis": "Brief analysis of what's found vs missing",
    "discovered_entities": ["entity1", "entity2"],
    "rewritten_query": "The rewritten query targeting missing info",
    "should_stop": false
}

Rules:
- If the retrieved documents already contain sufficient information to answer the query, set should_stop=true and leave rewritten_query empty.
- "discovered_entities" should list key entities found in the retrieved documents that are important for answering the query but were NOT in the original query. These are bridge entities that connect what's been found to what's still needed. Use lowercase. List 1-5 entities max.
- Rewritten query should be a natural language question, not keywords. It should incorporate discovered bridge entities.
- Focus on what information is missing and craft the query to find it.
"""


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

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    defaults = {"analysis": "", "discovered_entities": [], "rewritten_query": "", "should_stop": False}
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
        # Ensure discovered_entities is a list of lowercase strings
        if not isinstance(parsed.get("discovered_entities", []), list):
            parsed["discovered_entities"] = []
        parsed["discovered_entities"] = [
            str(e).lower().strip() for e in parsed["discovered_entities"] if e
        ][:5]
        return parsed
    except Exception as e:
        logger.warning(f"Reasoning failed: {e}")
        defaults["analysis"] = str(e)[:200]
        defaults["should_stop"] = True
        return defaults


# ── Entity resolution (from feature/graph-reshape controller.py) ──

def _resolve_entities_in_graph(index, entity_names: List[str], threshold: float = 0.6) -> Dict[str, Tuple[int, float]]:
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


def _degree_adaptive_weight(index, vid: int, base_weight: float, sim: float = 1.0) -> float:
    """Scale weight by semantic similarity and log(degree) to resist dilution at high-degree nodes."""
    deg = index.graph.degree(vid)
    return base_weight * sim * (1.0 + math.log(deg + 1))


def _get_existing_seed_ids(index, base_node_weights: np.ndarray) -> List[int]:
    """Extract entity vertex IDs that have non-zero weight in base PPR seeds."""
    passage_idx_set = set(index.passage_node_idxs)
    seed_ids = []
    for vid in range(len(base_node_weights)):
        if base_node_weights[vid] > 0 and vid not in passage_idx_set:
            seed_ids.append(vid)
    return seed_ids


def _mini_ppr_select_seeds(
    index,
    discovered_vertex_ids: Dict[str, int],
    existing_seed_vertex_ids: List[int],
    round0_top_doc_ids: List[int] = None,
    graph=None,
) -> List[Tuple[int, int, float]]:
    """Run mini-PPR from bridge entities on local 5-hop subgraph.

    Returns list of (bridge_vid, seed_vid, ppr_score) for selected connections.
    """
    graph = graph if graph is not None else index.graph
    passage_idx_set = set(index.passage_node_idxs)

    # Build subgraph from round 0 top docs' entities + bridge + seeds
    core_vids = set(discovered_vertex_ids.values()) | set(existing_seed_vertex_ids)
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

    log_items = [(index.graph.vs[sv]["content"][:30], f"{sc:.6f}") for sv, sc in seed_scores[:8]]
    logger.info(f"  Mini-PPR(subgraph, joint) -> seeds: {log_items}")

    for s_vid, score in seed_scores:
        if score >= MINI_PPR_THRESHOLD:
            for d_vid in bridge_vids:
                selected.append((d_vid, s_vid, score))

    return selected


def _collect_hop_subgraph_vids(index, core_vids: List[int], hops: int = 5) -> List[int]:
    """Collect k-hop subgraph vertices from core nodes on the full graph."""
    graph = index.graph
    subgraph_vids = set(core_vids)
    for _ in range(hops):
        frontier = set()
        for vid in subgraph_vids:
            for n in graph.neighbors(vid):
                frontier.add(n)
        subgraph_vids |= frontier
    return sorted(subgraph_vids)


def _build_masked_graph(index, subgraph_vids: List[int]):
    """Return a graph copy where edges outside the subgraph are removed while vertex ids stay stable."""
    g = index.graph.copy()
    keep = set(subgraph_vids)
    remove_eids = []
    for eid, (u, v) in enumerate(g.get_edgelist()):
        if u not in keep or v not in keep:
            remove_eids.append(eid)
    if remove_eids:
        g.delete_edges(remove_eids)
    return g


def _run_ppr(index, reset_prob: np.ndarray, damping: float = 0.5, graph=None) -> Tuple[np.ndarray, np.ndarray]:
    working_graph = graph if graph is not None else index.graph
    reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
    ppr_scores = working_graph.personalized_pagerank(
        vertices=range(working_graph.vcount()),
        damping=damping,
        directed=False,
        weights='weight' if 'weight' in working_graph.es.attributes() else None,
        reset=reset_prob,
        implementation='prpack',
    )
    doc_scores = np.array([ppr_scores[idx] for idx in index.passage_node_idxs])
    sorted_doc_ids = np.argsort(doc_scores)[::-1]
    sorted_doc_scores = doc_scores[sorted_doc_ids]
    return sorted_doc_ids, sorted_doc_scores


def _build_overlay_graph(index, temp_edges: List[Tuple[int, int, float]], base_graph=None):
    """Create a copy of the graph with temporary bridge edges added."""
    graph = base_graph if base_graph is not None else index.graph
    g = graph.copy()
    if temp_edges:
        edges = [(e[0], e[1]) for e in temp_edges]
        weights = [e[2] for e in temp_edges]
        g.add_edges(edges, attributes={"weight": weights})
    return g


def _llm_reasoning_rerank(query: str, docs: List[str], reasoning_traces: List[str],
                          llm_client) -> Optional[List[str]]:
    """Use LLM to reorder documents based on reasoning trajectory."""
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
        response = llm_client.infer(messages=messages)
        if isinstance(response, tuple):
            response = response[0]
        if not isinstance(response, str):
            response = response[0]["content"]

        text = response.strip()
        if "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()

        order = json.loads(text)
        if not isinstance(order, list) or len(order) != len(docs):
            logger.warning(f"  LLM rerank returned invalid order: {order}")
            return None

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


def iterative_retrieve(index, question: str, llm_client, max_rounds: int = 3,
                       gold_docs: List[str] = None, query_entities: Optional[List[str]] = None,
                       entity_top_k: int = 5):
    """Multi-round retrieval with one retrieval pass per round.

    Core semantics (old version minus hop-5 and cross-round accumulation):
      1. Each round: single PPR on full graph + this-round temp edges
      2. Temp edges and seed weights are freshly computed each round from all_discovered
      3. No hop-5 subgraph, no cross-round seed/edge accumulation
      4. RRF accumulates doc scores across rounds
    """
    num_docs = len(index.passage_keys)
    rrf_k = 60
    rrf_scores = np.zeros(num_docs)
    current_query = question
    reasoning_traces = []
    round_diagnostics = []
    all_discovered = {}
    base_node_weights = None
    round0_top_doc_ids = None

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
            # all_discovered: {name: (vid, sim)}
            discovered_vid_only = {name: vid for name, (vid, sim) in all_discovered.items()}
            existing_seeds = _get_existing_seed_ids(index, base_node_weights)
            ppr_connections = _mini_ppr_select_seeds(
                index,
                discovered_vertex_ids=discovered_vid_only,
                existing_seed_vertex_ids=existing_seeds,
                round0_top_doc_ids=round0_top_doc_ids,
            )

            # Build vid -> sim lookup
            vid_to_sim = {vid: sim for name, (vid, sim) in all_discovered.items()}

            for d_vid, s_vid, _ in ppr_connections:
                d_sim = vid_to_sim.get(d_vid, 1.0)
                bridge_weight = _degree_adaptive_weight(index, d_vid, 1.0, sim=d_sim)
                temp_edges.append((d_vid, s_vid, bridge_weight))
            d_list = [(vid, sim) for name, (vid, sim) in all_discovered.items()]
            for i, (v1, s1) in enumerate(d_list):
                for v2, s2 in d_list[i + 1:]:
                    w = max(_degree_adaptive_weight(index, v1, 1.0, sim=s1),
                            _degree_adaptive_weight(index, v2, 1.0, sim=s2))
                    temp_edges.append((v1, v2, w))

            # Fresh seed weights for this round
            extra_node_weights = np.zeros(index.graph.vcount())
            for name, (vid, sim) in all_discovered.items():
                extra_node_weights[vid] += _degree_adaptive_weight(index, vid, DEFAULT_ENTITY_SEED_WEIGHT, sim=sim)

            logger.info(f"  Overlay: {len(all_discovered)} bridge entities, {len(temp_edges)} temp edges")

        # Single PPR on full graph + temp edges
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

        round_weight = 1.0 + round_i * RRF_ROUND_BOOST
        for rank, doc_id in enumerate(sorted_doc_ids):
            rrf_scores[doc_id] += PIPELINE_RRF_WEIGHT * round_weight / (rrf_k + rank + 1)

        final_sorted_ids = np.argsort(rrf_scores)[::-1]
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

        reasoning_traces.append(reasoning_output.get("analysis", ""))
        round_diag["rewritten_query"] = reasoning_output.get("rewritten_query", "")
        round_diag["new_discovered_entities"] = reasoning_output.get("discovered_entities", [])
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
            all_discovered.update(resolved)
            logger.info(f"  Resolved {len(resolved)}/{len(discovered_entities)} bridge entities in graph")
            round_diag["discovered_entities_total"] = sorted(all_discovered.keys())

        round_diagnostics.append(round_diag)

    final_sorted_ids = np.argsort(rrf_scores)[::-1]
    retrieved_docs = [index.passages[index.passage_keys[idx]] for idx in final_sorted_ids]

    rerank_limit = min(5, len(retrieved_docs))
    if rerank_limit > 1:
        reranked = _llm_reasoning_rerank(question, retrieved_docs[:rerank_limit], reasoning_traces, llm_client)
        if reranked is not None:
            retrieved_docs = reranked + retrieved_docs[rerank_limit:]

    return retrieved_docs, reasoning_traces, round_diagnostics


def _compute_base_node_weights(index, query: str) -> np.ndarray:
    """Return the same base node weights used by retrieve()."""
    return index._compute_node_weights(query=query)


# ── IRCoT reasoning (ported from feature/ircot-baseline) ─────────

IRCOT_ONE_SHOT_DOCS = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million\n"""
)

IRCOT_ONE_SHOT_DEMO = (
    f'{IRCOT_ONE_SHOT_DOCS}'
    '\n\nQuestion: '
    f"When was Neville A. Stanton's employer founded?"
    '\nThought: '
    f"The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862."
    '\n\n'
)

IRCOT_SYSTEM = (
    'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
    '\n\n'
    f'{IRCOT_ONE_SHOT_DEMO}'
)


def ircot_reason_step(query: str, passages: List[str], thoughts: List[str], llm_client) -> str:
    """Generate one IRCoT thought step."""
    prompt_user = ''
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    messages = [
        {"role": "system", "content": IRCOT_SYSTEM},
        {"role": "user", "content": prompt_user},
    ]

    try:
        response = llm_client.infer(messages=messages)
        if isinstance(response, tuple):
            response = response[0]
        if not isinstance(response, str):
            response = response[0]["content"]
        return response
    except Exception as e:
        logger.warning(f"IRCoT reason_step failed: {e}")
        return ''


def ircot_retrieve(index, question: str, llm_client, max_rounds: int = 3):
    """IRCoT: iterative retrieval with chain-of-thought on NER pipeline.

    Each round:
    1. Retrieve with current query (original question + accumulated thoughts)
    2. LLM generates one thought based on retrieved docs
    3. If thought contains "So the answer is:", stop
    4. Otherwise, append thought and re-retrieve

    Returns: (retrieved_docs, thoughts)
    """
    thoughts = []

    for round_i in range(max_rounds):
        # Build query: original question + all previous thoughts
        current_query = question
        if thoughts:
            current_query = question + " " + " ".join(thoughts)

        # Retrieve using NER pipeline
        sorted_doc_ids, sorted_doc_scores = index.retrieve(current_query)
        retrieved_docs = [index.passages[index.passage_keys[did]] for did in sorted_doc_ids]

        # Generate next thought
        thought = ircot_reason_step(
            query=question,
            passages=retrieved_docs[:5],
            thoughts=thoughts,
            llm_client=llm_client,
        )

        logger.info(f"  IRCoT round {round_i}: thought='{thought[:80]}...'")

        # Check if LLM reached final answer
        if "so the answer is:" in thought.lower():
            thoughts.append(thought)
            logger.info(f"  IRCoT stopped at round {round_i} (answer found)")
            break

        thoughts.append(thought)

    # Final retrieval with all thoughts
    final_query = question + " " + " ".join(thoughts) if thoughts else question
    sorted_doc_ids, sorted_doc_scores = index.retrieve(final_query)
    retrieved_docs = [index.passages[index.passage_keys[did]] for did in sorted_doc_ids]

    return retrieved_docs, thoughts


class SimpleLLM:
    """Simple OpenAI-compatible LLM client with retry."""
    def __init__(self, model_name, base_url, max_retries=3, retry_delay=10):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, timeout=120)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def infer(self, messages):
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"  LLM API error (attempt {attempt+1}/{self.max_retries}): {e}, retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    raise


def run_evaluation(args):
    from src.hipporag.embedding_model import _get_embedding_model_class

    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    all_data = json.load(open(args.data_path))
    if args.sample_limit and args.sample_limit < len(all_data):
        data = all_data[:args.sample_limit]
    else:
        data = all_data
    logger.info(f"Loaded {len(data)} samples (total {len(all_data)} in dataset)")

    embedding_model_name = os.getenv(
        "EMBEDDING_MODEL_NAME", "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model_name = os.getenv("LLM_MODEL_NAME", "qwen-plus")
    aliyun_base_url = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        os.environ["OPENAI_API_KEY"] = "sk-396199ed7af84eff8a0cf7a71b797601"

    logger.info(f"Loading embedding model: {embedding_model_name}")
    emb_model = _get_embedding_model_class(
        embedding_model_name=embedding_model_name
    )(embedding_model_name=embedding_model_name)
    # Override batch_size for large models (GTE-Qwen2-7B needs smaller batch on L40s)
    emb_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "0"))
    if emb_batch_size > 0:
        emb_model.batch_size = emb_batch_size
        logger.info(f"  Overriding embedding batch_size to {emb_batch_size}")

    logger.info(f"Setting up LLM: {llm_model_name}")
    llm_client = SimpleLLM(
        model_name=llm_model_name,
        base_url=aliyun_base_url,
    )

    ner_cache_path = args.ner_cache
    ner_cache = {}
    if ner_cache_path and os.path.exists(ner_cache_path):
        logger.info(f"Loading NER cache: {ner_cache_path}")
        ner_cache = json.load(open(ner_cache_path))
        logger.info(f"  {len(ner_cache)} cached entries")

    openie_cache = {}
    if args.openie_cache and os.path.exists(args.openie_cache):
        logger.info(f"Loading OpenIE cache for entity extraction: {args.openie_cache}")
        openie_data = json.load(open(args.openie_cache))
        for doc in openie_data["docs"]:
            if "text" in doc:
                text_key = doc["text"]
            else:
                parts = doc["passage"].split("\n", 1)
                text_key = parts[1] if len(parts) > 1 else doc["passage"]

            entities = []
            if "named_entities" in doc and doc["named_entities"]:
                entities = doc["named_entities"]
            elif "extracted_triples" in doc:
                ent_set = set()
                for t in doc["extracted_triples"]:
                    if len(t) >= 3:
                        ent_set.add(t[0])
                        ent_set.add(t[2])
                entities = sorted(ent_set)
            openie_cache[text_key] = entities
        logger.info(f"  {len(openie_cache)} docs with entities from OpenIE")

    save_dir = "outputs/musique_ner_pipeline_eval"
    os.makedirs(save_dir, exist_ok=True)
    mode = getattr(args, 'mode', 'base')
    output_path = _make_results_output_path(save_dir, mode, args.max_rounds)

    all_results = []
    total_start = time.time()

    global_index = None
    if getattr(args, 'global_index', False):
        all_docs = []
        seen = set()
        for sample in all_data:  # use ALL samples for global corpus, not just data[:sample_limit]
            for para in sample.get("paragraphs", []):
                t = para.get("paragraph_text", "")
                if t and t not in seen:
                    seen.add(t)
                    all_docs.append(t)

        global_ner_results = {}
        for doc_text in all_docs:
            if doc_text in ner_cache:
                global_ner_results[doc_text] = ner_cache[doc_text]
            elif doc_text in openie_cache:
                global_ner_results[doc_text] = openie_cache[doc_text]
            else:
                entities = llm_ner(doc_text, llm_client)
                global_ner_results[doc_text] = entities
                ner_cache[doc_text] = entities

        # Check for cached global index
        _syn_th = float(os.environ.get("SYNONYMY_THRESHOLD", "0.8"))
        _syn_tag = f"_syn{_syn_th}" if _syn_th != 0.8 else ""
        cache_path = os.path.join(save_dir, f"global_ner_index{_syn_tag}.pkl")
        if os.path.exists(cache_path):
            global_index = NERIndex.load(cache_path, embedding_model=emb_model)
        else:
            logger.info(f"Building GLOBAL NER index for {len(all_docs)} passages...")
            global_index = NERIndex(embedding_model=emb_model)
            global_index.build(all_docs, global_ner_results)
            logger.info(f"  Global index: {global_index.graph.vcount()} nodes, {global_index.graph.ecount()} edges")
            global_index.save(cache_path)

    for idx, sample in enumerate(data):
        question = sample.get("question", "")
        paragraphs = sample.get("paragraphs", [])
        answer = sample.get("answer", "")
        answer_aliases = sample.get("answer_aliases", [])
        docs = [para.get("paragraph_text", "") for para in paragraphs]
        gold_docs = [
            para.get("paragraph_text", "")
            for para in paragraphs
            if para.get("is_supporting", False)
        ]

        logger.info(f"[{idx+1}/{len(data)}] {question[:80]}...")

        index = None
        reasoning_traces = []
        round_diagnostics = []
        try:
            if global_index is None:
                ner_results = {}
                for doc_text in docs:
                    if doc_text in ner_cache:
                        ner_results[doc_text] = ner_cache[doc_text]
                    elif doc_text in openie_cache:
                        ner_results[doc_text] = openie_cache[doc_text]
                    else:
                        entities = llm_ner(doc_text, llm_client)
                        ner_results[doc_text] = entities
                        ner_cache[doc_text] = entities

                index = NERIndex(embedding_model=emb_model)
                index.build(docs, ner_results)
            else:
                index = global_index

            q_entities = query_ner(question, llm_client)
            logger.info(f"  Query entity pairs: {q_entities}")

            # Baseline: single-round retrieval without query entity linking
            base_sorted_ids, _ = index.retrieve(question)
            base_docs = [index.passages[index.passage_keys[did]] for did in base_sorted_ids]
            base_answer = llm_qa(question, base_docs[:5], llm_client)
            base_em = int(check_em(base_answer, answer, answer_aliases))
            base_f1 = round(check_f1(base_answer, answer, answer_aliases), 4)
            base_r1 = recall_at_k(base_docs, gold_docs, 1)
            base_r2 = recall_at_k(base_docs, gold_docs, 2)
            base_r5 = recall_at_k(base_docs, gold_docs, 5)
            base_r10 = recall_at_k(base_docs, gold_docs, 10)

            if mode == 'ircot':
                retrieved_docs, reasoning_traces = ircot_retrieve(
                    index, question, llm_client, max_rounds=args.max_rounds)
            elif args.max_rounds > 1:
                retrieved_docs, reasoning_traces, round_diagnostics = iterative_retrieve(
                    index, question, llm_client, max_rounds=args.max_rounds, gold_docs=gold_docs,
                    query_entities=q_entities)
            else:
                sorted_doc_ids, sorted_doc_scores = index.retrieve(question, query_entities=q_entities)
                retrieved_docs = [index.passages[index.passage_keys[did]] for did in sorted_doc_ids]

            r1 = recall_at_k(retrieved_docs, gold_docs, 1)
            r2 = recall_at_k(retrieved_docs, gold_docs, 2)
            r5 = recall_at_k(retrieved_docs, gold_docs, 5)
            r10 = recall_at_k(retrieved_docs, gold_docs, 10)

            qa_answer = llm_qa(question, retrieved_docs[:5], llm_client)
            em_correct = check_em(qa_answer, answer, answer_aliases)
            f1_score = check_f1(qa_answer, answer, answer_aliases)

        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            retrieved_docs = []
            reasoning_traces = []
            round_diagnostics = []
            r1 = r2 = r5 = r10 = 0.0
            qa_answer = "Error"
            em_correct = False
            f1_score = 0.0
            base_answer = "Error"
            base_em = 0
            base_f1 = 0.0
            base_r1 = base_r2 = base_r5 = base_r10 = 0.0

        result = {
            "idx": idx,
            "question": question,
            "gold_answer": answer,
            "gold_aliases": answer_aliases,
            "baseline_answer": base_answer,
            "baseline_em": base_em,
            "baseline_f1": base_f1,
            "baseline_recall": {"R@1": base_r1, "R@2": base_r2, "R@5": base_r5, "R@10": base_r10},
            "ner_answer": qa_answer,
            "ner_em": int(em_correct),
            "ner_f1": round(f1_score, 4),
            "ner_recall": {"R@1": r1, "R@2": r2, "R@5": r5, "R@10": r10},
            "reasoning_traces": reasoning_traces,
            "round_diagnostics": round_diagnostics,
            "n_entities": len(index.entity_keys) if index is not None and hasattr(index, 'entity_keys') else 0,
            "n_sentences": len(index.sentences) if index is not None and hasattr(index, 'sentences') else 0,
            "graph_nodes": index.graph.vcount() if index is not None and index.graph else 0,
            "graph_edges": index.graph.ecount() if index is not None and index.graph else 0,
        }
        all_results.append(result)

        em_str = "Y" if em_correct else "N"
        logger.info(f"  Base EM={'Y' if base_em else 'N'} R@5={base_r5:.2f} | NER EM={em_str} F1={f1_score:.2f} R@5={r5:.2f} | Gold='{answer}'")

        if global_index is None and index is not None:
            del index
            gc.collect()

        if (idx + 1) % 5 == 0 or (idx + 1) == len(data):
            n = len(all_results)
            avg_base_em = sum(r["baseline_em"] for r in all_results) / n
            avg_base_f1 = sum(r["baseline_f1"] for r in all_results) / n
            avg_base_r5 = sum(r["baseline_recall"]["R@5"] for r in all_results) / n
            avg_em = sum(r["ner_em"] for r in all_results) / n
            avg_f1 = sum(r["ner_f1"] for r in all_results) / n
            avg_r1 = sum(r["ner_recall"]["R@1"] for r in all_results) / n
            avg_r2 = sum(r["ner_recall"]["R@2"] for r in all_results) / n
            avg_r5 = sum(r["ner_recall"]["R@5"] for r in all_results) / n
            avg_r10 = sum(r["ner_recall"]["R@10"] for r in all_results) / n
            logger.info(f"  >>> Progress: {n}/{len(data)} | Base EM={avg_base_em:.3f} R@5={avg_base_r5:.3f} | NER EM={avg_em:.3f} F1={avg_f1:.3f} R@5={avg_r5:.3f}")

            with open(output_path, "w") as f:
                json.dump({
                    "config": {
                        "method": "ner_pipeline",
                        "sample_limit": args.sample_limit,
                        "max_rounds": args.max_rounds,
                        "llm": llm_model_name,
                        "embedding": embedding_model_name,
                    },
                    "summary": {
                        "n_completed": n,
                        "baseline_em": round(avg_base_em, 4),
                        "baseline_f1": round(avg_base_f1, 4),
                        "baseline_recall": {"R@5": round(avg_base_r5, 4)},
                        "ner_em": round(avg_em, 4),
                        "ner_f1": round(avg_f1, 4),
                        "ner_recall": {
                            "R@1": round(avg_r1, 4),
                            "R@2": round(avg_r2, 4),
                            "R@5": round(avg_r5, 4),
                            "R@10": round(avg_r10, 4),
                        },
                    },
                    "results": all_results,
                }, f, indent=2)

    if ner_cache_path:
        existing = None
        if os.path.exists(ner_cache_path):
            with open(ner_cache_path) as f:
                existing = f.read()
        new_payload = json.dumps(ner_cache, ensure_ascii=False, indent=2, sort_keys=True)
        if existing != new_payload:
            with open(ner_cache_path, "w") as f:
                f.write(new_payload)
            logger.info(f"Saved NER cache: {ner_cache_path} ({len(ner_cache)} entries)")

    n = len(all_results)
    avg_base_em = sum(r["baseline_em"] for r in all_results) / n
    avg_base_f1 = sum(r["baseline_f1"] for r in all_results) / n
    avg_base_r5 = sum(r["baseline_recall"]["R@5"] for r in all_results) / n
    avg_em = sum(r["ner_em"] for r in all_results) / n
    avg_f1 = sum(r["ner_f1"] for r in all_results) / n
    avg_r1 = sum(r["ner_recall"]["R@1"] for r in all_results) / n
    avg_r5 = sum(r["ner_recall"]["R@5"] for r in all_results) / n

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"NER Pipeline Results: {n} samples")
    print(f"  Baseline EM:  {avg_base_em:.4f}  F1: {avg_base_f1:.4f}  R@5: {avg_base_r5:.4f}")
    print(f"  NER EM:       {avg_em:.4f}  F1: {avg_f1:.4f}")
    print(f"  NER R@1:      {avg_r1:.4f}")
    print(f"  NER R@5:      {avg_r5:.4f}")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"  Results: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="musique.json")
    parser.add_argument("--sample_limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ner_cache", type=str, default=None,
                        help="Path to pre-computed NER results JSON")
    parser.add_argument("--openie_cache", type=str,
                        default="outputs/musique/openie_results_ner_qwen-plus.json",
                        help="Path to OpenIE cache (fallback for entity extraction)")
    parser.add_argument("--max_rounds", type=int, default=1,
                        help="Max reasoning rounds (1=base only, >1=reasoning)")
    parser.add_argument("--mode", type=str, default="base",
                        choices=["base", "reasoning", "ircot"],
                        help="Reasoning mode: base, reasoning (graph-reshape), ircot")
    parser.add_argument("--global_index", action="store_true",
                        help="Build one global index from all samples' passages instead of per-sample")
    args = parser.parse_args()
    # For reasoning/ircot, ensure max_rounds > 1
    if args.mode in ("reasoning", "ircot") and args.max_rounds <= 1:
        args.max_rounds = 3
    run_evaluation(args)


if __name__ == "__main__":
    main()
