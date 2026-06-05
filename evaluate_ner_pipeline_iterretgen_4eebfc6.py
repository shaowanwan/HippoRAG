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
    elif mode == "self_ask":
        output_tag = f"_selfask_rounds{max_rounds}"
    elif mode == "iter_retgen":
        output_tag = f"_iterretgen_rounds{max_rounds}"
    elif mode == "iter_retgen_bridge":
        output_tag = f"_iterretgenbridge_rounds{max_rounds}"
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


def _first_json_obj(response):
    """Extract the FIRST complete JSON object from an LLM response.

    Tolerates trailing prose, code fences, and multiple objects: json.JSONDecoder
    .raw_decode() parses one object and ignores trailing data, so it does not raise
    'Extra data' the way json.loads on a greedy {.*} regex match does (the failure
    mode that broke this on current qwen-plus). Scans each '{' until one decodes.
    """
    if not isinstance(response, str):
        return None
    dec = json.JSONDecoder()
    start = response.find('{')
    while start != -1:
        try:
            obj, _ = dec.raw_decode(response, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = response.find('{', start + 1)
    return None


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
        data = _first_json_obj(response)
        if data is not None:
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
        # Parse the first complete JSON object (tolerant of trailing data)
        data = _first_json_obj(response)
        if data is not None:
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

        # Optional: randomly drop edges to test robustness
        _drop_ee = float(os.getenv("GRAPH_DROP_EE", "0.0"))
        _drop_ep = float(os.getenv("GRAPH_DROP_EP", "0.0"))
        if _drop_ee > 0 or _drop_ep > 0:
            import random
            random.seed(42)
            passage_set = set(index.passage_node_idxs)
            ee_ids = [e.index for e in index.graph.es
                      if e.source not in passage_set and e.target not in passage_set]
            ep_ids = [e.index for e in index.graph.es
                      if e.source in passage_set or e.target in passage_set]
            drop_ids = []
            if _drop_ee > 0:
                n = int(len(ee_ids) * _drop_ee)
                drop_ids += random.sample(ee_ids, n)
                logger.info(f"Dropping {_drop_ee*100}% EE edges: {len(ee_ids)} -> {len(ee_ids)-n}")
            if _drop_ep > 0:
                n = int(len(ep_ids) * _drop_ep)
                drop_ids += random.sample(ep_ids, n)
                logger.info(f"Dropping {_drop_ep*100}% EP edges: {len(ep_ids)} -> {len(ep_ids)-n}")
            index.graph.delete_edges(drop_ids)
            logger.info(f"Final graph: {index.graph.ecount()} edges")

        return index

    def augment_cross_sentence(self, cross_cache: Dict[str, dict]):
        """Augment graph with cross-sentence relations and coreference.

        cross_cache: {passage_text: {
            "extra_entities_by_sentence": {"0": ["entity1"], ...},
            "cross_sentence_pairs": [["e1", "e2", "description"], ...]
        }}
        """
        # Build passage_text → (passage_id, [sent_ids_in_order]) mapping
        passage_to_sents = defaultdict(list)
        for sent_id in self.sentence_ids:
            passage_id = self.sentences[sent_id]["passage_id"]
            passage_to_sents[passage_id].append(sent_id)

        text_to_passage_id = {text: pid for pid, text in self.passages.items()}

        new_edges = []
        new_weights = []
        new_pairs = []
        new_pair_texts = []
        stats = {"extra_entities": 0, "coref_edges": 0, "cross_pairs": 0, "cross_edges": 0}

        for passage_text, relations in cross_cache.items():
            passage_id = text_to_passage_id.get(passage_text)
            if not passage_id:
                continue
            sent_ids = passage_to_sents.get(passage_id, [])

            # 1) Coreference: add extra entities to sentences → new co-occurrence edges
            for sent_idx_str, extra_ents in relations.get("extra_entities_by_sentence", {}).items():
                sent_idx = int(sent_idx_str)
                if sent_idx >= len(sent_ids):
                    continue
                sent_id = sent_ids[sent_idx]
                sent_data = self.sentences[sent_id]
                existing_ent_keys = {ek for _, ek in sent_data["entities"]}

                for ent_name in extra_ents:
                    ent_key = compute_hash(ent_name.lower(), prefix="entity-")
                    if ent_key not in self.node_name_to_idx:
                        continue  # entity not in graph
                    if ent_key in existing_ent_keys:
                        continue  # already in this sentence

                    # Add entity to sentence
                    sent_data["entities"].append((ent_name.lower(), ent_key))
                    self.entity_to_sentences[ent_key].append(sent_id)
                    self.entity_to_passages[ent_key].add(passage_id)
                    stats["extra_entities"] += 1

                    # New co-occurrence pairs with existing entities in this sentence
                    ent_idx = self.node_name_to_idx.get(ent_key)
                    for existing_text, existing_key in sent_data["entities"]:
                        if existing_key == ent_key:
                            continue
                        if existing_key not in existing_ent_keys:
                            continue  # skip entities we just added
                        existing_idx = self.node_name_to_idx.get(existing_key)
                        if ent_idx is not None and existing_idx is not None:
                            new_edges.append((ent_idx, existing_idx))
                            new_weights.append(1.0)
                            stats["coref_edges"] += 1
                            new_pairs.append((ent_name.lower(), ent_key, existing_text, existing_key, sent_id))
                            new_pair_texts.append(f"{ent_name.lower()} | {existing_text} | {sent_data['text']}")

                    existing_ent_keys.add(ent_key)

            # 2) Cross-sentence pairs: add edges with synthetic sentence
            for pair in relations.get("cross_sentence_pairs", []):
                if len(pair) < 3:
                    continue
                e1, e2, desc = pair[0], pair[1], pair[2]
                k1 = compute_hash(e1.lower(), prefix="entity-")
                k2 = compute_hash(e2.lower(), prefix="entity-")
                idx1 = self.node_name_to_idx.get(k1)
                idx2 = self.node_name_to_idx.get(k2)
                if idx1 is None or idx2 is None:
                    continue

                # Add graph edge (parallel edge OK, different source)
                new_edges.append((idx1, idx2))
                new_weights.append(1.0)
                stats["cross_pairs"] += 1
                stats["cross_edges"] += 1

                # Create synthetic sentence for pair embedding
                synthetic_sent_id = f"cross-{passage_id}-{k1[:8]}-{k2[:8]}"
                self.sentences[synthetic_sent_id] = {
                    "text": desc,
                    "passage_id": passage_id,
                    "entities": [(e1.lower(), k1), (e2.lower(), k2)],
                }
                new_pairs.append((e1.lower(), k1, e2.lower(), k2, synthetic_sent_id))
                new_pair_texts.append(f"{e1.lower()} | {e2.lower()} | {desc}")

                # Ensure entities are linked to passage
                self.entity_to_passages[k1].add(passage_id)
                self.entity_to_passages[k2].add(passage_id)

                # Add entity-passage edges if not already connected
                p_idx = self.node_name_to_idx.get(passage_id)
                if p_idx is not None:
                    if not self.graph.are_connected(idx1, p_idx):
                        new_edges.append((idx1, p_idx))
                        new_weights.append(1.0)
                    if not self.graph.are_connected(idx2, p_idx):
                        new_edges.append((idx2, p_idx))
                        new_weights.append(1.0)

        # Add edges to graph
        if new_edges:
            self.graph.add_edges(new_edges, attributes={"weight": new_weights})

        # Compute pair embeddings for new pairs and append (no sentence embeddings for synthetic sentences)
        if new_pair_texts:
            new_embs = self.embedding_model.batch_encode(new_pair_texts)
            if self.pair_embeddings is not None and len(self.pair_embeddings) > 0:
                self.pair_embeddings = np.vstack([self.pair_embeddings, new_embs])
            else:
                self.pair_embeddings = new_embs
            self.pairs.extend(new_pairs)

        logger.info(f"  Augmented: +{stats['extra_entities']} coref entities, "
                     f"+{stats['coref_edges']} coref edges, "
                     f"+{stats['cross_pairs']} cross-sentence pairs, "
                     f"+{stats['cross_edges']} cross edges, "
                     f"+{len(new_pairs)} new pair embeddings")
        logger.info(f"  Graph now: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")

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


    def _compute_node_weights(self, query: str, passage_weight_scale: float = float(os.environ.get("PASSAGE_WEIGHT_SCALE", "0.05")),
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

        return node_weights

    def retrieve(self, query: str, top_k: int = 5, passage_weight_scale: float = float(os.environ.get("PASSAGE_WEIGHT_SCALE", "0.05")),
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
                return sorted_ids, passage_scores[sorted_ids], base_node_weights, np.zeros(self.graph.vcount())
            return sorted_ids, passage_scores[sorted_ids]

        ppr_result = _run_ppr(
            self,
            reset_prob=node_weights,
            damping=0.5,
            graph=working_graph,
            return_all_scores=return_node_weights,
        )
        if return_node_weights:
            sorted_doc_ids, sorted_doc_scores, all_ppr_scores = ppr_result
            return sorted_doc_ids, sorted_doc_scores, base_node_weights, all_ppr_scores
        else:
            sorted_doc_ids, sorted_doc_scores = ppr_result
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

# Second demo: 3-hop multi-step reasoning (same as IRCOT_TWO_SHOT_DEMO for consistency across reasoning/IRCoT)
QA_TWO_SHOT_INPUT = (
    "Wikipedia Title: Smoke in tha City\nSmoke in tha City is the ninth studio album by American rapper MC Eiht. The album was released in 2004 on Tha Hall Records.\n"
    "Wikipedia Title: MC Eiht\nAaron Tyler (born May 22, 1971), better known by his stage name MC Eiht, is an American rapper born in Compton, California. He is a former member of the rap group Compton's Most Wanted.\n"
    "Wikipedia Title: Compton, California\nCompton is a city in southern Los Angeles County, California, located south of downtown Los Angeles.\n\n"
    "Question: In what county is the birthplace of the performer of Smoke in tha City?\nThought: "
)

QA_TWO_SHOT_OUTPUT = (
    "The performer of Smoke in tha City is MC Eiht. MC Eiht was born in Compton, California. Compton is located in Los Angeles County. "
    "\nAnswer: Los Angeles County."
)


def llm_qa(query: str, docs: List[str], llm_client, reasoning_traces: List[str] = None) -> str:
    """QA with HippoRAG-aligned prompt: CoT Thought + Answer extraction.
    If QA_USE_TRACES env=1 and reasoning_traces provided, prepend trace summary
    (gap analysis) to the prompt before docs."""
    prompt_user = ''
    _qa_trace_mode = os.getenv("QA_USE_TRACES", "0")
    if _qa_trace_mode != "0" and reasoning_traces:
        prompt_user += "Confirmed facts from prior retrieval rounds:\n" if _qa_trace_mode == "confirmed" else "Reasoning trace from prior retrieval rounds:\n"
        for i, t in enumerate(reasoning_traces):
            if _qa_trace_mode == "confirmed":
                # Trace format: "What is confirmed: X | What is missing: Y"
                if "|" in t and "What is confirmed:" in t:
                    confirmed = t.split("|")[0].strip()
                    prompt_user += f"- Round {i+1}: {confirmed}\n"
                else:
                    prompt_user += f"- Round {i+1}: {t}\n"  # fallback
            else:
                prompt_user += f"- Round {i+1}: {t}\n"
        prompt_user += "\n"
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
        # Extract answer after "Answer:" — regex stops at first period/newline/end
        # (consistent with ircot_retrieve's "so the answer is:" extraction)
        import re as _re
        m = _re.search(r'Answer:?\s*(.+?)(?:\.\s*$|\n|$)', response, _re.IGNORECASE | _re.DOTALL)
        if m:
            pred_ans = m.group(1).strip().strip('"').strip("'").rstrip('.').strip()
        else:
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
MINI_PPR_USE_LAST_ROUND = os.getenv("MINI_PPR_LAST_ROUND", "0") == "1"
HUB_PRUNE = os.getenv("HUB_PRUNE", "0") == "1"
HUB_PRUNE_KEEP = float(os.getenv("HUB_PRUNE_KEEP", "0.3"))
HUB_PRUNE_ALPHA = float(os.getenv("HUB_PRUNE_ALPHA", "2.0"))
PPR_SOFT_REWEIGHT = os.getenv("PPR_SOFT_REWEIGHT", "0") == "1"
PIPELINE_RRF_WEIGHT = 1.0
EXPANSION_RRF_WEIGHT = float(os.getenv("EXPANSION_RRF_WEIGHT", "0.5"))

# Prompt (from feature/graph-reshape query_rewriter.py — includes discovered_entities)
REWRITE_SYSTEM_PROMPT = """You are an expert in multi-hop question answering. You help solve complex questions through reasoning analysis and structured entity inference. Multi-hop questions can be abstracted as a chain: query entities → bridge entities → target entities (answer), where bridge entities are intermediate concepts that link what is asked to what is needed.

Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

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


# 1-shot demo for bridge discovery (used when BRIDGE_ONESHOT=1).
# Active bridge case: docs identify the bridge entity but not the target answer.
# Demonstrates the JSON output format + the discovered_entities/rewritten_query pattern.
BRIDGE_ONESHOT_USER = """Original query: Where was the founder of FC Barcelona born?
Current query (round 0): Where was the founder of FC Barcelona born?

Retrieved documents so far:
[1] FC Barcelona was founded in 1899 by Joan Gamper, a Swiss footballer who emigrated to Spain.
[2] FC Barcelona is one of the most successful clubs in football history.
[3] El Clasico refers to matches between Barcelona and Real Madrid.
[4] Lionel Messi joined Barcelona in 2003 at age 13.
[5] Camp Nou is the home stadium of FC Barcelona.

Analyze and provide your reasoning output in JSON."""

BRIDGE_ONESHOT_ASSISTANT = """{
    "analysis": "Found: FC Barcelona was founded by Joan Gamper (Doc 1). Missing: Joan Gamper's specific birthplace — Doc 1 says 'Swiss' but no city.",
    "discovered_entities": ["joan gamper"],
    "rewritten_query": "Where was Joan Gamper born?",
    "should_stop": false
}"""


# Variant: default REWRITE_SYSTEM_PROMPT + one rule encouraging bridges from DIFFERENT docs.
# Gated by env DIVERSE_BRIDGES=1. Targets 2Wiki R@2 limitation: bridges often cluster in one doc.
REWRITE_SYSTEM_PROMPT_DIVERSE_BRIDGES = """You are an expert in multi-hop question answering. You help solve complex questions through reasoning analysis and structured entity inference. Multi-hop questions can be abstracted as a chain: query entities → bridge entities → target entities (answer), where bridge entities are intermediate concepts that link what is asked to what is needed.

Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

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
- Prefer bridge entities from DIFFERENT retrieved documents to ensure broader coverage across the reasoning chain — avoid clustering all bridges in a single document.
- Rewritten query should be a natural language question, not keywords. It should incorporate discovered bridge entities.
- Focus on what information is missing and craft the query to find it.
"""


PLAN_SYSTEM_PROMPT = """You are a retrieval reasoning assistant. Given a complex multi-hop query and retrieved documents, your job is to:

1. Decompose the original query into sequential sub-questions that build on each other.
2. Identify which sub-question should be addressed next based on what's already found.
3. Discover bridge entities in the retrieved documents that help answer the current sub-question.
4. Rewrite the query to target the current sub-question's missing information.

Respond in JSON format:
{
    "sub_questions": ["Q1: ...", "Q2: ...", "Q3: ..."],
    "current_sub_question": "The sub-question to address next",
    "analysis": "What's found so far and what's needed for the current sub-question",
    "discovered_entities": ["entity1", "entity2"],
    "rewritten_query": "Query targeting the current sub-question's missing info",
    "should_stop": false
}

Rules:
- Decompose the query into 2-4 sub-questions that form a reasoning chain.
- If the retrieved documents already answer ALL sub-questions, set should_stop=true.
- "discovered_entities" should list bridge entities found in docs that connect to the CURRENT sub-question's answer. Use lowercase. List 1-3 entities max.
- Focus on ONE sub-question per round. Move to the next only when the current one is answered.
"""


REWRITE_SYSTEM_PROMPT_HOTPOTQA = """You are an expert in multi-hop question answering. You help solve complex questions through reasoning analysis and structured entity inference. Multi-hop questions can be abstracted as a chain: query entities → bridge entities → target entities (answer), where bridge entities are intermediate concepts that link what is asked to what is needed.

Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

1. Gap analysis: What specific information has been confirmed? What specific piece is still MISSING to answer the original query?
2. Identify bridge entities in the retrieved documents that connect confirmed information to the missing piece.
3. Write a query that targets the identified gap.
4. Decide whether to continue retrieval or stop.

Respond in JSON format:
{
    "analysis": "What is confirmed: ... | What is missing: ...",
    "bridge_entities": ["entity1", "entity2"],
    "discovered_entities": ["entity1", "entity2"],
    "context_scenario": "Hypothesized context: time period, location, domain, related concepts",
    "rewritten_query": "Natural-language question targeting the missing gap",
    "should_stop": false
}

Rules:
- Do not fabricate a gap when the docs already provide a clear answer for straightforward questions.
- "bridge_entities": List 2-5 entities found in the documents that connect what is already confirmed to what is still missing. Include specific event related entities or broader contextual entities that could bridge to the missing information. Use lowercase. Always list at least 2. Do not repeat entities already mentioned in previous discovered entities.
- "discovered_entities": same as bridge_entities.
- "context_scenario": Short hypothesized context (time period, location, domain, related concepts) that helps frame the reasoning. Stored as reasoning trace for future rounds. Do NOT use context_scenario terms directly in rewritten_query.
- The rewritten query should reference the original query as anchor. Write it as a natural language question.
- Build upon previous reasoning traces — do not abandon previously confirmed facts, but focus more on the retrieved docs and find out entities that will help reach the missing answer.
"""


# ──────────────────────────────────────────────────────────────────
# CANONICAL 1000-SAMPLE QWEN-PLUS BASELINE PROMPT
# This is the EXACT prompt that produced PID 34587 (MuSiQue 1000-sample,
# qwen-plus + MiniLM, EM 0.391 / F1 0.491 / R@5 0.636 — paper main table).
# Extracted from git commit 5cae12e (2026-04-02, last commit before PID 34587).
# DO NOT MODIFY without renaming the constant.
# To reproduce paper baseline: set env `BASELINE_PID34587=1` (gated in
# reason_and_rewrite — see prompt selection logic).
# ──────────────────────────────────────────────────────────────────
REWRITE_SYSTEM_PROMPT_PID34587_BASELINE = """You are a retrieval reasoning assistant. Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

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


# Variant: PID34587_BASELINE + chain framing prefix (same as default REWRITE_SYSTEM_PROMPT).
# Gated by env CHAIN_FRAMING=1. Tests whether the chain framing line lifts MuSiQue
# (it lifts 2Wiki by ~6.3pp EM in pilot).
REWRITE_SYSTEM_PROMPT_PID34587_WITH_CHAIN = """You are an expert in multi-hop question answering. You help solve complex questions through reasoning analysis and structured entity inference. Multi-hop questions can be abstracted as a chain: query entities → bridge entities → target entities (answer), where bridge entities are intermediate concepts that link what is asked to what is needed.

You are a retrieval reasoning assistant. Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

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


# Variant: rewrite query FIRST, then bridge discovery from docs (targeting the rewritten query).
# Gated by env REWRITE_FIRST=1. Minimal-edit variant of PID34587_BASELINE.
REWRITE_SYSTEM_PROMPT_REWRITE_FIRST = """You are a retrieval reasoning assistant. Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

1. Analyze what information has been found and what is still missing.
2. Rewrite the query to better target the missing information.
3. Identify key bridge entities in the retrieved documents that help answer the rewritten query.
4. Decide whether to continue retrieval or stop.

Respond in JSON format:
{
    "analysis": "Brief analysis of what's found vs missing",
    "rewritten_query": "The rewritten query targeting missing info",
    "discovered_entities": ["entity1", "entity2"],
    "should_stop": false
}

Rules:
- If the retrieved documents already contain sufficient information to answer the query, set should_stop=true and leave rewritten_query empty.
- "discovered_entities" should list key entities found in the retrieved documents that are important for answering the rewritten query but were NOT in the original query. These are bridge entities that help reach the missing information. Use lowercase. List 1-5 entities max.
- Rewritten query should be a natural language question, not keywords.
- Focus on what information is missing and craft the query to find it.
"""


# ──────────────────────────────────────────────────────────────────
# ⚠️  EXPERIMENTAL VARIANT — NOT the canonical baseline.
# Despite the name "QWENPLUS_BASELINE", this prompt was NEVER used for
# the 1000-sample paper baseline (PID 34587, see REWRITE_SYSTEM_PROMPT_PID34587_BASELINE).
# It's a working-tree experimental variant exploring "human memory / context_scenario"
# framing. Was never committed to git, never run at 1000-sample scale.
# Keep gated behind `QWENPLUS_BASELINE=1` env flag if you want to test it.
# For canonical reproduction, use REWRITE_SYSTEM_PROMPT_PID34587_BASELINE instead.
# ──────────────────────────────────────────────────────────────────
REWRITE_SYSTEM_PROMPT_QWENPLUS_BASELINE = """You are a retrieval reasoning assistant inspired by how human memory uses both observed cues and hypothetical context. Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

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
- "context_scenario": Short natural language description of the hypothesized context. Stored as your reasoning trace and seen by future rounds, but NOT used directly for retrieval.
- "rewritten_query": Focused query using ONLY bridge entities + reformulation of the original question. DO NOT add context_scenario, time periods, or hypothesized terms here. Keep it short and concrete.
- "should_stop": true ONLY if retrieved documents already contain a complete answer.
"""


def reason_and_rewrite(original_query: str, current_query: str, retrieved_docs: List[str],
                       round_idx: int, previous_traces: List[str], llm_client) -> dict:
    """LLM reasoning: analyze retrieved docs, rewrite query, discover bridge entities."""
    _plan_mode = os.getenv("PLAN_MODE", "0") == "1"

    docs_text = ""
    _n_docs = int(os.getenv("REASONING_TOP_K", "5"))
    for i, doc in enumerate(retrieved_docs[:_n_docs]):
        docs_text += f"[Doc {i+1}] {doc}\n\n"

    user_content = f"""Original query: {original_query}
Current query (round {round_idx}): {current_query}

Retrieved documents so far:
{docs_text}"""

    if previous_traces:
        user_content += "\nPrevious reasoning traces:\n"
        for t in previous_traces[-3:]:
            user_content += f"- {t}\n"

    # Feed back previously discovered bridges for iterative refinement
    if os.getenv("BRIDGE_REFINEMENT", "0") == "1" and round_idx > 0:
        # Get bridges from previous rounds (passed via traces or separate)
        # The discovered entities are embedded in the analysis text
        user_content += "\nIMPORTANT: Previous rounds already discovered general bridge entities. "
        user_content += "Now find MORE SPECIFIC entities that narrow down the answer. "
        user_content += "For example, if previous rounds found a country name, now find a specific city, person, or date within that country.\n"

    user_content += "\nAnalyze and provide your reasoning output in JSON."

    _hotpotqa_mode = os.getenv("HOTPOTQA_MODE", "0") == "1"
    _qwenplus_baseline = os.getenv("QWENPLUS_BASELINE", "0") == "1"
    _pid34587_baseline = os.getenv("BASELINE_PID34587", "0") == "1"
    _rewrite_first = os.getenv("REWRITE_FIRST", "0") == "1"
    _chain_framing = os.getenv("CHAIN_FRAMING", "0") == "1"
    _diverse_bridges = os.getenv("DIVERSE_BRIDGES", "0") == "1"
    if _plan_mode:
        sys_prompt = PLAN_SYSTEM_PROMPT
    elif _diverse_bridges:
        # default REWRITE_SYSTEM_PROMPT + diversity rule
        sys_prompt = REWRITE_SYSTEM_PROMPT_DIVERSE_BRIDGES
    elif _rewrite_first:
        # Variant: rewrite query first, then bridge discovery
        sys_prompt = REWRITE_SYSTEM_PROMPT_REWRITE_FIRST
    elif _chain_framing and _pid34587_baseline:
        # PID34587_BASELINE + chain framing prefix
        sys_prompt = REWRITE_SYSTEM_PROMPT_PID34587_WITH_CHAIN
    elif _pid34587_baseline:
        # Canonical 1000-sample qwen-plus baseline prompt (EM 0.391 / R@5 0.636)
        sys_prompt = REWRITE_SYSTEM_PROMPT_PID34587_BASELINE
    elif _qwenplus_baseline:
        sys_prompt = REWRITE_SYSTEM_PROMPT_QWENPLUS_BASELINE
    elif _hotpotqa_mode:
        sys_prompt = REWRITE_SYSTEM_PROMPT_HOTPOTQA
    else:
        sys_prompt = REWRITE_SYSTEM_PROMPT
    _bridge_oneshot = os.getenv("BRIDGE_ONESHOT", "0") == "1"
    messages = [
        {"role": "system", "content": sys_prompt},
    ]
    if _bridge_oneshot:
        messages.append({"role": "user", "content": BRIDGE_ONESHOT_USER})
        messages.append({"role": "assistant", "content": BRIDGE_ONESHOT_ASSISTANT})
    messages.append({"role": "user", "content": user_content})

    _hierarchical = os.getenv("HIERARCHICAL_BRIDGES", "0") == "1"
    defaults = {"analysis": "", "discovered_entities": [], "general_bridges": [], "specific_bridges": [],
                "discard_entities": [], "query_entities": [], "bridge_entities": [], "target_entities": [],
                "rewritten_query": "", "should_stop": False}
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
        # Structured 2-layer fields (HotpotQA prompt) — normalize then fall back to discovered_entities
        for _lk in ("query_entities", "bridge_entities"):
            if not isinstance(parsed.get(_lk, []), list):
                parsed[_lk] = []
            parsed[_lk] = [str(e).lower().strip() for e in parsed.get(_lk, [])
                           if e and isinstance(e, (str, int, float))][:4]
        # If LLM didn't populate discovered_entities, derive from bridge_entities
        if not isinstance(parsed.get("discovered_entities", []), list):
            parsed["discovered_entities"] = []
        parsed["discovered_entities"] = [
            str(e).lower().strip() for e in parsed["discovered_entities"] if e and isinstance(e, (str, int, float))
        ][:5]
        if not parsed["discovered_entities"] and parsed["bridge_entities"]:
            # Option A: MERGE_QUERY_ENTITIES=1 to add query_entities as expansion seeds
            if os.getenv("MERGE_QUERY_ENTITIES", "0") == "1":
                parsed["discovered_entities"] = (parsed.get("query_entities", []) + parsed["bridge_entities"])[:5]
            else:
                parsed["discovered_entities"] = parsed["bridge_entities"][:5]
        # Parse hierarchical bridges
        if not isinstance(parsed.get("general_bridges", []), list):
            parsed["general_bridges"] = []
        parsed["general_bridges"] = [
            str(e).lower().strip() for e in parsed.get("general_bridges", []) if e
        ][:3]
        if not isinstance(parsed.get("specific_bridges", []), list):
            parsed["specific_bridges"] = []
        parsed["specific_bridges"] = [
            str(e).lower().strip() for e in parsed.get("specific_bridges", []) if e
        ][:3]
        # Merge into discovered_entities if hierarchical mode (with type tag)
        if _hierarchical and (parsed["general_bridges"] or parsed["specific_bridges"]):
            parsed["discovered_entities"] = parsed["general_bridges"] + parsed["specific_bridges"]
            parsed["_bridge_types"] = {e: "general" for e in parsed["general_bridges"]}
            parsed["_bridge_types"].update({e: "specific" for e in parsed["specific_bridges"]})
        if not isinstance(parsed.get("discard_entities", []), list):
            parsed["discard_entities"] = []
        parsed["discard_entities"] = [
            str(e).lower().strip() for e in parsed.get("discard_entities", []) if e
        ]
        return parsed
    except Exception as e:
        logger.warning(f"Reasoning failed: {e}")
        defaults["analysis"] = str(e)[:200]
        defaults["should_stop"] = True
        return defaults


# ── Entity resolution (from feature/graph-reshape controller.py) ──

def _resolve_entities_in_graph(index, entity_names: List[str], threshold: float = None) -> Dict[str, Tuple[int, float]]:
    """Resolve entity names to graph vertex IDs with similarity scores.
    Returns {name: (vid, sim_score)}. Exact match gets sim=1.0.
    Uses: exact → fuzzy (edit distance ≤ 2) → embedding (high threshold)."""
    if threshold is None:
        threshold = float(os.getenv("ENTITY_MATCH_THRESHOLD", "0.55"))
    resolved = {}
    unresolved = []

    for name in entity_names:
        ent_key = compute_hash(name.lower(), prefix="entity-")
        vid = index.node_name_to_idx.get(ent_key)
        if vid is not None:
            resolved[name] = (vid, 1.0)
            logger.info(f"  Entity '{name}' found (exact) -> vertex {vid}")
        else:
            # Optional fuzzy match: try edit distance ≤ 2
            fuzzy_found = False
            if os.getenv("FUZZY_MATCH", "0") == "1":
                name_lower = name.lower().strip()
                for eidx, ent_name in enumerate(index.entities):
                    ent_lower = ent_name.lower()
                    if abs(len(name_lower) - len(ent_lower)) > 2:
                        continue
                    if len(name_lower) <= 30:
                        dist = sum(1 for a, b in zip(name_lower, ent_lower) if a != b) + abs(len(name_lower) - len(ent_lower))
                        if dist <= 2:
                            ent_key = index.entity_keys[eidx]
                            vid = index.node_name_to_idx.get(ent_key)
                            if vid is not None:
                                resolved[name] = (vid, 0.95)
                                logger.info(f"  Entity '{name}' found (fuzzy, dist={dist}) -> vertex {vid} ('{ent_name}')")
                                fuzzy_found = True
                                break
            if not fuzzy_found:
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
    if os.getenv("NO_DEGREE_ADAPTIVE", "0") == "1":
        return base_weight * sim
    deg = index.graph.degree(vid)
    _scale = float(os.getenv("BRIDGE_WEIGHT_SCALE", "1.0"))  # Option B: ×0.6 to lower bridge weight
    return base_weight * sim * (1.0 + math.log(deg + 1)) * _scale


def _get_bridge_query_sims(index, discovered: Dict[str, Tuple[int, float, float]],
                           query: str) -> Dict[int, float]:
    """For each bridge entity, compute similarity of 'entity | query' against pair embeddings.

    Returns {vid: max_pair_sim}. This measures how relevant the bridge entity is
    to the query in the context of existing entity relationships.
    """
    if index.pair_embeddings is None or len(index.pair_embeddings) == 0:
        return {vid: 1.0 for name, (vid, sim, dr) in discovered.items()}

    # Compute "bridge_entity | query" embeddings for all discovered entities
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

    # Normalize
    b_norms = np.linalg.norm(bridge_embs, axis=1, keepdims=True)
    b_norms = np.where(b_norms == 0, 1, b_norms)
    bridge_embs = bridge_embs / b_norms

    p_norms = np.linalg.norm(index.pair_embeddings, axis=1, keepdims=True)
    p_norms = np.where(p_norms == 0, 1, p_norms)
    normed_pairs = index.pair_embeddings / p_norms

    # Compute similarities: (n_bridge, n_pairs) -> max per bridge
    sims = bridge_embs @ normed_pairs.T
    max_sims = sims.max(axis=1)

    result = {}
    for i, vid in enumerate(bridge_vids):
        result[vid] = float(max(max_sims[i], 0.0))

    return result


def _llm_select_dead_end_bridges(index, base_node_weights: np.ndarray,
                                  ppr_scores: np.ndarray,
                                  original_query: str, current_query: str,
                                  top_docs: List[str], llm_client,
                                  previous_traces: List[str] = None) -> List[str]:
    """When dead-end seeds are detected, make a separate LLM call to select
    bridge entities from cross-passage candidates.
    Returns list of entity names (lowercase) to add as bridges.
    """
    passage_idx_set = set(index.passage_node_idxs)
    g = index.graph

    passage_to_ents = defaultdict(set)
    for ek, pids in index.entity_to_passages.items():
        for pid in pids:
            passage_to_ents[pid].add(ek)

    # Find dead-end seeds
    seed_vids = [v for v in range(len(base_node_weights))
                 if base_node_weights[v] > 0 and v not in passage_idx_set]
    has_dead_end = False
    candidates = []
    seen = set()
    for vid in seed_vids:
        ent_hash = g.vs[vid]["name"]
        pids = index.entity_to_passages.get(ent_hash, set())
        if len(pids) > 1:
            continue
        has_dead_end = True
        for pid in pids:
            for nek in passage_to_ents[pid]:
                if nek == ent_hash or nek in seen:
                    continue
                seen.add(nek)
                npids = index.entity_to_passages.get(nek, set())
                if len(npids) < 2:
                    continue
                nvid = index.node_name_to_idx.get(nek)
                if nvid is None or nvid in passage_idx_set:
                    continue
                name = g.vs[nvid]["content"]
                ppr = float(ppr_scores[nvid]) if ppr_scores is not None else 0
                score = ppr / math.sqrt(len(npids))
                candidates.append((name, len(npids), score))

    if not has_dead_end or not candidates:
        return []

    candidates.sort(key=lambda x: x[2], reverse=True)
    cand_list = candidates[:15]

    # Build prompt
    docs_text = ""
    for i, doc in enumerate(top_docs[:5]):
        docs_text += f"[Doc {i+1}] {doc[:200]}\n\n"

    cand_text = "\n".join(f"- {name} ({np_} passages)" for name, np_, _ in cand_list)

    traces_text = ""
    if previous_traces:
        traces_text = "\nPrevious reasoning:\n" + "\n".join(f"- {t[:100]}" for t in previous_traces[-2:])

    prompt = f"""Question: {original_query}
Current focus: {current_query}
{traces_text}
Retrieved documents:
{docs_text}
The following entities appear in retrieved documents and connect to multiple passages in the knowledge base:
{cand_text}

Which of these entities would help find the missing information to answer the question?
You may also suggest other entities not in this list.
Only select entities clearly relevant to the reasoning chain. Be selective.

Respond in JSON: {{"bridges": ["entity1", "entity2"], "reasoning": "brief"}}"""

    messages = [
        {"role": "system", "content": "Select bridge entities for multi-hop question answering. Be selective."},
        {"role": "user", "content": prompt},
    ]

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
        bridges = [str(e).lower().strip() for e in parsed.get("bridges", []) if e][:5]
        logger.info(f"  Dead-end LLM bridges: {bridges} | reasoning: {parsed.get('reasoning', '')[:80]}")
        return bridges
    except Exception as e:
        logger.warning(f"  Dead-end LLM bridge selection failed: {e}")
        return []


def _dead_end_cross_passage_seeds(index, base_node_weights: np.ndarray,
                                   ppr_scores: np.ndarray,
                                   original_query: str, current_query: str,
                                   accumulated_seeds: Dict = None,
                                   round_i: int = 1) -> Dict[str, Tuple[int, float, int]]:
    """For dead-end seed entities, find 1-hop cross-passage neighbors.
    Returns new seeds to ADD to accumulated_seeds: {name: (vid, weight, disc_round)}.
    - Query-filtered: top-1 new per round (cumulative across rounds)
    - Explore: 1 random per round (re-sampled, uses round_i as random seed)
    """
    passage_idx_set = set(index.passage_node_idxs)
    g = index.graph

    passage_to_ents = defaultdict(set)
    for ek, pids in index.entity_to_passages.items():
        for pid in pids:
            passage_to_ents[pid].add(ek)

    q_lower = original_query.lower() + " " + current_query.lower()

    # Find dead-end seeds
    seed_vids = [v for v in range(len(base_node_weights))
                 if base_node_weights[v] > 0 and v not in passage_idx_set]

    candidates = []
    unfiltered_candidates = []
    seen = set()
    for vid in seed_vids:
        ent_hash = g.vs[vid]["name"]
        pids = index.entity_to_passages.get(ent_hash, set())
        if len(pids) > 1:
            continue
        for pid in pids:
            for nek in passage_to_ents[pid]:
                if nek == ent_hash or nek in seen:
                    continue
                seen.add(nek)
                npids = index.entity_to_passages.get(nek, set())
                if len(npids) < 2:
                    continue
                nvid = index.node_name_to_idx.get(nek)
                if nvid is None or nvid in passage_idx_set:
                    continue
                name = g.vs[nvid]["content"]
                ppr = float(ppr_scores[nvid]) if ppr_scores is not None else 0
                score = ppr / math.sqrt(len(npids))
                unfiltered_candidates.append((nvid, score, name, len(npids)))
                if name.lower() in q_lower:
                    candidates.append((nvid, score, name, len(npids)))

    if accumulated_seeds is None:
        accumulated_seeds = {}
    already_vids = set(vid for _, (vid, _, _) in accumulated_seeds.items())

    new_seeds = {}

    # Query-filtered: progressive — round_i new seeds per round
    candidates.sort(key=lambda x: x[1], reverse=True)
    added = 0
    for nvid, score, name, np_ in candidates:
        if nvid not in already_vids and added < round_i:
            weight = DEFAULT_ENTITY_SEED_WEIGHT * 0.5  # 0.25
            new_seeds[name] = (nvid, weight, round_i)
            logger.info(f"    Dead-end cross-passage seed: '{name}' (ppr/{np_}p score={score:.6f}) w={weight:.4f}")
            added += 1

    return new_seeds


def _get_existing_seed_ids(index, base_node_weights: np.ndarray) -> List[int]:
    """Extract entity vertex IDs that have non-zero weight in base PPR seeds."""
    passage_idx_set = set(index.passage_node_idxs)
    seed_ids = []
    for vid in range(len(base_node_weights)):
        if base_node_weights[vid] > 0 and vid not in passage_idx_set:
            seed_ids.append(vid)
    return seed_ids


def _llm_filter_passages(llm_client, query: str, bridges: List[str], candidates: List[Tuple[int, str]]) -> List[int]:
    """Use LLM to filter which candidate passages are truly relevant to the query given the bridge entities.
    candidates: list of (p_vid, passage_content)
    Returns: list of p_vid that are relevant
    """
    if not candidates or llm_client is None:
        return [p_vid for p_vid, _ in candidates]
    try:
        passages_text = ""
        for i, (_, content) in enumerate(candidates):
            passages_text += f"[{i+1}] {content[:300]}\n\n"
        bridge_str = ", ".join(bridges[:5])
        prompt = f"""Given a query and some bridge entities discovered during multi-hop reasoning, identify which candidate passages are TRULY relevant to answering the query.

Query: {query}
Bridge entities: {bridge_str}

Candidate passages:
{passages_text}

Return a JSON array of passage numbers that are likely relevant to the query. Include only passages that could actually help answer the query.
Output format: {{"relevant": [1, 3]}}"""
        messages = [{"role": "user", "content": prompt}]
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
        import json as _json
        parsed = _json.loads(text)
        relevant_idxs = parsed.get("relevant", [])
        keep_vids = [candidates[i-1][0] for i in relevant_idxs if 1 <= i <= len(candidates)]
        return keep_vids
    except Exception as e:
        logger.warning(f"LLM passage filter failed: {e}, keeping all candidates")
        return [p_vid for p_vid, _ in candidates]


def _mini_ppr_select_seeds(
    index,
    discovered_vertex_ids: Dict[str, int],
    existing_seed_vertex_ids: List[int],
    round0_top_doc_ids: List[int] = None,
    graph=None,
    gold_docs: List[str] = None,
    llm_client=None,
    query: str = None,
) -> List[Tuple[int, int, float]]:
    """Run mini-PPR from bridge entities on local 5-hop subgraph.

    discovered_vertex_ids: {name: (vid, weight)} where weight incorporates decay.
    Returns list of (bridge_vid, seed_vid, ppr_score) for selected connections.
    """
    graph = graph if graph is not None else index.graph
    passage_idx_set = set(index.passage_node_idxs)
    _include_passages = os.getenv("MINI_PPR_INCLUDE_PASSAGES", "0") == "1"

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
                # When include_passages, allow passage nodes in subgraph
                if _include_passages or n not in passage_idx_set:
                    frontier.add(n)
        subgraph_vids |= frontier

    subgraph_vids = sorted(subgraph_vids)
    if len(subgraph_vids) < 2:
        return []

    sub = graph.subgraph(subgraph_vids)
    vid_to_sub = {vid: i for i, vid in enumerate(subgraph_vids)}
    n_sub = len(subgraph_vids)

    # All bridge entities as seeds, weighted by decay
    reset = np.zeros(n_sub)
    for d_vid, d_weight in discovered_vertex_ids.values():
        if d_vid in vid_to_sub:
            reset[vid_to_sub[d_vid]] = d_weight
    if reset.sum() == 0:
        return []
    reset /= reset.sum()

    _mini_algo = os.getenv("MAIN_DIFFUSION_ALGO", "ppr")
    if _mini_algo == "katz":
        from scipy.sparse import diags as sparse_diags
        _alpha = float(os.getenv("KATZ_ALPHA", "0.2"))
        _K = int(os.getenv("KATZ_ITERS", "10"))
        adj_sparse = sub.get_adjacency_sparse(attribute='weight' if 'weight' in sub.es.attributes() else None).astype(float)
        row_sums = np.array(adj_sparse.sum(axis=1)).flatten()
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        adj_norm = sparse_diags(1.0 / row_sums) @ adj_sparse
        score = np.zeros(n_sub)
        current = reset.copy()
        for _ in range(_K):
            current = _alpha * (adj_norm @ current)
            score += current
        ppr_scores = score
    else:
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
    bridge_vids = set(vid for vid, w in discovered_vertex_ids.values())
    seed_scores = []
    for s_vid in existing_seed_vertex_ids:
        if s_vid not in bridge_vids and s_vid in vid_to_sub:
            score = ppr_scores[vid_to_sub[s_vid]]
            seed_scores.append((s_vid, score))
    seed_scores.sort(key=lambda x: x[1], reverse=True)

    log_items = [(index.graph.vs[sv]["content"][:30], f"{sc:.6f}") for sv, sc in seed_scores[:8]]
    logger.info(f"  Mini-PPR(subgraph, joint) -> seeds: {log_items}")

    # Log gold doc PPR scores (for analysis)
    if gold_docs is not None:
        gold_set = set(gold_docs)
        gold_info = []
        for p_vid in passage_idx_set:
            content = index.graph.vs[p_vid]["content"]
            if content in gold_set:
                if p_vid in vid_to_sub:
                    gold_info.append((content[:30], f"{ppr_scores[vid_to_sub[p_vid]]:.6f}", "in_sub"))
                else:
                    gold_info.append((content[:30], "N/A", "out_sub"))
        if gold_info:
            logger.info(f"  [GOLD-SCORES] {gold_info}")

    # Bridge entity scores (they are the reset sources)
    bridge_scores = {}
    for d_vid in bridge_vids:
        if d_vid in vid_to_sub:
            bridge_scores[d_vid] = ppr_scores[vid_to_sub[d_vid]]

    # Collect raw joint scores
    raw_joints = []
    for s_vid, seed_score in seed_scores:
        if seed_score >= MINI_PPR_THRESHOLD:
            for d_vid in bridge_vids:
                b_score = bridge_scores.get(d_vid, 0.0)
                joint = b_score * seed_score
                raw_joints.append((d_vid, s_vid, seed_score, joint))

    # Normalize joint scores to [0.2, 1.0] if requested; otherwise use seed_score as 3rd element
    if raw_joints:
        max_j = max(j for _, _, _, j in raw_joints)
        min_j = min(j for _, _, _, j in raw_joints)
        for d_vid, s_vid, seed_score, joint in raw_joints:
            if max_j > min_j:
                norm = 0.2 + 0.8 * (joint - min_j) / (max_j - min_j)
            else:
                norm = 0.6
            selected.append((d_vid, s_vid, seed_score, norm))

    # If include_passages, also collect bridge↔passage overlay edges
    if _include_passages:
        passage_threshold = float(os.getenv("MINI_PPR_PASSAGE_THRESHOLD", str(MINI_PPR_THRESHOLD)))
        passage_topk = int(os.getenv("MINI_PPR_PASSAGE_TOPK", "2"))
        passage_scores = []
        for p_vid in passage_idx_set:
            if p_vid in vid_to_sub:
                score = ppr_scores[vid_to_sub[p_vid]]
                if score >= passage_threshold:
                    passage_scores.append((p_vid, score))
        passage_scores.sort(key=lambda x: x[1], reverse=True)
        # Cap to top-K after threshold filtering
        passage_scores = passage_scores[:passage_topk]
        # Log top passages
        top_p = [(index.graph.vs[pv]["content"][:30], f"{sc:.6f}") for pv, sc in passage_scores]
        logger.info(f"  Mini-PPR passages (threshold={passage_threshold}, top{passage_topk}): {top_p}")

        # Optional LLM filter: keep only passages the LLM judges as relevant
        if os.getenv("LLM_PASSAGE_FILTER", "0") == "1" and passage_scores and llm_client is not None and query is not None:
            bridge_names = [index.graph.vs[d_vid]["content"][:40] for d_vid in bridge_vids if d_vid in vid_to_sub]
            candidates = [(pv, index.graph.vs[pv]["content"]) for pv, _ in passage_scores]
            keep_vids = _llm_filter_passages(llm_client, query, bridge_names, candidates)
            keep_set = set(keep_vids)
            before_n = len(passage_scores)
            passage_scores = [(pv, sc) for pv, sc in passage_scores if pv in keep_set]
            logger.info(f"  LLM filter: {before_n} -> {len(passage_scores)} passages")

        # Build passage overlay edges: bridge -> passage
        if passage_scores:
            p_max = max(sc for _, sc in passage_scores)
            p_min = min(sc for _, sc in passage_scores)
            for p_vid, p_score in passage_scores:
                if p_max > p_min:
                    p_norm = 0.2 + 0.8 * (p_score - p_min) / (p_max - p_min)
                else:
                    p_norm = 0.6
                for d_vid in bridge_vids:
                    b_score = bridge_scores.get(d_vid, 0.0)
                    if b_score > 0:
                        # Passage overlay: 4th element is normalized score
                        selected.append((d_vid, p_vid, p_score, p_norm))

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


def _run_ppr(index, reset_prob: np.ndarray, damping: float = 0.5, graph=None,
             return_all_scores: bool = False):
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
    if return_all_scores:
        return sorted_doc_ids, sorted_doc_scores, np.array(ppr_scores)
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


def _ppr_soft_reweight(index, ppr_scores):
    """Reweight edges using symmetric-normalized PPR activation, interpolated with original.

    factor = ppr[u] × ppr[v] / sqrt(degree(u) × degree(v))   (symmetric normalization)
    new_weight = (1 - β) × old_weight + β × old_weight × normalized_factor

    β controls how much PPR signal mixes in (default 0.3 = conservative).
    Symmetric normalization prevents hub edges from being over-suppressed.
    """
    beta = float(os.getenv("PPR_REWEIGHT_BETA", "0.3"))
    g = index.graph.copy()
    has_weight = 'weight' in g.es.attributes()
    if not has_weight:
        g.es['weight'] = [1.0] * g.ecount()

    degrees = np.array(g.degree(), dtype=float)
    degrees = np.maximum(degrees, 1.0)  # avoid div by zero

    # Compute symmetric-normalized PPR factor for each edge
    factors = np.zeros(g.ecount())
    for eid in range(g.ecount()):
        src, tgt = g.es[eid].source, g.es[eid].target
        factors[eid] = ppr_scores[src] * ppr_scores[tgt] / math.sqrt(degrees[src] * degrees[tgt])

    # Normalize to [0, 1] range
    max_f = factors.max() if len(factors) > 0 else 1.0
    if max_f > 0:
        factors /= max_f

    # Interpolate: keep most of original, nudge by PPR signal
    n_nudged = 0
    for eid in range(g.ecount()):
        old_w = g.es[eid]['weight']
        g.es[eid]['weight'] = (1 - beta) * old_w + beta * old_w * factors[eid]
        if factors[eid] < 0.1:
            n_nudged += 1

    logger.info(f"  PPR-soft-reweight (β={beta}): {n_nudged}/{g.ecount()} edges nudged down (factor<0.1)")
    return g


def _compute_hub_danger(index):
    """Precompute hub danger score for each entity node.

    danger = n_passages × log(1 + entity_degree)
    where n_passages = number of passage neighbors,
          entity_degree = degree minus passage neighbors.
    Returns danger array and percentile ranks (0-1).
    """
    graph = index.graph
    passage_idx_set = set(index.passage_node_idxs)
    n = graph.vcount()
    danger = np.zeros(n)
    for v in range(n):
        if v in passage_idx_set:
            continue
        neighbors = graph.neighbors(v)
        n_passages = sum(1 for nb in neighbors if nb in passage_idx_set)
        entity_degree = len(neighbors) - n_passages
        danger[v] = n_passages * math.log(1 + entity_degree)
    # Percentile rank among entity nodes only
    entity_vids = [v for v in range(n) if v not in passage_idx_set]
    entity_dangers = danger[entity_vids]
    from scipy.stats import rankdata
    ranks = rankdata(entity_dangers, method='average')
    pct = ranks / len(ranks)  # 0-1
    danger_pct = np.zeros(n)
    for i, v in enumerate(entity_vids):
        danger_pct[v] = pct[i]
    return danger, danger_pct


def _prune_graph_by_ppr(index, ppr_scores, danger_pct, keep_frac, alpha):
    """Prune entity nodes by zeroing edges to low-PPR high-danger nodes.

    adjusted_score = ppr_score / (1 + alpha × danger_percentile)
    Bottom (1-keep_frac) entity nodes by adjusted_score get all edges zeroed.
    Returns a graph copy with same vertex numbering (safe for index.passage_node_idxs).
    """
    graph = index.graph
    passage_idx_set = set(index.passage_node_idxs)
    entity_vids = [v for v in range(graph.vcount()) if v not in passage_idx_set]
    adjusted = []
    for v in entity_vids:
        adj = ppr_scores[v] / (1 + alpha * danger_pct[v])
        adjusted.append((v, adj))
    adjusted.sort(key=lambda x: x[1], reverse=True)
    n_keep = max(1, int(len(entity_vids) * keep_frac))
    prune_set = set(v for v, _ in adjusted[n_keep:])

    # Copy graph and zero out edges touching pruned nodes
    g = graph.copy()
    has_weight = 'weight' in g.es.attributes()
    if not has_weight:
        g.es['weight'] = [1.0] * g.ecount()
    zero_count = 0
    for eid in range(g.ecount()):
        src, tgt = g.es[eid].source, g.es[eid].target
        if src in prune_set or tgt in prune_set:
            g.es[eid]['weight'] = 0.0
            zero_count += 1
    n_pruned = len(prune_set)
    logger.info(f"  Hub-prune: disconnected {n_pruned}/{len(entity_vids)} entity nodes "
                f"({n_pruned/len(entity_vids)*100:.1f}%), zeroed {zero_count} edges")
    pruned_sample = [(v, danger_pct[v], ppr_scores[v]) for v, _ in adjusted[n_keep:n_keep+5]]
    if pruned_sample:
        items = [(graph.vs[v]['content'][:25], f"danger={dp:.2f} ppr={pp:.2e}") for v, dp, pp in pruned_sample]
        logger.info(f"  Sample pruned: {items}")
    return g, None, None


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


def _build_focused_subgraph(index, top_doc_ids: List[int], n_hops: int = 2):
    """Build a focused subgraph from top-K passages' entities + n-hop neighbors.
    Returns the subgraph as a copy of the full graph with non-included edges removed."""
    import igraph as ig
    graph = index.graph
    passage_set = set(index.passage_node_idxs)

    # Collect seed entities from top passages
    core_vids = set()
    passage_vids = set()
    for doc_id in top_doc_ids:
        if doc_id < len(index.passage_node_idxs):
            p_vid = index.passage_node_idxs[doc_id]
            passage_vids.add(p_vid)
            for n in graph.neighbors(p_vid):
                if n not in passage_set:
                    core_vids.add(n)

    # Expand n-hops from core entities (entity-entity only)
    subgraph_vids = set(core_vids)
    for _ in range(n_hops):
        frontier = set()
        for vid in subgraph_vids:
            for n in graph.neighbors(vid):
                if n not in passage_set:
                    frontier.add(n)
        subgraph_vids |= frontier

    # Add ALL passages connected to included entities (don't break entity-passage links)
    for vid in list(subgraph_vids):
        for n in graph.neighbors(vid):
            if n in passage_set:
                passage_vids.add(n)

    keep_vids = subgraph_vids | passage_vids
    logger.info(f"  [PRUNE] Focused subgraph: {len(core_vids)} seed entities, "
                f"{n_hops}-hop -> {len(subgraph_vids)} entities, {len(passage_vids)} passages, "
                f"total {len(keep_vids)}/{graph.vcount()} nodes ({len(keep_vids)*100/graph.vcount():.1f}%)")

    # Build subgraph: prune EE edges, but keep cross-passage bridge edges
    # An entity is a "cross-passage bridge" if it connects to 2+ passages
    cross_passage_entities = set()
    for vid in range(graph.vcount()):
        if vid in passage_set:
            continue
        passage_neighbors = [n for n in graph.neighbors(vid) if n in passage_set]
        if len(passage_neighbors) >= 2:
            cross_passage_entities.add(vid)

    focused = graph.copy()
    remove_eids = []
    for e in focused.es:
        src, tgt = e.source, e.target
        # Keep all entity-passage edges
        if src in passage_set or tgt in passage_set:
            continue
        # Keep edge if both endpoints are cross-passage bridge entities
        if src in cross_passage_entities and tgt in cross_passage_entities:
            continue
        # Keep edge if both endpoints in focused area
        if src in keep_vids and tgt in keep_vids:
            continue
        # Remove: outside focused area AND not a cross-passage bridge
        remove_eids.append(e.index)
    focused.delete_edges(remove_eids)
    logger.info(f"  [PRUNE] Removed {len(remove_eids)} EE edges "
                f"(kept {len(cross_passage_entities)} cross-passage bridges + all EP), "
                f"remaining {focused.ecount()}")
    return focused, keep_vids


def _collect_bridge_neighborhood_edges(full_graph, bridge_vids: List[int], n_hops: int = 2, passage_set: set = None):
    """Collect edges from bridge entities' n-hop neighborhood as temp_edges.
    Returns list of (src, dst, weight) tuples to add as overlay."""
    if passage_set is None:
        passage_set = set()

    # Get bridge's n-hop entity neighborhood from FULL graph
    new_vids = set(bridge_vids)
    expand_front = set(bridge_vids)
    for _ in range(n_hops):
        frontier = set()
        for vid in expand_front:
            for n in full_graph.neighbors(vid):
                if n not in passage_set:
                    frontier.add(n)
        expand_front = frontier - new_vids
        new_vids |= frontier

    # Collect all edges within this neighborhood (entity-entity + entity-passage)
    edges = []
    seen = set()
    for vid in new_vids:
        for n in full_graph.neighbors(vid):
            pair = (min(vid, n), max(vid, n))
            if pair not in seen:
                seen.add(pair)
                eid = full_graph.get_eid(vid, n, error=False)
                if eid != -1:
                    edges.append((vid, n, full_graph.es[eid]['weight']))

    logger.info(f"  [EXPAND] Bridge neighborhood: {len(new_vids)} entities, "
                f"{len(edges)} edges from {len(bridge_vids)} bridges ({n_hops}-hop)")
    return edges


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
    bridge_type_map = {}

    # Bridge-first targeted decompose: after Round 0 bridge discovery,
    # generate sub-queries for missing information, each independently retrieves
    _decompose = os.getenv("DECOMPOSE_AGENT", "0") == "1"
    base_node_weights = None
    round0_top_doc_ids = None
    per_round_seeds = []  # List of (normed_seed_vector, round_weight) per round
    per_round_ppr_docs = []  # List of doc score arrays per round for PPR fusion
    accumulated_docs = []  # Accumulated unique docs across rounds for stable reasoning
    accumulated_de_seeds = {}  # Dead-end cross-passage seeds accumulated across rounds
    pruned_graph = None
    focused_graph = None  # For PRUNE_AND_EXPAND mode
    bridge_neighborhood_edges = []  # Accumulated bridge neighborhood edges across rounds
    accumulated_seed_origin = {}  # {vid: origin_round} for CUMULATIVE_MINI_PPR_SEEDS ablation
    hub_danger_pct = None
    _prune_expand = os.getenv("PRUNE_AND_EXPAND", "0") == "1"
    _prune_hops = int(os.getenv("PRUNE_HOPS", "2"))
    _expand_hops = int(os.getenv("EXPAND_HOPS", "2"))
    _prune_top_k = int(os.getenv("PRUNE_TOP_K", "10"))
    if HUB_PRUNE:
        _, hub_danger_pct = _compute_hub_danger(index)

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
        # DISABLE_BRIDGES=1 ablation: don't inject bridges as PPR seeds (text-only retrieval)
        _disable_bridges = os.getenv("DISABLE_BRIDGES", "0") == "1"
        if all_discovered and base_node_weights is not None and not _disable_bridges:
            # Compute query-pair sim for all bridge entities (sentence-level relevance)
            query_sims = _get_bridge_query_sims(index, all_discovered, retrieve_query)

            # Build vid -> (query_sim, decay_factor) lookup
            vid_to_info = {}
            discovered_with_decay = {}
            for name, (vid, res_sim, disc_round) in all_discovered.items():
                decay = SEED_DECAY ** (round_i - disc_round)
                q_sim = query_sims.get(vid, 1.0)
                vid_to_info[vid] = (q_sim, decay)
                discovered_with_decay[name] = (vid, decay)

            _cumulative_seeds = os.getenv("CUMULATIVE_MINI_PPR_SEEDS", "0") == "1"
            if _cumulative_seeds:
                # Add current round's pair seeds to accumulated dict (origin = round_i)
                _curr_pair_seeds = index._compute_node_weights(retrieve_query)
                _passage_idx_set = set(index.passage_node_idxs)
                for _vid in range(len(_curr_pair_seeds)):
                    if _curr_pair_seeds[_vid] > 0 and _vid not in _passage_idx_set:
                        if _vid not in accumulated_seed_origin:
                            accumulated_seed_origin[_vid] = round_i
                existing_seeds = list(accumulated_seed_origin.keys())
                logger.info(f"  CUMULATIVE_SEEDS: {len(existing_seeds)} accumulated seeds across rounds")
            else:
                existing_seeds = _get_existing_seed_ids(index, base_node_weights)
            _direct_overlay = os.getenv("DIRECT_OVERLAY", "0") == "1"
            _bfs_overlay = os.getenv("BFS_OVERLAY", "0") == "1"
            if _direct_overlay:
                # Direct connect: every bridge to every existing seed, no mini-PPR
                bridge_vids = set(vid for vid, w in discovered_with_decay.values())
                seed_set = set(existing_seeds) - bridge_vids
                ppr_connections = []
                for d_vid in bridge_vids:
                    for s_vid in seed_set:
                        ppr_connections.append((d_vid, s_vid, 1.0, 1.0))
                logger.info(f"  Direct overlay: {len(ppr_connections)} connections ({len(bridge_vids)} bridges x {len(seed_set)} seeds)")
            elif _bfs_overlay:
                # Simple 5-hop BFS reachability: connect bridge to any seed within 5 hops
                passage_idx_set = set(index.passage_node_idxs)
                bridge_vids = set(vid for vid, w in discovered_with_decay.values())
                seed_set = set(existing_seeds) - bridge_vids
                ppr_connections = []
                for d_vid in bridge_vids:
                    # BFS from bridge, up to 5 hops, skip passage nodes
                    visited = {d_vid}
                    frontier = {d_vid}
                    for hop in range(5):
                        next_frontier = set()
                        for v in frontier:
                            for n in index.graph.neighbors(v):
                                if n not in visited and n not in passage_idx_set:
                                    next_frontier.add(n)
                                    visited.add(n)
                        frontier = next_frontier
                        if not frontier:
                            break
                    # Connect to any seed found within 5 hops
                    reachable_seeds = visited & seed_set
                    for s_vid in reachable_seeds:
                        ppr_connections.append((d_vid, s_vid, 1.0, 1.0))
                logger.info(f"  BFS overlay: {len(ppr_connections)} connections from {len(bridge_vids)} bridges")
            else:
                ppr_connections = _mini_ppr_select_seeds(
                    index,
                    discovered_vertex_ids=discovered_with_decay,
                    existing_seed_vertex_ids=existing_seeds,
                    round0_top_doc_ids=round0_top_doc_ids,
                    gold_docs=gold_docs,
                    llm_client=llm_client,
                    query=retrieve_query,
                )

            _edge_mode = os.getenv("EDGE_WEIGHT_MODE", "degree")
            _disable_overlay = os.getenv("DISABLE_OVERLAY", "0") == "1"
            _disable_bridge_edges = os.getenv("DISABLE_BRIDGE_EDGES", "0") == "1"
            _disable_inter_edges = os.getenv("DISABLE_INTER_EDGES", "0") == "1"
            if not _disable_overlay and not _disable_bridge_edges:
                for d_vid, s_vid, seed_score, joint_norm in ppr_connections:
                    d_qsim, d_decay = vid_to_info.get(d_vid, (1.0, 1.0))
                    _overlay_w = float(os.getenv("OVERLAY_EDGE_WEIGHT", "2.0"))
                    if _cumulative_seeds:
                        _s_origin = accumulated_seed_origin.get(s_vid, 0)
                        if os.getenv("CUMULATIVE_R0_ANCHOR", "0") == "1" and _s_origin == 0:
                            _s_decay = 1.0  # R0 seed always full weight (anchor)
                        else:
                            _cum_decay = float(os.getenv("CUMULATIVE_SEED_DECAY", str(SEED_DECAY)))
                            _s_decay = _cum_decay ** (round_i - _s_origin)
                    else:
                        _s_decay = 1.0
                    bridge_weight = _overlay_w * d_decay * _s_decay
                    temp_edges.append((d_vid, s_vid, bridge_weight))

            # Inter-discovered edges: only connect entities from the same round
            _inter_ppr = os.getenv("INTER_PPR", "0") == "1"
            if not _disable_overlay and not _disable_inter_edges:
                if _inter_ppr:
                    # Use mini-PPR to select which inter-bridge edges to build
                    by_round = {}
                    for name, (vid, res_sim, disc_round) in all_discovered.items():
                        by_round.setdefault(disc_round, []).append((name, vid, disc_round))
                    for dr, bridges_in_round in by_round.items():
                        if len(bridges_in_round) < 2:
                            continue
                        decay = SEED_DECAY ** (round_i - dr)
                        # For each bridge, run mini-PPR and check if other bridges have high score
                        for i, (n1, v1, _) in enumerate(bridges_in_round):
                            for n2, v2, _ in bridges_in_round[i+1:]:
                                # Check if v2 is reachable from v1 via mini-PPR threshold
                                reset_single = np.zeros(index.graph.vcount())
                                reset_single[v1] = 1.0
                                ppr_single = index.graph.personalized_pagerank(
                                    vertices=[v2], damping=0.5, directed=False,
                                    weights='weight' if 'weight' in index.graph.es.attributes() else None,
                                    reset=reset_single, implementation='prpack')
                                if ppr_single[0] >= MINI_PPR_THRESHOLD:
                                    q_sim1 = query_sims.get(v1, 1.0)
                                    q_sim2 = query_sims.get(v2, 1.0)
                                    w = max(_degree_adaptive_weight(index, v1, 1.0, sim=q_sim1),
                                            _degree_adaptive_weight(index, v2, 1.0, sim=q_sim2)) * decay
                                    temp_edges.append((v1, v2, w))
                    logger.info(f"  Inter-PPR: filtered inter-bridge edges")
                else:
                    _cross_round_inter = os.getenv("CROSS_ROUND_INTER", "0") == "1"
                    if _cross_round_inter:
                        all_bridges = []
                        for name, (vid, res_sim, disc_round) in all_discovered.items():
                            q_sim = query_sims.get(vid, 1.0)
                            all_bridges.append((vid, q_sim, disc_round))
                        for i, (v1, s1, dr1) in enumerate(all_bridges):
                            for v2, s2, dr2 in all_bridges[i + 1:]:
                                avg_dr = (dr1 + dr2) / 2.0
                                decay = SEED_DECAY ** (round_i - avg_dr)
                                w = max(_degree_adaptive_weight(index, v1, 1.0, sim=s1),
                                        _degree_adaptive_weight(index, v2, 1.0, sim=s2)) * decay
                                temp_edges.append((v1, v2, w))
                    else:
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

            # Seed weights with decay (using query-pair sim)
            _hierarchical_mode = os.getenv("HIERARCHICAL_BRIDGES", "0") == "1"
            _general_weight = float(os.getenv("GENERAL_BRIDGE_WEIGHT", "0.3"))
            extra_node_weights = np.zeros(index.graph.vcount())
            for name, (vid, res_sim, disc_round) in all_discovered.items():
                decay = SEED_DECAY ** (round_i - disc_round)
                q_sim = query_sims.get(vid, 1.0)
                base_w = DEFAULT_ENTITY_SEED_WEIGHT
                # Hierarchical: general bridges get lower weight
                if _hierarchical_mode and name in bridge_type_map and bridge_type_map[name] == "general":
                    base_w *= _general_weight
                # Decompose entities get lower weight
                if name in bridge_type_map and bridge_type_map[name] == "decompose":
                    base_w *= float(os.getenv("DECOMPOSE_WEIGHT", "0.3"))
                extra_node_weights[vid] += _degree_adaptive_weight(index, vid, base_w, sim=q_sim) * decay

            # Synonym merge: for each bridge entity, find near-synonym entities and build temp edges
            _syn_merge = os.getenv("SYNONYM_MERGE", "0") == "1"
            _syn_merge_threshold = float(os.getenv("SYNONYM_MERGE_THRESHOLD", "0.7"))
            _syn_merge_topk = int(os.getenv("SYNONYM_MERGE_TOPK", "3"))
            syn_merge_count = 0
            if _syn_merge and index.entity_embeddings is not None and len(index.entity_embeddings) > 0:
                _e_norms = np.linalg.norm(index.entity_embeddings, axis=1, keepdims=True)
                _e_norms = np.where(_e_norms == 0, 1, _e_norms)
                _normed_ents = index.entity_embeddings / _e_norms
                for name, (vid, res_sim, disc_round) in all_discovered.items():
                    if vid >= len(index.entity_keys):
                        continue
                    bridge_emb = _normed_ents[vid:vid+1]
                    sims = (bridge_emb @ _normed_ents.T).flatten()
                    sims[vid] = 0  # exclude self
                    top_syns = np.argsort(sims)[::-1][:_syn_merge_topk]
                    decay = SEED_DECAY ** (round_i - disc_round)
                    for syn_vid in top_syns:
                        if sims[syn_vid] < _syn_merge_threshold:
                            break
                        # Only merge if synonym connects to different passages
                        bridge_ek = index.entity_keys[vid]
                        syn_ek = index.entity_keys[syn_vid]
                        bridge_passages = index.entity_to_passages.get(bridge_ek, set())
                        syn_passages = index.entity_to_passages.get(syn_ek, set())
                        if syn_passages - bridge_passages:
                            w = sims[syn_vid] * decay
                            temp_edges.append((vid, int(syn_vid), w))
                            extra_node_weights[int(syn_vid)] += extra_node_weights[vid] * sims[syn_vid] * 0.5
                            syn_merge_count += 1
                if syn_merge_count > 0:
                    logger.info(f"  Synonym merge: {syn_merge_count} edges added (threshold={_syn_merge_threshold})")

            logger.info(f"  Overlay: {len(all_discovered)} bridge entities, {len(temp_edges)} temp edges")

        # Add accumulated bridge neighborhood edges (from previous rounds)
        if _prune_expand and bridge_neighborhood_edges:
            temp_edges.extend(bridge_neighborhood_edges)
            logger.info(f"  [EXPAND] Added {len(bridge_neighborhood_edges)} neighborhood edges to overlay")

        # Select base graph for this round's PPR
        if _prune_expand and focused_graph is not None and round_i > 0:
            # Prune-and-expand mode: use focused graph (which grows each round)
            base_g = focused_graph
        elif pruned_graph is not None and round_i > 0:
            base_g = pruned_graph
        else:
            base_g = index.graph
        working_graph = _build_overlay_graph(index, temp_edges, base_graph=base_g) if temp_edges else base_g

        # Lateral inhibition: suppress edges around Round 0 high-activation entities
        _inhibit = os.getenv("LATERAL_INHIBITION", "0") == "1"
        _inhibit_factor = float(os.getenv("INHIBIT_FACTOR", "0.3"))
        if _inhibit and round_i > 0 and round0_ppr_scores is not None:
            passage_set_local = set(index.passage_node_idxs)
            bridge_vids_set = set(vid for _, (vid, _, _) in all_discovered.items())
            # Find top-50 activated entities from Round 0 (excluding bridges and passages)
            entity_scores = [(vid, round0_ppr_scores[vid])
                             for vid in range(len(round0_ppr_scores))
                             if vid not in passage_set_local and vid not in bridge_vids_set
                             and round0_ppr_scores[vid] > 0]
            entity_scores.sort(key=lambda x: x[1], reverse=True)
            suppress_vids = set(vid for vid, _ in entity_scores[:50])
            # Reduce edge weights for suppressed entities in working graph
            suppressed = 0
            for e in working_graph.es:
                if e.source in suppress_vids or e.target in suppress_vids:
                    e['weight'] *= _inhibit_factor
                    suppressed += 1
            logger.info(f"  [INHIBIT] Suppressed {len(suppress_vids)} entities, {suppressed} edges (factor={_inhibit_factor})")

        # Passage aggregation: only on LAST round when all bridges are accumulated
        _passage_agg = os.getenv("PASSAGE_AGG", "0") == "1"
        _passage_agg_max_np = int(os.getenv("PASSAGE_AGG_MAX_NP", "20"))
        if _passage_agg and round_i == max_rounds - 1 and extra_node_weights is not None and extra_node_weights.sum() > 0:
            _agg = defaultdict(float)
            _n_skipped = 0
            for vid in range(len(index.entity_keys)):
                w = extra_node_weights[vid]
                if w <= 0:
                    continue
                ek = index.entity_keys[vid]
                passages = index.entity_to_passages.get(ek, set())
                np_ = len(passages)
                if np_ == 0:
                    continue
                if np_ > _passage_agg_max_np:
                    _n_skipped += 1
                    continue
                contrib = w / math.log(np_ + 1)
                for pk in passages:
                    _agg[pk] += contrib
            n_boosted = 0
            for pk, agg_val in _agg.items():
                pvid = index.node_name_to_idx.get(pk)
                if pvid is not None:
                    extra_node_weights[pvid] += math.log(1 + agg_val)
                    n_boosted += 1
            if n_boosted > 0:
                logger.info(f"  Passage aggregation (last round): {n_boosted} passages boosted, {_n_skipped} hubs skipped")

        # ── Multi-peak PPR: per-round PPR → cross-round max → short re-diffusion ──
        _multi_peak = os.getenv("MULTI_PEAK_PPR", "0") == "1"
        _mp_rediffuse_steps = int(os.getenv("MP_REDIFFUSE_STEPS", "2"))
        if _multi_peak and all_discovered and base_node_weights is not None and round_i > 0:
            n_nodes = working_graph.vcount()
            # Step 1: group bridges by discovery round, each group runs ONE PPR together
            by_round = {}
            for name, (vid, res_sim, disc_round) in all_discovered.items():
                by_round.setdefault(disc_round, []).append((name, vid, res_sim, disc_round))

            round_ppr_list = []
            for dr, bridges in by_round.items():
                # Build reset_prob: base seeds + all bridges from this round
                round_reset = base_node_weights.copy()
                round_temp = []
                for name, vid, res_sim, disc_round in bridges:
                    decay = SEED_DECAY ** (round_i - disc_round)
                    q_sim = query_sims.get(vid, 1.0) if 'query_sims' in dir() else 1.0
                    round_reset[vid] += _degree_adaptive_weight(index, vid, DEFAULT_ENTITY_SEED_WEIGHT, sim=q_sim) * decay
                    # Collect overlay edges for this round's bridges
                    if temp_edges:
                        for te in temp_edges:
                            if te[0] == vid or te[1] == vid:
                                round_temp.append(te)
                round_graph = _build_overlay_graph(index, round_temp, base_graph=index.graph) if round_temp else working_graph
                _, _, round_ppr = _run_ppr(index, reset_prob=round_reset, damping=0.5,
                                           graph=round_graph, return_all_scores=True)
                round_ppr_list.append(np.array(round_ppr))

            # Step 2: cross-round max (preserve multi-peak across rounds)
            max_ppr = np.zeros(n_nodes)
            for rp in round_ppr_list:
                max_ppr = np.maximum(max_ppr, rp)

            # Step 3: short re-diffusion (1-2 steps)
            if _mp_rediffuse_steps > 0:
                import scipy.sparse as sp
                adj = working_graph.get_adjacency_sparse(attribute='weight')
                row_sums = np.array(adj.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1
                T = sp.diags(1.0 / row_sums) @ adj
                teleport_alpha = 0.5
                v = max_ppr.copy()
                for _step in range(_mp_rediffuse_steps):
                    v = (1 - teleport_alpha) * (T @ v) + teleport_alpha * max_ppr
                max_ppr = v

            # Extract passage scores
            doc_scores = np.array([max_ppr[idx] for idx in index.passage_node_idxs])
            sorted_doc_ids = np.argsort(doc_scores)[::-1]
            sorted_doc_scores = doc_scores[sorted_doc_ids]
            current_node_weights = base_node_weights
            all_ppr_scores = max_ppr
            logger.info(f"  Multi-peak PPR: {len(round_ppr_list)} round-groups, {_mp_rediffuse_steps} re-diffuse steps")
        else:
            # ── Per-round norm seed accumulation ──
            _per_round_norm = os.getenv("PER_ROUND_NORM", "0") == "1"
            _prn_last_only = os.getenv("PRN_LAST_ONLY", "0") == "1"
            if _per_round_norm:
                # Compute this round's pair + bridge seeds together, then L1 normalize
                this_round_seeds = index._compute_node_weights(retrieve_query)
                if all_discovered:
                    for name, (vid, res_sim, disc_round) in all_discovered.items():
                        bridge_decay = SEED_DECAY ** (round_i - disc_round)
                        q_sim = query_sims.get(vid, 1.0) if 'query_sims' in dir() and query_sims else 1.0
                        this_round_seeds[vid] += _degree_adaptive_weight(index, vid, DEFAULT_ENTITY_SEED_WEIGHT, sim=q_sim) * bridge_decay
                s = this_round_seeds.sum()
                this_round_normed = this_round_seeds / s if s > 0 else this_round_seeds
                per_round_seeds.append((this_round_normed, round_i))

                # PRN_LAST_ONLY: earlier rounds use baseline, only last round uses PRN
                if _prn_last_only and round_i < max_rounds - 1:
                    sorted_doc_ids, sorted_doc_scores, current_node_weights, all_ppr_scores = index.retrieve(
                        retrieve_query, working_graph=working_graph,
                        extra_node_weights=extra_node_weights, return_node_weights=True, query_entities=None)
                    logger.info(f"  PRN last-only: R{round_i} baseline (saved seeds)")
                else:
                    # Accumulate with window and decay
                    _prn_decay = float(os.getenv("PRN_DECAY", "0.5"))
                    _prn_window = int(os.getenv("PRN_WINDOW", "0"))  # 0 = all rounds
                    seeds_to_use = per_round_seeds if _prn_window == 0 else per_round_seeds[-_prn_window:]
                    accumulated_reset = np.zeros(index.graph.vcount())
                    for normed_seeds, stored_round in seeds_to_use:
                        decay = _prn_decay ** (round_i - stored_round)
                        accumulated_reset += normed_seeds * decay
                    sorted_doc_ids, sorted_doc_scores, all_ppr_scores = _run_ppr(
                        index, reset_prob=accumulated_reset, damping=0.5,
                        graph=working_graph, return_all_scores=True)
                    current_node_weights = this_round_seeds
                    logger.info(f"  PRN: {len(seeds_to_use)} rounds (window={_prn_window}), {len(all_discovered)} bridges")
            else:
                sorted_doc_ids, sorted_doc_scores, current_node_weights, all_ppr_scores = index.retrieve(
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
            # CUMULATIVE_MINI_PPR_SEEDS: initialize accumulated dict with R0 pair seeds (origin=0)
            if os.getenv("CUMULATIVE_MINI_PPR_SEEDS", "0") == "1":
                _passage_idx_set = set(index.passage_node_idxs)
                for _vid in range(len(base_node_weights)):
                    if base_node_weights[_vid] > 0 and _vid not in _passage_idx_set:
                        accumulated_seed_origin[_vid] = 0
            # Save entity PPR scores for lateral inhibition in later rounds
            round0_ppr_scores = np.array(all_ppr_scores) if all_ppr_scores is not None else None
            # Prune-and-expand: build focused subgraph from top passages
            if _prune_expand:
                focused_graph, _keep_vids = _build_focused_subgraph(
                    index, [int(d) for d in sorted_doc_ids[:_prune_top_k]], n_hops=_prune_hops)
            # Hub-aware pruning: use PPR diffused scores (not seed weights)
            elif HUB_PRUNE and hub_danger_pct is not None:
                pruned_graph, _keep_vids, _vid_map = _prune_graph_by_ppr(
                    index, all_ppr_scores, hub_danger_pct,
                    HUB_PRUNE_KEEP, HUB_PRUNE_ALPHA)
            # Soft reweight: use round 0 PPR scores to reweight all edges
            if PPR_SOFT_REWEIGHT:
                pruned_graph = _ppr_soft_reweight(index, all_ppr_scores)
        elif MINI_PPR_USE_LAST_ROUND:
            # Ablation: refresh subgraph anchor docs to current round's top docs
            round0_top_doc_ids = [int(d) for d in sorted_doc_ids[:20]]

        # Current round PPR scores (RRF-normalized)
        current_rrf = np.zeros(num_docs)
        for rank, doc_id in enumerate(sorted_doc_ids):
            current_rrf[doc_id] = 1.0 / (rrf_k + rank + 1)

        # Final = history RRF (small weight) + current PPR (large weight)
        # Normalize history by number of past rounds to keep range comparable
        HISTORY_WEIGHT = 0.3
        CURRENT_WEIGHT = 1.0
        history_norm = rrf_scores / max(round_i, 1)
        combined_scores = HISTORY_WEIGHT * history_norm + CURRENT_WEIGHT * current_rrf

        # Accumulate current round into history for next round
        rrf_scores += current_rrf

        # Save per-round PPR doc scores for fusion
        _ppr_fusion = os.getenv("PPR_FUSION", "0") == "1"
        if _ppr_fusion:
            doc_scores_this_round = np.array([all_ppr_scores[idx_] for idx_ in index.passage_node_idxs]) if all_ppr_scores is not None else np.zeros(num_docs)
            per_round_ppr_docs.append(doc_scores_this_round)

        # PRN + RRF mode
        _prn_with_rrf = os.getenv("PRN_WITH_RRF", "0") == "1"
        if _ppr_fusion and round_i == max_rounds - 1 and len(per_round_ppr_docs) >= 2:
            # Weighted PPR fusion: 0.1*R0 + 0.3*R1 + 0.6*R2
            _fusion_weights = [float(x) for x in os.getenv("PPR_FUSION_WEIGHTS", "0.1,0.3,0.6").split(",")]
            fused = np.zeros(num_docs)
            for fi, fw in enumerate(_fusion_weights):
                if fi < len(per_round_ppr_docs):
                    fused += fw * per_round_ppr_docs[fi]
            final_sorted_ids = np.argsort(fused)[::-1]
            logger.info(f"  PPR fusion: {len(per_round_ppr_docs)} rounds, weights={_fusion_weights[:len(per_round_ppr_docs)]}")
        elif _per_round_norm and not _prn_with_rrf:
            final_sorted_ids = sorted_doc_ids
        else:
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
            # ── Distributed Exploration: multiple sub-agents explore, main agent selects ──
            _dist_explore = os.getenv("DISTRIBUTED_EXPLORE", "0") == "1"
            _last_stop_de = round_diagnostics[-1].get("stop", False) if round_diagnostics else False
            if _dist_explore and all_discovered and not _last_stop_de:
                try:
                    # Step 1: Generate 3 sub-questions (no traces — avoid echo chamber)
                    _bridges_list = list(all_discovered.keys())[:8]
                    _de_gen_prompt = f"""Question: {question}

Bridge entities found so far: {_bridges_list}
But the answer is still NOT fully found.

Generate 3 DIFFERENT focused sub-questions that each explore a DIFFERENT missing direction.
Each sub-question should target a different piece of missing information.
Do NOT repeat what's already known from the bridges.

Output JSON only:
{{"sub_questions": ["Q1", "Q2", "Q3"]}}"""

                    _de_gen_resp = llm_client.infer([{"role": "user", "content": _de_gen_prompt}]).strip()
                    if "```json" in _de_gen_resp: _de_gen_resp = _de_gen_resp.split("```json")[1].split("```")[0].strip()
                    elif "```" in _de_gen_resp: _de_gen_resp = _de_gen_resp.split("```")[1].split("```")[0].strip()
                    _sub_qs = json.loads(_de_gen_resp).get("sub_questions", [])[:3]

                    if len(_sub_qs) >= 2:
                        # Step 2: Each sub-agent independently runs PPR and summarizes
                        _sub_reports = []
                        for _sq_i, _sq in enumerate(_sub_qs):
                            try:
                                _sq_weights = index._compute_node_weights(_sq)
                                _sq_sorted, _ = _run_ppr(index, reset_prob=_sq_weights, damping=0.5)
                                _sq_top3 = [index.passages[index.passage_keys[_sq_sorted[j]]] for j in range(3)]

                                # Sub-agent reads docs and summarizes what it found + connection
                                _sq_docs_text = "\n".join([f"[Doc {j+1}] {d}" for j, d in enumerate(_sq_top3)])
                                _sq_sum_prompt = f"""Sub-question: {_sq}
Original question: {question}

Documents found:
{_sq_docs_text}

Summarize in 2-3 sentences:
1. What key information did you find?
2. How does this connect to the original question?

Output JSON only:
{{"summary": "2-3 sentence summary", "key_entities": ["entity1", "entity2"]}}"""
                                _sq_sum_resp = llm_client.infer([{"role": "user", "content": _sq_sum_prompt}]).strip()
                                if "```json" in _sq_sum_resp: _sq_sum_resp = _sq_sum_resp.split("```json")[1].split("```")[0].strip()
                                elif "```" in _sq_sum_resp: _sq_sum_resp = _sq_sum_resp.split("```")[1].split("```")[0].strip()
                                _sq_parsed = json.loads(_sq_sum_resp)
                                _sub_reports.append({
                                    "sub_question": _sq,
                                    "summary": _sq_parsed.get("summary", ""),
                                    "key_entities": _sq_parsed.get("key_entities", []),
                                    "doc_ids": [int(_sq_sorted[j]) for j in range(3)],
                                    "docs": _sq_top3,
                                })
                            except Exception as _sq_err:
                                logger.warning(f"  DistExplore sub-agent {_sq_i} error: {_sq_err}")

                        if _sub_reports:
                            # Step 3: Main agent selects (sees its own doc summaries, NO traces)
                            _main_doc_summaries = "\n".join([
                                f"[Your Doc {j+1}] {index.passages[index.passage_keys[sorted_doc_ids[j]]][:120]}"
                                for j in range(min(5, len(sorted_doc_ids)))
                            ])

                            _options_text = ""
                            for _oi, _rep in enumerate(_sub_reports):
                                _options_text += f"\n--- Option {_oi+1} ---\n"
                                _options_text += f"Sub-question: {_rep['sub_question']}\n"
                                _options_text += f"Summary: {_rep['summary']}\n"
                                _options_text += f"Key entities: {_rep['key_entities']}\n"

                            _judge_prompt = f"""Question: {question}

Your currently retrieved documents:
{_main_doc_summaries}

Three sub-agents explored different directions:
{_options_text}

Which option (if any) provides NEW information that helps answer the question?
- Pick the option that fills a MISSING piece not covered by your current documents.
- If none of them add useful new information, reject all.

Output JSON only:
{{"selected": 1, "reason": "brief"}}
or
{{"selected": 0, "reason": "none add useful info"}}"""

                            _judge_resp = llm_client.infer([{"role": "user", "content": _judge_prompt}]).strip()
                            if "```json" in _judge_resp: _judge_resp = _judge_resp.split("```json")[1].split("```")[0].strip()
                            elif "```" in _judge_resp: _judge_resp = _judge_resp.split("```")[1].split("```")[0].strip()
                            _judge_parsed = json.loads(_judge_resp)
                            _selected = int(_judge_parsed.get("selected", 0))
                            _judge_reason = _judge_parsed.get("reason", "")

                            if 1 <= _selected <= len(_sub_reports):
                                _chosen = _sub_reports[_selected - 1]
                                # Append top-2 docs from chosen sub-agent (don't replace anything)
                                _inject_docs = _chosen["docs"][:2]
                                top_docs = top_docs[:5] + _inject_docs  # 5+2=7 docs for QA
                                logger.info(f"  DistExplore: selected option {_selected} ({_chosen['sub_question'][:50]}), reason: {_judge_reason[:80]}")
                                round_diag["dist_explore"] = {
                                    "sub_questions": [r["sub_question"] for r in _sub_reports],
                                    "selected": _selected,
                                    "reason": _judge_reason,
                                    "injected_doc_ids": _chosen["doc_ids"][:2],
                                }
                            else:
                                logger.info(f"  DistExplore: rejected all, reason: {_judge_reason[:80]}")
                                round_diag["dist_explore"] = {
                                    "sub_questions": [r["sub_question"] for r in _sub_reports],
                                    "selected": 0,
                                    "reason": _judge_reason,
                                }
                except Exception as _de_err:
                    logger.warning(f"  DistExplore error: {_de_err}")

            # Targeted sub-question for missing hop (with main agent review)
            _missing_hop = os.getenv("MISSING_HOP", "0") == "1"
            # Only trigger if last round's reasoning said should_stop=False
            _last_stop = round_diagnostics[-1].get("stop", False) if round_diagnostics else False
            if _missing_hop and all_discovered and not _last_stop:
                try:
                    # Step 1: Generate sub-question (no docs, just question + bridges)
                    _mh_prompt = f"""Question: {question}

You have found these bridge entities so far: {list(all_discovered.keys())[:8]}
But the answer is still NOT found.

What specific piece of information is still MISSING?
Generate ONE focused sub-question to find the missing piece.

Output JSON only:
{{"sub_question": "focused question"}}"""

                    _mh_resp = llm_client.infer([{"role": "user", "content": _mh_prompt}]).strip()
                    if "```json" in _mh_resp: _mh_resp = _mh_resp.split("```json")[1].split("```")[0].strip()
                    elif "```" in _mh_resp: _mh_resp = _mh_resp.split("```")[1].split("```")[0].strip()
                    _sub_q = json.loads(_mh_resp).get("sub_question", "")

                    if _sub_q:
                        # Step 2: Independent PPR for sub-question
                        _sq_weights = index._compute_node_weights(_sub_q)
                        _sq_sorted, _ = _run_ppr(index, reset_prob=_sq_weights, damping=0.5)
                        _sq_docs_summary = "\n".join([f"[SubDoc {j+1}] {index.passages[index.passage_keys[_sq_sorted[j]]][:150]}..." for j in range(5)])

                        # Step 3: Extract entities from sub-docs
                        _sq_extract = f'Sub-question: {_sub_q}\nDocuments:\n{_sq_docs_summary}\nExtract key entities. Output JSON only:\n{{"key_entities": ["entity1"]}}'
                        _sq_resp = llm_client.infer([{"role": "user", "content": _sq_extract}]).strip()
                        if "```json" in _sq_resp: _sq_resp = _sq_resp.split("```json")[1].split("```")[0].strip()
                        elif "```" in _sq_resp: _sq_resp = _sq_resp.split("```")[1].split("```")[0].strip()
                        _sq_ents = [str(e).lower().strip() for e in json.loads(_sq_resp).get("key_entities", []) if isinstance(e, str)]

                        if _sq_ents:
                            # Step 4: Main agent reviews with its own top-5 docs context
                            _main_docs_text = "\n".join([f"[Doc {j+1}] {index.passages[index.passage_keys[sorted_doc_ids[j]]][:150]}..." for j in range(min(5, len(sorted_doc_ids)))])
                            # Sub-agent top-2 docs summary for review
                            _sq_top2_texts = [index.passages[index.passage_keys[_sq_sorted[j]]][:150] for j in range(2)]
                            _review_prompt = f"""Question: {question}

Your current bridges: {list(all_discovered.keys())[:8]}
Your retrieved documents:
{_main_docs_text}

A sub-agent searched for '{_sub_q}' and found:
  SubDoc 1: {_sq_top2_texts[0]}...
  SubDoc 2: {_sq_top2_texts[1]}...
Suggested entities: {_sq_ents}

Do these documents and entities help answer the original question?
Accept ONLY if they clearly fill a missing piece. Reject if redundant or unrelated.

Output JSON only:
{{"accept": true/false, "accepted_entities": ["entity1"], "reason": "brief"}}"""

                            _review_resp = llm_client.infer([{"role": "user", "content": _review_prompt}]).strip()
                            if "```json" in _review_resp: _review_resp = _review_resp.split("```json")[1].split("```")[0].strip()
                            elif "```" in _review_resp: _review_resp = _review_resp.split("```")[1].split("```")[0].strip()
                            _review_parsed = json.loads(_review_resp)

                            _mh_mode = os.getenv("MISSING_HOP_MODE", "ppr")  # "ppr" or "docs"

                            if _review_parsed.get("accept", False):
                                _accepted = [str(e).lower().strip() for e in _review_parsed.get("accepted_entities", []) if isinstance(e, str)]

                                if _mh_mode == "docs":
                                    # V2: directly add sub-agent's top-2 docs to base_top_docs
                                    _sq_full_docs = [index.passages[index.passage_keys[_sq_sorted[j]]] for j in range(2)]
                                    base_top_docs = base_top_docs[:5] + _sq_full_docs
                                    logger.info(f"  Missing-hop(docs): '{_sub_q[:40]}' → added 2 docs directly")
                                else:
                                    # V1: add entities to PPR seeds
                                    _mh_new = []
                                    for _ent in _accepted:
                                        _ek = compute_hash(_ent, prefix="entity-")
                                        _vid = index.node_name_to_idx.get(_ek)
                                        if _vid is not None and _ent not in all_discovered:
                                            all_discovered[_ent] = (_vid, 1.0, round_i)
                                            extra_node_weights[_vid] += _degree_adaptive_weight(index, _vid, DEFAULT_ENTITY_SEED_WEIGHT)
                                            _mh_new.append(_ent)
                                    if _mh_new:
                                        working_graph_mh = _build_overlay_graph(index, temp_edges, base_graph=index.graph) if temp_edges else index.graph
                                        sorted_doc_ids, sorted_doc_scores, current_node_weights, all_ppr_scores = index.retrieve(
                                            retrieve_query, working_graph=working_graph_mh,
                                            extra_node_weights=extra_node_weights, return_node_weights=True, query_entities=None)
                                        base_top_docs = [index.passages[index.passage_keys[idx]] for idx in sorted_doc_ids[:10]]
                                        logger.info(f"  Missing-hop(ppr): '{_sub_q[:40]}' → seeds: {_mh_new[:3]}")
                            else:
                                logger.info(f"  Missing-hop: '{_sub_q[:40]}' → rejected")
                except Exception as _mh_err:
                    logger.warning(f"  Missing-hop error: {_mh_err}")

            # Explorer+Judge: look at rank 6-20 for missed bridge entities
            _explore_judge = os.getenv("EXPLORE_JUDGE", "0") == "1"
            if _explore_judge and all_discovered:
                try:
                    # Top-5 doc summaries (so explorer knows what's covered)
                    _top5_summary = ", ".join([index.passages[index.passage_keys[sorted_doc_ids[j]]][:60] for j in range(min(5, len(sorted_doc_ids)))])
                    # Rank 6-20 full docs
                    _docs_620 = "\n".join([f"[Doc {j-4}] {index.passages[index.passage_keys[sorted_doc_ids[j]]]}" for j in range(5, min(20, len(sorted_doc_ids)))])

                    _explorer_prompt = f"""You are exploring lower-ranked documents for a multi-hop question. Top-5 documents already cover these directions: {_top5_summary}...

Question: {question}
Bridge entities already found: {list(all_discovered.keys())[:8]}

Lower-ranked documents (may contain useful NEW directions):
{_docs_620}

Do any documents open a NEW direction not covered by top-5? Extract NEW bridge entities only if clearly connected to the question.
Output JSON only:
{{"new_bridges": ["entity1"], "reasoning": "brief"}}"""

                    _e_resp = llm_client.infer([{"role": "user", "content": _explorer_prompt}]).strip()
                    if "```json" in _e_resp: _e_resp = _e_resp.split("```json")[1].split("```")[0].strip()
                    elif "```" in _e_resp: _e_resp = _e_resp.split("```")[1].split("```")[0].strip()
                    _e_parsed = json.loads(_e_resp)
                    _new_bridges = _e_parsed.get("new_bridges", [])
                    _e_reasoning = _e_parsed.get("reasoning", "")

                    if _new_bridges:
                        _judge_prompt = f"""Judge whether to accept new bridge entities from lower-ranked documents.

Question: {question}
Existing bridges (high confidence): {list(all_discovered.keys())[:8]}
Proposed new bridges: {_new_bridges}
Explorer reasoning: {_e_reasoning[:200]}

Accept ONLY if the bridge clearly connects to the question AND opens a new direction. Reject if redundant or noisy.
Output JSON only:
{{"accept": true, "accepted_bridges": ["entity1"]}}"""

                        _j_resp = llm_client.infer([{"role": "user", "content": _judge_prompt}]).strip()
                        if "```json" in _j_resp: _j_resp = _j_resp.split("```json")[1].split("```")[0].strip()
                        elif "```" in _j_resp: _j_resp = _j_resp.split("```")[1].split("```")[0].strip()
                        _j_parsed = json.loads(_j_resp)
                        if _j_parsed.get("accept", False):
                            _accepted = [str(e).lower().strip() for e in _j_parsed.get("accepted_bridges", []) if isinstance(e, str)]
                            for _ent in _accepted:
                                _ek = compute_hash(_ent, prefix="entity-")
                                _vid = index.node_name_to_idx.get(_ek)
                                if _vid is not None and _ent not in all_discovered:
                                    all_discovered[_ent] = (_vid, 1.0, round_i)
                                    extra_node_weights[_vid] += _degree_adaptive_weight(index, _vid, DEFAULT_ENTITY_SEED_WEIGHT)
                            if _accepted:
                                logger.info(f"  Explorer+Judge accepted: {_accepted}")
                                # Re-run PPR with new bridges
                                working_graph_ej = _build_overlay_graph(index, temp_edges, base_graph=index.graph) if temp_edges else index.graph
                                sorted_doc_ids, sorted_doc_scores, current_node_weights, all_ppr_scores = index.retrieve(
                                    retrieve_query, working_graph=working_graph_ej,
                                    extra_node_weights=extra_node_weights, return_node_weights=True, query_entities=None)
                                base_top_docs = [index.passages[index.passage_keys[idx]] for idx in sorted_doc_ids[:10]]
                except Exception as _ej_err:
                    logger.warning(f"  Explorer+Judge error: {_ej_err}")

            # Last-round reasoning: predict bridges and re-run PPR (no next round, just improve current)
            _last_round_reasoning = os.getenv("LAST_ROUND_REASONING", "0") == "1"
            if _last_round_reasoning:
                try:
                    _lr_output = reason_and_rewrite(
                        original_query=question,
                        current_query=current_query,
                        retrieved_docs=top_docs,
                        round_idx=round_i,
                        previous_traces=reasoning_traces,
                        llm_client=llm_client,
                    )
                    _lr_entities = _lr_output.get("discovered_entities", [])
                    if _lr_entities and not _lr_output.get("should_stop", False):
                        _lr_new = []
                        _lr_resolved = _resolve_entities_in_graph(index, _lr_entities)
                        for _ent, (_vid, _sim) in _lr_resolved.items():
                            if _ent not in all_discovered:
                                all_discovered[_ent] = (_vid, _sim, round_i)
                                extra_node_weights[_vid] += _degree_adaptive_weight(index, _vid, DEFAULT_ENTITY_SEED_WEIGHT)
                                _lr_new.append(_ent)
                        if _lr_new:
                            # Rebuild overlay and re-run PPR
                            _lr_temp = list(temp_edges) if temp_edges else []
                            existing_seeds = _get_existing_seed_ids(index, base_node_weights)
                            discovered_with_decay_lr = {name: (vid, 1.0) for name, (vid, _, _) in all_discovered.items()}
                            ppr_connections_lr = _mini_ppr_select_seeds(
                                index, discovered_vertex_ids=discovered_with_decay_lr,
                                existing_seed_vertex_ids=existing_seeds,
                                round0_top_doc_ids=round0_top_doc_ids,
                                gold_docs=gold_docs, llm_client=llm_client, query=retrieve_query)
                            for d_vid, s_vid, seed_score, joint_norm in ppr_connections_lr:
                                _lr_temp.append((d_vid, s_vid, 2.0))
                            working_graph_lr = _build_overlay_graph(index, _lr_temp, base_graph=index.graph) if _lr_temp else index.graph

                            # If PRN mode: add last-round seeds as new group and re-run with accumulated
                            if _per_round_norm and per_round_seeds:
                                _lr_bridge_seeds = np.zeros(index.graph.vcount())
                                for _ent in _lr_new:
                                    if _ent in all_discovered:
                                        _vid = all_discovered[_ent][0]
                                        _lr_bridge_seeds[_vid] += _degree_adaptive_weight(index, _vid, DEFAULT_ENTITY_SEED_WEIGHT)
                                _lr_round_seeds = index._compute_node_weights(retrieve_query) + _lr_bridge_seeds
                                _lr_s = _lr_round_seeds.sum()
                                if _lr_s > 0:
                                    _lr_normed = _lr_round_seeds / _lr_s
                                else:
                                    _lr_normed = _lr_round_seeds
                                # Replace last round's seeds with updated version
                                per_round_seeds[-1] = (_lr_normed, round_i)
                                # Re-accumulate
                                _prn_decay = float(os.getenv("PRN_DECAY", "0.5"))
                                accumulated_reset_lr = np.zeros(index.graph.vcount())
                                for normed_seeds, stored_round in per_round_seeds:
                                    decay = _prn_decay ** (round_i - stored_round)
                                    accumulated_reset_lr += normed_seeds * decay
                                sorted_doc_ids, sorted_doc_scores, all_ppr_scores = _run_ppr(
                                    index, reset_prob=accumulated_reset_lr, damping=0.5,
                                    graph=working_graph_lr, return_all_scores=True)
                            else:
                                extra_node_weights_lr = extra_node_weights.copy() if extra_node_weights is not None else np.zeros(index.graph.vcount())
                                for _ent in _lr_new:
                                    if _ent in all_discovered:
                                        _vid = all_discovered[_ent][0]
                                        extra_node_weights_lr[_vid] += _degree_adaptive_weight(index, _vid, DEFAULT_ENTITY_SEED_WEIGHT)
                                sorted_doc_ids, sorted_doc_scores, current_node_weights, all_ppr_scores = index.retrieve(
                                    retrieve_query, working_graph=working_graph_lr,
                                    extra_node_weights=extra_node_weights_lr, return_node_weights=True, query_entities=None)

                            base_top_docs = [index.passages[index.passage_keys[idx]] for idx in sorted_doc_ids[:10]]
                            # Update RRF
                            current_rrf = np.zeros(num_docs)
                            for rank, doc_id in enumerate(sorted_doc_ids):
                                current_rrf[doc_id] = 1.0 / (rrf_k + rank + 1)
                            combined_scores = HISTORY_WEIGHT * (rrf_scores / max(round_i, 1)) + CURRENT_WEIGHT * current_rrf
                            rrf_scores += current_rrf
                            final_sorted_ids = np.argsort(combined_scores)[::-1]
                            top_docs = [index.passages[index.passage_keys[idx]] for idx in final_sorted_ids[:10]]
                            logger.info(f"  Last-round reasoning: new bridges {_lr_new[:3]}")
                            round_diag["last_round_bridges"] = _lr_new
                except Exception as _lr_err:
                    logger.warning(f"  Last-round reasoning error: {_lr_err}")

            round_diag["stop"] = False
            round_diag["rewritten_query"] = ""
            round_diag["new_discovered_entities"] = list(all_discovered.keys())
            round_diagnostics.append(round_diag)
            break

        try:
            # Accumulate unique docs across rounds for stable reasoning
            _accumulate_docs = os.getenv("ACCUMULATE_DOCS", "0") == "1"
            if _accumulate_docs:
                for doc in top_docs[:5]:
                    if doc not in accumulated_docs:
                        accumulated_docs.append(doc)
                # Current round's new docs first, then old docs at the back
                current_docs = top_docs[:5]
                old_docs = [d for d in accumulated_docs if d not in current_docs]
                _reasoning_docs = current_docs + old_docs[:5]  # current 5 + up to 5 old
            else:
                _reasoning_docs = top_docs

            reasoning_output = reason_and_rewrite(
                original_query=question,
                current_query=current_query,
                retrieved_docs=_reasoning_docs,
                round_idx=round_i,
                previous_traces=reasoning_traces,
                llm_client=llm_client,
            )
        except Exception as e:
            logger.warning(f"  Reasoning error at round {round_i}: {e}")
            break

        _analysis = reasoning_output.get("analysis", "")
        # Only keep confirmed info in traces (remove missing/gap part to avoid anchoring)
        if os.getenv("TRACE_CONFIRMED_ONLY", "0") == "1" and "|" in _analysis:
            _analysis = _analysis.split("|")[0].strip()
        reasoning_traces.append(_analysis)
        round_diag["rewritten_query"] = reasoning_output.get("rewritten_query", "")
        round_diag["new_discovered_entities"] = reasoning_output.get("discovered_entities", [])
        # Save structured 2-layer entities for case analysis
        round_diag["query_entities"] = reasoning_output.get("query_entities", [])
        round_diag["bridge_entities"] = reasoning_output.get("bridge_entities", [])
        round_diag["stop"] = reasoning_output.get("should_stop", False)

        # FORCE_FULL_ROUNDS=1 disables early stop for ablation (run all max_rounds)
        _force_full = os.getenv("FORCE_FULL_ROUNDS", "0") == "1"
        if reasoning_output.get("should_stop", False) and not _force_full:
            logger.info(f"  Reasoning stop at round {round_i}")
            round_diagnostics.append(round_diag)
            break

        # Bridge-first targeted decompose: after Round 0, generate sub-queries for missing info
        if _decompose and round_i == 0 and not reasoning_output.get("should_stop", False):
            try:
                _bridges_so_far = reasoning_output.get("discovered_entities", [])
                _docs_text = "\n".join([f"[Doc {_di+1}] {d}" for _di, d in enumerate(top_docs[:5])])
                _decompose_prompt = f"""Question: {question}

Bridge entities found: {_bridges_so_far[:5]}

Retrieved documents:
{_docs_text}

Based on the bridges and documents, what specific information is STILL MISSING to answer the question?

Rules:
- Do NOT repeat the original question
- Each sub-query should explore from a bridge entity toward the missing information
- Sub-queries should be simple factual questions, not complex multi-hop
- Only generate sub-queries for information NOT already in the documents
- 1-2 sub-queries maximum

Output JSON only:
{{"sub_queries": ["focused sub-query"]}}"""

                _d_resp = llm_client.infer([{"role": "user", "content": _decompose_prompt}]).strip()
                if "```json" in _d_resp: _d_resp = _d_resp.split("```json")[1].split("```")[0].strip()
                elif "```" in _d_resp: _d_resp = _d_resp.split("```")[1].split("```")[0].strip()
                _sub_qs = json.loads(_d_resp).get("sub_queries", [])

                _sq_extract = 'Sub-question: {sq}\n\nDocuments:\n{docs}\n\nExtract key named entities (people, places, organizations) that help answer this sub-question. Only extract entities from the documents.\nOutput JSON only:\n{{"key_entities": ["entity1", "entity2"]}}'

                _decompose_new = []
                for _sq in _sub_qs[:2]:
                    _sq_w = index._compute_node_weights(_sq)
                    _sq_s, _ = _run_ppr(index, reset_prob=_sq_w, damping=0.5)
                    _sq_docs = "\n".join([f"[Doc {_di+1}] {index.passages[index.passage_keys[_sq_s[_di]]]}" for _di in range(5)])
                    _sq_r = llm_client.infer([{"role": "user", "content": _sq_extract.format(sq=_sq, docs=_sq_docs)}]).strip()
                    if "```json" in _sq_r: _sq_r = _sq_r.split("```json")[1].split("```")[0].strip()
                    elif "```" in _sq_r: _sq_r = _sq_r.split("```")[1].split("```")[0].strip()
                    try:
                        _sq_ents = [str(e).lower().strip() for e in json.loads(_sq_r).get("key_entities", []) if isinstance(e, str)]
                        _decompose_new.extend(_sq_ents)
                    except: pass

                # Add decompose entities to discovered (lower weight via _decompose_weight_tag)
                _decompose_weight = float(os.getenv("DECOMPOSE_WEIGHT", "0.3"))
                _new_from_decompose = []
                existing_ents = reasoning_output.get("discovered_entities", [])
                for _ent in _decompose_new:
                    if _ent not in existing_ents:
                        existing_ents.append(_ent)
                        _new_from_decompose.append(_ent)
                        # Tag decompose entities for lower weight in seed calculation
                        bridge_type_map[_ent] = "decompose"
                reasoning_output["discovered_entities"] = existing_ents
                round_diag["new_discovered_entities"] = existing_ents

                if _new_from_decompose:
                    logger.info(f"  Decompose sub-queries: {_sub_qs[:2]}")
                    logger.info(f"  Decompose new entities: {_new_from_decompose[:5]}")
            except Exception as e:
                logger.warning(f"  Decompose error: {e}")

        # Escalation: on last round, use stronger model + top-10 docs for extra bridge discovery
        _escalation = os.getenv("ESCALATION", "0") == "1"
        if _escalation and round_i == max_rounds - 2 and not reasoning_output.get("should_stop", False):
            _esc_api_key = os.getenv("ESCALATION_API_KEY", "")
            _esc_base_url = os.getenv("ESCALATION_BASE_URL", "")
            _esc_model = os.getenv("ESCALATION_MODEL", "qwen3.6-plus")
            if _esc_api_key and _esc_base_url:
                try:
                    from openai import OpenAI as _EscOpenAI
                    _esc_client = _EscOpenAI(api_key=_esc_api_key, base_url=_esc_base_url, timeout=120)
                    # Get top-10 docs (not just top-5)
                    _esc_docs = ""
                    for _di in range(min(10, len(sorted_doc_ids))):
                        _pk = index.passage_keys[sorted_doc_ids[_di]]
                        _esc_docs += f"[Doc {_di+1}] {index.passages[_pk]}\n\n"

                    _esc_prompt = f"""You are helping a multi-hop QA retrieval system find missing connections in a knowledge graph.

"Bridge entities" are named entities (people, places, organizations, events) that exist in the knowledge graph and can connect retrieved documents to find the answer.

Example:
Question: "When was the birthplace of the designer of Southeast Library abolished?"
Bridge entities already found: ["southeast library", "ralph rapson"]
Retrieved docs mention: Ralph Rapson designed buildings in Minneapolis...
Good new bridge entities: ["minneapolis", "minnesota"]
(These are real entity names that help connect "ralph rapson" → "minneapolis" → answer)

Now your task:
Question: {question}
Bridge entities already found: {list(all_discovered.keys())[:8]}
The answer has NOT been found yet. Read all 10 documents below and find entity names that could bridge the gap.

{_esc_docs}
Do NOT answer the question directly. Output entity NAMES that exist in the documents above.
Output JSON only:
{{"new_entities": ["entity1", "entity2"]}}"""

                    _esc_resp = _esc_client.chat.completions.create(
                        model=_esc_model,
                        messages=[{"role": "user", "content": _esc_prompt}],
                        temperature=0, max_tokens=300
                    )
                    _esc_text = _esc_resp.choices[0].message.content.strip()
                    if "</think>" in _esc_text:
                        _esc_text = _esc_text.split("</think>")[-1].strip()

                    import re as _re_esc
                    _esc_match = _re_esc.search(r'"new_entities"\s*:\s*\[([^\]]*)\]', _esc_text)
                    if _esc_match:
                        _esc_ents = [e.strip().strip('"').strip("'") for e in _esc_match.group(1).split(",") if e.strip()]
                        # Add to discovered entities (will be used in next round)
                        existing_discovered = reasoning_output.get("discovered_entities", [])
                        existing_discovered.extend(_esc_ents)
                        reasoning_output["discovered_entities"] = existing_discovered
                        round_diag["new_discovered_entities"] = existing_discovered
                        logger.info(f"  Escalation ({_esc_model}): {_esc_ents}")
                except Exception as e:
                    logger.warning(f"  Escalation error: {e}")

        # Reflect: discard unhelpful entities from previous rounds
        if os.getenv("REFLECT_DISCARD", "0") == "1":
            discard_entities = reasoning_output.get("discard_entities", [])
            if discard_entities:
                logger.info(f"  Discarding entities: {discard_entities}")
                for d_name in discard_entities:
                    d_name_lower = d_name.lower().strip()
                    if d_name_lower in all_discovered:
                        del all_discovered[d_name_lower]
                        logger.info(f"    Removed '{d_name_lower}'")

        new_query = reasoning_output.get("rewritten_query", "")
        _fix_query = os.getenv("FIX_QUERY", "0") == "1"
        _concat_query = os.getenv("CONCAT_QUERY", "1") == "1"  # default: concat original + rewrite (IRCoT-style)
        _traces_in_query = os.getenv("TRACES_IN_QUERY", "0") == "1"  # add accumulated traces for pair seeds
        if not _fix_query and new_query and new_query != current_query:
            logger.info(f"  Round {round_i} rewrite: '{new_query[:60]}...'")
            if _concat_query:
                current_query = f"{question} {new_query}"  # preserve original signal for comparison cases
            else:
                current_query = new_query
        elif _fix_query:
            logger.info(f"  Round {round_i} query fixed (no rewrite)")
        if _traces_in_query and reasoning_traces:
            traces_text = " ".join(t for t in reasoning_traces if t)
            current_query = f"{current_query} {traces_text}"
            logger.info(f"  Round {round_i} query enriched with {len(reasoning_traces)} traces (total len={len(current_query)})")

        discovered_entities = reasoning_output.get("discovered_entities", [])
        # (entity_meta removed - using code-based from_doc detection instead)
        # Save bridge type map for hierarchical weighting
        if reasoning_output.get("_bridge_types"):
            bridge_type_map.update(reasoning_output["_bridge_types"])
        if discovered_entities:
            resolved = _resolve_entities_in_graph(index, discovered_entities)
            # From-doc fallback: for unresolved entities, auto-detect source doc and match locally
            _unresolved = [e for e in discovered_entities if e not in resolved]
            if _unresolved:
                _passage_set_local = set(index.passage_node_idxs)
                for ent_name in _unresolved:
                    # Auto-detect from_doc: find which top doc contains this entity name
                    _from_doc_idx = -1
                    for _di in range(min(10, len(sorted_doc_ids))):
                        _pk = index.passage_keys[sorted_doc_ids[_di]]
                        if ent_name.lower() in index.passages[_pk].lower():
                            _from_doc_idx = _di
                            break
                    if _from_doc_idx < 0:
                        continue
                    doc_pk = index.passage_keys[sorted_doc_ids[_from_doc_idx]]
                    # Get doc's entities
                    doc_ent_vids = []
                    for ek, pids in index.entity_to_passages.items():
                        if doc_pk in pids:
                            v = index.node_name_to_idx.get(ek)
                            if v is not None and v not in _passage_set_local and v < len(index.entity_keys):
                                doc_ent_vids.append(v)
                    if not doc_ent_vids:
                        continue
                    # Match entity name against doc entities by embedding
                    _q_emb = index.embedding_model.batch_encode([ent_name])
                    _q_emb = _q_emb / (np.linalg.norm(_q_emb, axis=1, keepdims=True) + 1e-10)
                    _d_embs = index.entity_embeddings[doc_ent_vids]
                    _d_norms = np.linalg.norm(_d_embs, axis=1, keepdims=True)
                    _d_norms = np.where(_d_norms == 0, 1, _d_norms)
                    _d_embs = _d_embs / _d_norms
                    _sims = (_q_emb @ _d_embs.T).flatten()
                    _best = int(np.argmax(_sims))
                    if _sims[_best] > 0.4:
                        matched_vid = doc_ent_vids[_best]
                        matched_name = index.graph.vs[matched_vid]["content"]
                        resolved[ent_name] = (matched_vid, float(_sims[_best]))
                        logger.info(f"  From-doc fallback: '{ent_name}' → '{matched_name}' (sim={_sims[_best]:.3f}, doc={_from_doc_idx+1})")
            for name, (vid, sim) in resolved.items():
                if name not in all_discovered:
                    all_discovered[name] = (vid, sim, round_i)
                else:
                    old_vid, old_sim, old_round = all_discovered[name]
                    avg_round = (old_round + round_i) / 2.0
                    all_discovered[name] = (old_vid, max(old_sim, sim), avg_round)
            logger.info(f"  Resolved {len(resolved)}/{len(discovered_entities)} bridge entities in graph")

        # Dead-end LLM bridge selection: DISABLED
            round_diag["discovered_entities_total"] = sorted(all_discovered.keys())

            # Prune-and-expand: collect bridge neighborhood edges for next round
            if _prune_expand and focused_graph is not None and resolved:
                bridge_vids = [vid for vid, sim in resolved.values()]
                neighborhood_edges = _collect_bridge_neighborhood_edges(
                    index.graph, bridge_vids,
                    n_hops=_expand_hops,
                    passage_set=set(index.passage_node_idxs)
                )
                bridge_neighborhood_edges.extend(neighborhood_edges)

        round_diagnostics.append(round_diag)

    # Final ranking: history + last round (already computed as combined_scores)
    final_sorted_ids = np.argsort(combined_scores)[::-1]
    retrieved_docs = [index.passages[index.passage_keys[idx]] for idx in final_sorted_ids]

    # ABLATION: Run PPR with Round 0 seeds only + last-round overlay edges
    # Tests if overlay edges alone (without bridge seed expansion) help retrieval
    if os.getenv("OVERLAY_ONLY_ABLATION", "0") == "1" and temp_edges and base_node_weights is not None:
        try:
            abl_graph = _build_overlay_graph(index, temp_edges, base_graph=index.graph)
            abl_sorted, abl_scores, _, _ = index.retrieve(
                question,  # original query, no rewrite
                working_graph=abl_graph,
                extra_node_weights=None,  # no bridge seeds
                return_node_weights=True,
                query_entities=None,
            )
            abl_top_docs = [index.passages[index.passage_keys[idx]] for idx in abl_sorted[:10]]
            if gold_docs is not None:
                abl_r5 = recall_at_k(abl_top_docs, gold_docs, 5)
                abl_r1 = recall_at_k(abl_top_docs, gold_docs, 1)
                logger.info(f"  [OVERLAY-ONLY ABLATION] edges={len(temp_edges)}  R@1={abl_r1:.3f}  R@5={abl_r5:.3f}")
        except Exception as e:
            logger.warning(f"  Overlay-only ablation failed: {e}")

    # ABLATION: Run PPR with bridge seeds but no overlay edges
    # Tests if overlay edges add value beyond bridge seed expansion
    if os.getenv("SEEDS_ONLY_ABLATION", "0") == "1" and extra_node_weights is not None and all_discovered:
        try:
            abl_sorted, abl_scores, _, _ = index.retrieve(
                current_query,  # use latest rewritten query (same as full method)
                working_graph=index.graph,  # base graph, no overlay edges
                extra_node_weights=extra_node_weights,  # bridge seeds
                return_node_weights=True,
                query_entities=None,
            )
            abl_top_docs = [index.passages[index.passage_keys[idx]] for idx in abl_sorted[:10]]
            if gold_docs is not None:
                abl_r5 = recall_at_k(abl_top_docs, gold_docs, 5)
                abl_r1 = recall_at_k(abl_top_docs, gold_docs, 1)
                logger.info(f"  [SEEDS-ONLY ABLATION] bridges={len(all_discovered)}  R@1={abl_r1:.3f}  R@5={abl_r5:.3f}")
        except Exception as e:
            logger.warning(f"  Seeds-only ablation failed: {e}")

    # CONSOLIDATION ROUND: single PPR with original query + all accumulated bridges + all overlay edges
    # Uses the 3-round memory to do one final clean diffusion (no RRF fusion)
    if os.getenv("CONSOLIDATION_ROUND", "0") == "1" and all_discovered and base_node_weights is not None:
        try:
            # Build extra weights: all accumulated bridges with their original decay
            consol_extra = np.zeros(index.graph.vcount())
            for name, (vid, res_sim, disc_round) in all_discovered.items():
                q_sim = 1.0  # no decay for consolidation
                consol_extra[vid] += _degree_adaptive_weight(index, vid, DEFAULT_ENTITY_SEED_WEIGHT, sim=q_sim)

            # Use last-round temp_edges as overlay (current accumulated state)
            consol_graph = _build_overlay_graph(index, temp_edges, base_graph=index.graph) if temp_edges else index.graph

            consol_sorted, consol_scores, _, _ = index.retrieve(
                question,  # ORIGINAL query, not rewritten
                working_graph=consol_graph,
                extra_node_weights=consol_extra,
                return_node_weights=True,
                query_entities=None,
            )
            consol_top_docs = [index.passages[index.passage_keys[idx]] for idx in consol_sorted[:10]]
            if gold_docs is not None:
                c_r1 = recall_at_k(consol_top_docs, gold_docs, 1)
                c_r5 = recall_at_k(consol_top_docs, gold_docs, 5)
                logger.info(f"  [CONSOLIDATION] bridges={len(all_discovered)} edges={len(temp_edges)}  R@1={c_r1:.3f}  R@5={c_r5:.3f}")
        except Exception as e:
            logger.warning(f"  Consolidation round failed: {e}")

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

# Second demo: 3-hop chain (adapted from Trivedi 2023 IRCoT prompt — demonstrates multi-step reasoning so LLM doesn't always terminate at round 0)
IRCOT_TWO_SHOT_DOCS = (
    """Wikipedia Title: Smoke in tha City\nSmoke in tha City is the ninth studio album by American rapper MC Eiht. The album was released in 2004 on Tha Hall Records.\n\n"""
    """Wikipedia Title: MC Eiht\nAaron Tyler (born May 22, 1971), better known by his stage name MC Eiht, is an American rapper born in Compton, California. He is a former member of the rap group Compton's Most Wanted.\n\n"""
    """Wikipedia Title: Compton, California\nCompton is a city in southern Los Angeles County, California, located south of downtown Los Angeles. The city was founded in 1867 by Griffith Dickenson Compton.\n\n"""
    """Wikipedia Title: Los Angeles County, California\nLos Angeles County, officially the County of Los Angeles, is the most populous county in the United States, with more than 9.8 million residents as of 2024.\n\n"""
    """Wikipedia Title: Tha Hall Records\nTha Hall Records was an independent record label active in the early 2000s, primarily releasing West Coast hip-hop.\n"""
)

IRCOT_TWO_SHOT_DEMO = (
    f'{IRCOT_TWO_SHOT_DOCS}'
    '\n\nQuestion: '
    f"In what county is the birthplace of the performer of Smoke in tha City?"
    '\nThought: '
    f"The performer of Smoke in tha City is MC Eiht. MC Eiht was born in Compton, California. Compton is located in Los Angeles County. So the answer is: Los Angeles County."
    '\n\n'
)

_IRCOT_INSTRUCTION_MODE = os.getenv("IRCOT_USE_INSTRUCTION", "1")
if _IRCOT_INSTRUCTION_MODE == "1":
    # Full instruction (current)
    IRCOT_SYSTEM = (
        'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! Only conclude with "So the answer is:" when the final answer is explicitly supported by the retrieved passages above; if the passages don\'t contain enough evidence, output the next fact you need to find so the system can retrieve more.'
        '\n\n'
        f'{IRCOT_ONE_SHOT_DEMO}'
        f'{IRCOT_TWO_SHOT_DEMO}'
    )
elif _IRCOT_INSTRUCTION_MODE == "A":
    # Option A: evidence-supported only, no query-expansion hint
    IRCOT_SYSTEM = (
        'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! Only conclude with "So the answer is:" when the final answer is explicitly supported by the retrieved passages above.'
        '\n\n'
        f'{IRCOT_ONE_SHOT_DEMO}'
        f'{IRCOT_TWO_SHOT_DEMO}'
    )
else:
    # Pure few-shot, no instruction (Trivedi-style minimal)
    IRCOT_SYSTEM = (
        f'{IRCOT_ONE_SHOT_DEMO}'
        f'{IRCOT_TWO_SHOT_DEMO}'
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
    """IRCoT with configurable query + doc modes.

    ENV vars:
      IRCOT_QUERY_MODE: 'last_thought' (default) or 'last_sentence' (Trivedi-strict)
      IRCOT_DOC_MODE:   'accumulate' (default) or 'last_round' (replace each round)
      IRCOT_PER_ROUND_K: top-K per round (default 5)
      IRCOT_MAX_PARAS:  cap for accumulate mode (default 15)

    Returns: (reader_docs, last_round_docs, thoughts, ircot_answer)
    """
    import os as _os, re as _re
    per_round_k = int(_os.getenv("IRCOT_PER_ROUND_K", "5"))
    max_paras = int(_os.getenv("IRCOT_MAX_PARAS", "15"))
    query_mode = _os.getenv("IRCOT_QUERY_MODE", "last_thought")
    doc_mode = _os.getenv("IRCOT_DOC_MODE", "accumulate")

    thoughts = []
    accumulated_docs = []
    accumulated_set = set()
    last_round_docs = []
    ircot_answer = None

    for round_i in range(max_rounds):
        if round_i == 0 or not thoughts:
            current_query = question
        else:
            if query_mode == "next_fact":
                # Extract sub-question after "next fact I need to find is:" pattern
                _m = _re.search(
                    r'(?:next fact i need to find is|i need to find|next,?\s*i need to)[:\s]+(.+?)(?:[\?]|\.\s*$|$)',
                    thoughts[-1], _re.IGNORECASE | _re.DOTALL)
                if _m:
                    current_query = _m.group(1).strip().rstrip('?.').strip()
                else:
                    sents = _re.split(r'(?<=[.!?])\s+', thoughts[-1].strip())
                    current_query = sents[-1] if sents else thoughts[-1]
            elif query_mode == "last_sentence":
                sents = _re.split(r'(?<=[.!?])\s+', thoughts[-1].strip())
                current_query = sents[-1] if sents else thoughts[-1]
            else:
                current_query = thoughts[-1]

        sorted_doc_ids, _ = index.retrieve(current_query)
        last_round_docs = [index.passages[index.passage_keys[did]]
                           for did in sorted_doc_ids[:per_round_k]]
        round_added = 0
        for did in sorted_doc_ids:
            if round_added >= per_round_k or len(accumulated_docs) >= max_paras:
                break
            doc = index.passages[index.passage_keys[did]]
            if doc not in accumulated_set:
                accumulated_set.add(doc)
                accumulated_docs.append(doc)
                round_added += 1

        reader_docs_for_thought = accumulated_docs if doc_mode == "accumulate" else last_round_docs
        thought = ircot_reason_step(
            query=question,
            passages=reader_docs_for_thought,
            thoughts=thoughts,
            llm_client=llm_client,
        )
        logger.info(f"  IRCoT round {round_i}: +{round_added} docs (acc={len(accumulated_docs)}, last_round={len(last_round_docs)}), thought='{thought[:60]}...'")

        if "so the answer is:" in thought.lower():
            thoughts.append(thought)
            m = _re.search(r'so the answer is:?\s*(.+?)(?:\.\s*$|\n|$)', thought, _re.IGNORECASE | _re.DOTALL)
            if m:
                ircot_answer = m.group(1).strip().strip('"').strip("'").rstrip('.').strip()
            logger.info(f"  IRCoT stopped at round {round_i}, extracted answer: '{ircot_answer}'")
            break
        thoughts.append(thought)
        if len(accumulated_docs) >= max_paras and doc_mode == "accumulate":
            logger.info(f"  IRCoT cap reached at {max_paras}, stopping early")
            break

    reader_docs = accumulated_docs if doc_mode == "accumulate" else last_round_docs
    return reader_docs, last_round_docs, thoughts, ircot_answer


# ── Self-Ask (Press et al. 2022/2023) ──────────────────────────

SELF_ASK_DEMO_1 = (
    "Question: When was Neville A. Stanton's employer founded?\n"
    "Are follow up questions needed here: Yes.\n"
    "Follow up: Who is Neville A. Stanton's employer?\n"
    "Intermediate answer: Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton.\n"
    "Follow up: When was the University of Southampton founded?\n"
    "Intermediate answer: The University of Southampton was founded in 1862 and received its Royal Charter as a university in 1952.\n"
    "So the final answer is: 1862\n\n"
)

SELF_ASK_DEMO_2 = (
    "Question: In what county is the birthplace of the performer of Smoke in tha City?\n"
    "Are follow up questions needed here: Yes.\n"
    "Follow up: Who is the performer of Smoke in tha City?\n"
    "Intermediate answer: Smoke in tha City is the ninth studio album by American rapper MC Eiht.\n"
    "Follow up: Where was MC Eiht born?\n"
    "Intermediate answer: Aaron Tyler, better known as MC Eiht, is an American rapper born in Compton, California.\n"
    "Follow up: What county is Compton, California in?\n"
    "Intermediate answer: Compton is a city in southern Los Angeles County, California.\n"
    "So the final answer is: Los Angeles County\n\n"
)

SELF_ASK_PROMPT_PREFIX = SELF_ASK_DEMO_1 + SELF_ASK_DEMO_2


def _selfask_summarize_docs(sub_q: str, docs: List[str], llm_client) -> str:
    """Summarize retrieved docs into ONE-sentence intermediate answer (like google snippet)."""
    docs_text = "\n\n".join(d[:500] for d in docs[:5])
    user_content = (
        f"Passages:\n{docs_text}\n\n"
        f"Question: {sub_q}\n"
        f"Provide a brief, factual answer in ONE sentence based on the passages. "
        f"If the passages don't contain the answer, say 'Information not found in passages.'"
    )
    messages = [{"role": "user", "content": user_content}]
    try:
        kwargs = dict(model=llm_client.model_name, messages=messages, temperature=0, max_tokens=100)
        if "qwen3" in llm_client.model_name.lower():
            kwargs["extra_body"] = {"enable_thinking": False}
        resp = llm_client.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Self-Ask summarize failed: {e}")
        return (docs[0][:200].strip() if docs else "Information not found in passages.")


def self_ask_retrieve(index, question: str, llm_client, max_rounds: int = 3):
    """Strict Self-Ask + NER+PPR (replaces google):
    1. LLM sees ONLY demos + question (no docs)
    2. Stop tokens cut at Intermediate answer / Follow up / final answer boundaries
    3. After Follow-up Q: NER+PPR retrieve top-K → LLM summarize docs → 1-sentence intermediate answer
    4. Inject "Intermediate answer: <text>" into context, LLM continues
    5. Repeat until "So the final answer is: X" → extract X

    Returns: (last_round_docs, accumulated_docs, chain_full, final_answer_or_None)
    """
    import os as _os, re as _re
    per_round_k = int(_os.getenv("IRCOT_PER_ROUND_K", "5"))

    prompt = SELF_ASK_PROMPT_PREFIX + f"Question: {question}\nAre follow up questions needed here:"
    generated = ""
    accumulated_docs = []
    accumulated_set = set()
    last_round_docs = []
    follow_ups = []
    final_answer = None

    stop_tokens = ["\nIntermediate answer:", "\nSo the final answer is:"]

    for round_i in range(max_rounds):
        messages = [{"role": "user", "content": prompt + generated}]
        try:
            kwargs = dict(model=llm_client.model_name, messages=messages, temperature=0,
                          max_tokens=256, stop=stop_tokens)
            if "qwen3" in llm_client.model_name.lower():
                kwargs["extra_body"] = {"enable_thinking": False}
            resp = llm_client.client.chat.completions.create(**kwargs)
            response_text = resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"Self-Ask round {round_i} LLM gen failed: {e}")
            break

        generated += response_text

        if "so the final answer is:" in response_text.lower():
            m = _re.search(r'so the final answer is:?\s*(.+?)(?:\.\s*$|\n|$)', response_text, _re.IGNORECASE | _re.DOTALL)
            if m:
                final_answer = m.group(1).strip().strip('"').strip("'").rstrip('.').strip()
            logger.info(f"  Self-Ask round {round_i}: FINAL='{final_answer}'")
            break

        # Try to extract Follow-up from response_text
        m = _re.search(r'follow up\s*:?\s*(.+?)(?=\n|$)', response_text, _re.IGNORECASE)
        if not m:
            logger.info(f"  Self-Ask round {round_i}: no Follow-up extracted. Last: '{response_text.strip().split(chr(10))[-1][:80]}'")
            break

        sub_q = m.group(1).strip().rstrip('?').strip()
        if sub_q and not sub_q.endswith('?'):
            sub_q = sub_q + '?'
        follow_ups.append(sub_q)

        # NER+PPR retrieve for sub-question
        sorted_doc_ids, _ = index.retrieve(sub_q)
        last_round_docs = [index.passages[index.passage_keys[did]] for did in sorted_doc_ids[:per_round_k]]
        for d in last_round_docs:
            if d not in accumulated_set:
                accumulated_set.add(d)
                accumulated_docs.append(d)

        # Summarize docs → 1-sentence intermediate answer (2nd LLM call this round)
        intermediate = _selfask_summarize_docs(sub_q, last_round_docs, llm_client)
        generated += f"\nIntermediate answer: {intermediate}\n"

        logger.info(f"  Self-Ask round {round_i}: Q='{sub_q[:60]}' → A='{intermediate[:80]}'")

    return last_round_docs, accumulated_docs, [generated], final_answer


# ── ITER-RETGEN (Shao et al. EMNLP Findings 2023) ──────────────

_QA_JSON_DEMO_INPUT = (
    f"{QA_ONE_SHOT_DOCS}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?\n"
    'Output JSON: {"thought": "<brief reasoning>", "answer": "<concise answer>"}'
)
_QA_JSON_DEMO_OUTPUT = (
    '{"thought": "The employer of Neville A. Stanton is University of Southampton, '
    'founded in 1862.", "answer": "1862"}'
)


def _llm_qa_full(query: str, docs: List[str], llm_client) -> str:
    """JSON-format QA: LLM outputs {"thought": "...", "answer": "..."}.
    Returns reconstructed "Thought: ...\\nAnswer: ..." string for downstream compat.
    Ensures answer is never cut off by max_tokens (JSON is structured)."""
    import json as _j, re as _re_local
    prompt_user = ''
    for passage in docs[:5]:
        prompt_user += f'Wikipedia Title: {passage}\n\n'
    prompt_user += (
        f'Question: {query}\n'
        'Output JSON: {"thought": "<brief reasoning>", "answer": "<concise answer>"}'
    )
    messages = [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": _QA_JSON_DEMO_INPUT},
        {"role": "assistant", "content": _QA_JSON_DEMO_OUTPUT},
        {"role": "user", "content": prompt_user},
    ]
    try:
        kwargs = dict(model=llm_client.model_name, messages=messages, temperature=0, max_tokens=200)
        if "qwen3" in llm_client.model_name.lower():
            kwargs["extra_body"] = {"enable_thinking": False}
        resp = llm_client.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content.strip()
        # Parse JSON
        m = _re_local.search(r'\{[^{}]*\}', text, _re_local.DOTALL)
        if m:
            try:
                data = _j.loads(m.group())
                thought = data.get("thought", "")
                answer = data.get("answer", "")
                return f"Thought: {thought}\nAnswer: {answer}"
            except Exception:
                pass
        # Fallback: return text as-is
        return text
    except Exception as e:
        logger.warning(f"llm_qa_full failed: {e}")
        return ""


def iter_retgen_retrieve(index, question: str, llm_client, max_rounds: int = 3):
    """ITER-RETGEN: iter_num rounds of (retrieve → generate); each round's query =
    question + previous round's FULL generation. No structural prompts, no stop tokens.
    Final answer = extracted from last round's generation via "Answer:" regex.

    Reference: FlashRAG IterativePipeline (https://github.com/RUC-NLPIR/FlashRAG).

    Returns: (last_round_docs, accumulated_docs, generations, final_answer)
    """
    import os as _os, re as _re
    per_round_k = int(_os.getenv("IRCOT_PER_ROUND_K", "5"))

    past_gen = ""
    generations = []
    last_round_docs = []
    accumulated_docs = []
    accumulated_set = set()

    per_round_doc_scores = []  # for KL/distribution analysis
    for iter_idx in range(max_rounds):
        query = question + (" " + past_gen if past_gen else "")

        sorted_doc_ids, doc_scores = index.retrieve(query)
        last_round_docs = [index.passages[index.passage_keys[did]] for did in sorted_doc_ids[:per_round_k]]
        for d in last_round_docs:
            if d not in accumulated_set:
                accumulated_set.add(d)
                accumulated_docs.append(d)
        # Save FULL doc scores in original passage order (for KL analysis)
        if _os.getenv("SAVE_PPR_DIST", "0") == "1":
            import numpy as _np_local
            score_vec = _np_local.zeros(len(index.passage_keys))
            for did, score in zip(sorted_doc_ids, doc_scores):
                if did < len(score_vec):
                    score_vec[did] = score
            per_round_doc_scores.append(score_vec.tolist())

        past_gen = _llm_qa_full(question, last_round_docs, llm_client)
        generations.append(past_gen)

        logger.info(f"  ITER-RETGEN iter {iter_idx}: query_len={len(query)}, docs={len(last_round_docs)}, gen='{past_gen[:80]}...'")

    # Extract final answer from last generation using "Answer:" regex
    final_answer = None
    if generations:
        m = _re.search(r'Answer:?\s*(.+?)(?:\.\s*$|\n|$)', generations[-1], _re.IGNORECASE | _re.DOTALL)
        if m:
            final_answer = m.group(1).strip().strip('"').strip("'").rstrip('.').strip()
        else:
            final_answer = generations[-1].strip()
    # Attach per_round_doc_scores to function attr for main loop pickup
    iter_retgen_retrieve._last_per_round_doc_scores = per_round_doc_scores

    return last_round_docs, accumulated_docs, generations, final_answer


# ── ITER-RETGEN + Bridge Injection (ours) ─────────────────────

def _extract_strict_bridges(question: str, docs: List[str], generation: str,
                             implicit_seeds: str, llm_client) -> List[str]:
    """v5: bridge entities = specific + contextual entities NOT in original question
    but important for answering (aligned with original reasoning method's definition)."""
    import re as _re_local
    docs_text = "\n\n".join(f"Passage {i+1}: {d[:300]}" for i, d in enumerate(docs[:5]))
    user_content = (
        f"Original question: {question}\n\n"
        f"Retrieved passages:\n{docs_text}\n\n"
        f"Current answer attempt:\n{generation[:500]}\n\n"
        f"Entities already covered by current retrieval: {implicit_seeds}\n\n"
        f"Identify BRIDGE ENTITIES that connect what's been found to what's still needed. "
        f"Bridge entities are entities from the passages/generation that:\n"
        f"  • Are NOT in the original question text\n"
        f"  • Are important for answering the question\n"
        f"  • Are NOT yet in the already-covered list\n\n"
        f"Include both:\n"
        f"  - Specific entities (precise named entities like person, place, date, organization)\n"
        f"  - Contextual entities (related concepts/topics that help locate relevant evidence)\n\n"
        f"List 2-4 such bridges. Prefer named entities. Use natural casing (e.g., 'Martin Eberhard').\n"
        f'Output ONLY JSON: ["entity1", "entity2", ...]'
    )
    messages = [{"role": "user", "content": user_content}]
    try:
        kwargs = dict(model=llm_client.model_name, messages=messages, temperature=0, max_tokens=200)
        if "qwen3" in llm_client.model_name.lower():
            kwargs["extra_body"] = {"enable_thinking": False}
        resp = llm_client.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        m = _re_local.search(r'\[.*?\]', text, _re_local.DOTALL)
        if m:
            try:
                return [str(e).strip() for e in json.loads(m.group()) if str(e).strip()][:5]
            except Exception:
                return []
        return []
    except Exception as e:
        logger.warning(f"Strict bridge extraction failed: {e}")
        return []


def _extract_complementary_bridges(question: str, docs: List[str], generation: str,
                                    implicit_seeds: str, llm_client) -> List[str]:
    """v4: extract bridges that are COMPLEMENTARY to current implicit (text-derived) seeds.
    Sees: question, docs, generation, AND the implicit top-5 entities already covered.
    LLM picks NEW entities that add information beyond what's already seeded."""
    import re as _re_local
    docs_text = "\n\n".join(f"Passage {i+1}: {d[:300]}" for i, d in enumerate(docs[:5]))
    user_content = (
        f"Question: {question}\n\n"
        f"Retrieved passages:\n{docs_text}\n\n"
        f"Current answer attempt:\n{generation[:500]}\n\n"
        f"Entities ALREADY covered by current retrieval: {implicit_seeds}\n\n"
        f"Identify 2-4 ADDITIONAL bridge entities (people, places, organizations, events, dates) "
        f"mentioned in passages or generation that are NOT in the already-covered list and "
        f"are critical for answering the question. Focus on entities that would help retrieve "
        f"more evidence in the next round.\n"
        f'Output ONLY a JSON list: ["entity1", "entity2", ...]'
    )
    messages = [{"role": "user", "content": user_content}]
    try:
        kwargs = dict(model=llm_client.model_name, messages=messages, temperature=0, max_tokens=200)
        if "qwen3" in llm_client.model_name.lower():
            kwargs["extra_body"] = {"enable_thinking": False}
        resp = llm_client.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        m = _re_local.search(r'\[.*?\]', text, _re_local.DOTALL)
        if m:
            try:
                return [str(e).strip() for e in json.loads(m.group()) if str(e).strip()][:5]
            except Exception:
                return []
        return []
    except Exception as e:
        logger.warning(f"Complementary bridge extraction failed: {e}")
        return []


def _extract_bridges_gap_analysis(question: str, docs: List[str], generation: str, llm_client) -> List[str]:
    """2-call mode: extract bridge entities from generation that should be verified next round."""
    import re as _re_local
    docs_text = "\n\n".join(f"Passage {i+1}: {d[:400]}" for i, d in enumerate(docs[:5]))
    user_content = (
        f"Question: {question}\n\n"
        f"Retrieved passages:\n{docs_text}\n\n"
        f"Current answer attempt:\n{generation[:800]}\n\n"
        f"From the answer attempt, extract the KEY ENTITIES (people, places, organizations, events, dates) "
        f"that bridge between the question and the answer. These will be used to retrieve more evidence "
        f"in the next round.\n"
        f"Output 2-4 most important entities. Prefer named entities (proper nouns, specific dates) over generic terms.\n"
        f'Output ONLY a JSON list: ["entity1", "entity2", ...]'
    )
    messages = [{"role": "user", "content": user_content}]
    try:
        kwargs = dict(model=llm_client.model_name, messages=messages, temperature=0, max_tokens=200)
        if "qwen3" in llm_client.model_name.lower():
            kwargs["extra_body"] = {"enable_thinking": False}
        resp = llm_client.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        m = _re_local.search(r'\[.*?\]', text, _re_local.DOTALL)
        if m:
            try:
                return [str(e).strip() for e in json.loads(m.group()) if str(e).strip()][:5]
            except Exception:
                return []
        return []
    except Exception as e:
        logger.warning(f"Bridge gap analysis failed: {e}")
        return []


def _llm_qa_with_bridges_json(question: str, docs: List[str], llm_client) -> dict:
    """1-call mode: LLM outputs JSON with both full answer and unverified bridge entities."""
    docs_text = ""
    for passage in docs[:5]:
        docs_text += f'Wikipedia Title: {passage}\n\n'
    user_content = (
        f"{docs_text}"
        f"Question: {question}\n\n"
        f"Provide a step-by-step answer using the passages. Then identify entities your answer "
        f"depends on but the passages don't fully support (for the next retrieval round).\n\n"
        f"Output ONLY valid JSON:\n"
        f'{{"thought": "<reasoning>", "answer": "<final answer>", "unverified_entities": ["entity1", ...]}}'
    )
    messages = [{"role": "user", "content": user_content}]
    try:
        import re as _re_local
        kwargs = dict(model=llm_client.model_name, messages=messages, temperature=0, max_tokens=512)
        if "qwen3" in llm_client.model_name.lower():
            kwargs["extra_body"] = {"enable_thinking": False}
        resp = llm_client.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        m = _re_local.search(r'\{.*\}', text, _re_local.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                return {
                    "generation": f"Thought: {data.get('thought','')}\nAnswer: {data.get('answer','')}",
                    "answer": data.get("answer", ""),
                    "bridges": [str(e).strip() for e in data.get("unverified_entities", []) if str(e).strip()][:5],
                }
            except Exception:
                pass
        return {"generation": text, "answer": "", "bridges": []}
    except Exception as e:
        logger.warning(f"1-call JSON failed: {e}")
        return {"generation": "", "answer": "", "bridges": []}


def iter_retgen_bridge_retrieve(index, question: str, llm_client, max_rounds: int = 3):
    """ITER-RETGEN + bridge injection (v4).

    Each round (2 LLM calls):
    1. Retrieve with text expansion (ITER-RETGEN) + accumulated bridge PPR seeds (degree-adaptive)
    2. LLM #1: generate full answer (ITER-RETGEN style)
    3. LLM #2: extract complementary bridges (entities in docs/gen NOT covered by current top-5 seeds)
    4. Accumulate bridges with degree-adaptive weights

    Returns: (last_round_docs, accumulated_docs, generations, final_answer)
    """
    import os as _os, re as _re
    import numpy as _np
    per_round_k = int(_os.getenv("IRCOT_PER_ROUND_K", "5"))
    base_bridge_weight = float(_os.getenv("IRGENBR_BRIDGE_WEIGHT", "0.5"))

    past_gen = ""
    generations = []
    accumulated_bridges = {}   # name -> (vid, base_weight, disc_round)
    last_round_docs = []
    accumulated_docs = []
    accumulated_set = set()
    seed_decay = float(_os.getenv("IRGENBR_SEED_DECAY", "0.5"))  # 0.5 per round age (matches reasoning)
    per_round_doc_scores = []

    for iter_idx in range(max_rounds):
        # Step 1: build PPR extra seeds from accumulated bridges (degree-adaptive + round decay)
        extra_w = None
        if accumulated_bridges:
            extra_w = _np.zeros(index.graph.vcount())
            for name, (vid, base_w, disc_round) in accumulated_bridges.items():
                age = iter_idx - disc_round
                decay = seed_decay ** age
                extra_w[vid] += base_w * decay

        # Step 2: text expansion (ITER-RETGEN style)
        text_query = question + (" " + past_gen if past_gen else "")

        # Step 3: retrieve with dual signals (text query + bridge PPR seeds)
        sorted_doc_ids, doc_scores = index.retrieve(text_query, extra_node_weights=extra_w)
        last_round_docs = [index.passages[index.passage_keys[did]] for did in sorted_doc_ids[:per_round_k]]
        # Save full doc scores for KL analysis
        if _os.getenv("SAVE_PPR_DIST", "0") == "1":
            score_vec = _np.zeros(len(index.passage_keys))
            for did, score in zip(sorted_doc_ids, doc_scores):
                if did < len(score_vec):
                    score_vec[did] = score
            per_round_doc_scores.append(score_vec.tolist())
        for d in last_round_docs:
            if d not in accumulated_set:
                accumulated_set.add(d)
                accumulated_docs.append(d)

        # Build "already-covered" entity set = this-round text-derived implicit seeds
        # + ALL accumulated bridges from previous rounds (so LLM doesn't re-pick them)
        implicit_seeds_text = ""
        try:
            base_w = index._compute_node_weights(text_query)
            top_ent_idxs = sorted(
                [(int(i), float(base_w[i])) for i in index.entity_node_idxs if base_w[i] > 0],
                key=lambda x: -x[1])[:5]
            implicit_names = []
            for vid, _ in top_ent_idxs:
                hash_key = None
                for nm, vidx in index.node_name_to_idx.items():
                    if vidx == vid:
                        hash_key = nm
                        break
                if hash_key and hash_key in index.entity_keys:
                    i = index.entity_keys.index(hash_key)
                    implicit_names.append(index.entities[i])
            # Union with accumulated bridges (avoid re-picking historical bridges)
            covered_set = list(dict.fromkeys(implicit_names + list(accumulated_bridges.keys())))
            implicit_seeds_text = ", ".join(covered_set[:15])  # cap to keep prompt short
        except Exception as e:
            logger.debug(f"implicit seeds extraction failed: {e}")

        # Step 4: LLM #1 — generate full answer (ITER-RETGEN)
        past_gen = _llm_qa_full(question, last_round_docs, llm_client)
        generations.append(past_gen)

        # Step 5: LLM #2 — extract bridges from docs + generation (mode-controlled)
        _bridge_style = _os.getenv("IRGENBR_BRIDGE_STYLE", "complementary")
        if _bridge_style == "strict":
            new_bridges_names = _extract_strict_bridges(
                question=question, docs=last_round_docs, generation=past_gen,
                implicit_seeds=implicit_seeds_text, llm_client=llm_client,
            )
        else:
            new_bridges_names = _extract_complementary_bridges(
                question=question, docs=last_round_docs, generation=past_gen,
                implicit_seeds=implicit_seeds_text, llm_client=llm_client,
            )

        # Step 6: resolve bridges to KG nodes, accumulate with degree-adaptive weight + disc_round tag
        if new_bridges_names:
            resolved = _resolve_entities_in_graph(index, new_bridges_names)
            for name, (vid, sim) in resolved.items():
                w = _degree_adaptive_weight(index, vid, base_bridge_weight, sim=sim)
                # If already exists, refresh to current round (re-discover = full weight)
                accumulated_bridges[name] = (vid, w, iter_idx)

        logger.info(f"  ITER-RETGEN+BR v4 iter {iter_idx}: implicit=[{implicit_seeds_text[:80]}], new_bridges={new_bridges_names}, total={len(accumulated_bridges)}, gen='{past_gen[:60]}...'")

    # Extract final answer
    final_answer = None
    if generations:
        m = _re.search(r'Answer:?\s*(.+?)(?:\.\s*$|\n|$)', generations[-1], _re.IGNORECASE | _re.DOTALL)
        if m:
            final_answer = m.group(1).strip().strip('"').strip("'").rstrip('.').strip()
        else:
            final_answer = generations[-1].strip()
    iter_retgen_bridge_retrieve._last_per_round_doc_scores = per_round_doc_scores
    return last_round_docs, accumulated_docs, generations, final_answer


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
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                )
                # Disable thinking mode for qwen3.6 models
                if "qwen3" in self.model_name.lower():
                    kwargs["extra_body"] = {"enable_thinking": False}
                resp = self.client.chat.completions.create(**kwargs)
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

    # Auto-select save dir based on dataset
    _data_name = os.path.splitext(os.path.basename(args.data_path))[0]
    save_dir = f"outputs/{_data_name}_ner_pipeline_eval"
    os.makedirs(save_dir, exist_ok=True)
    mode = getattr(args, 'mode', 'base')
    output_path = _make_results_output_path(save_dir, mode, args.max_rounds)

    all_results = []
    completed_idxs = set()
    # Resume from existing results if available
    resume_path = getattr(args, 'resume', None)
    if resume_path and os.path.exists(resume_path):
        with open(resume_path) as f:
            prev = json.load(f)
        prev_results = prev.get("results", [])
        all_results = prev_results
        completed_idxs = {r["idx"] for r in prev_results}
        logger.info(f"Resuming from {resume_path}: {len(completed_idxs)} samples already done")
    total_start = time.time()

    global_index = None
    if getattr(args, 'global_index', False):
        # Check for cached global index FIRST — if it exists, load it and skip the
        # expensive corpus-wide NER pass entirely (that pass only feeds index BUILD).
        _syn_th = float(os.environ.get("SYNONYMY_THRESHOLD", "0.8"))
        _syn_tag = f"_syn{_syn_th}" if _syn_th != 0.8 else ""
        cache_path = os.path.join(save_dir, f"global_ner_index{_syn_tag}.pkl")
        override_path = os.environ.get("OVERRIDE_INDEX_PATH", "")
        if override_path and os.path.exists(override_path):
            logger.info(f"Loading OVERRIDE index from {override_path}")
            global_index = NERIndex.load(override_path, embedding_model=emb_model)
        elif os.path.exists(cache_path):
            logger.info(f"Loading cached GLOBAL NER index from {cache_path} (skipping corpus NER)")
            global_index = NERIndex.load(cache_path, embedding_model=emb_model)
        else:
            # No cached index — collect corpus docs and NER them, then build.
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

            logger.info(f"Building GLOBAL NER index for {len(all_docs)} passages...")
            global_index = NERIndex(embedding_model=emb_model)
            global_index.build(all_docs, global_ner_results)
            logger.info(f"  Global index: {global_index.graph.vcount()} nodes, {global_index.graph.ecount()} edges")
            global_index.save(cache_path)

        # Augment with cross-sentence relations if cache available
        cross_cache_path = getattr(args, 'cross_sentence_cache', None)
        if cross_cache_path and os.path.exists(cross_cache_path):
            logger.info(f"Loading cross-sentence cache: {cross_cache_path}")
            with open(cross_cache_path) as f:
                cross_cache = json.load(f)
            logger.info(f"  {len(cross_cache)} passages with cross-sentence relations")
            augmented_cache_path = cross_cache_path.replace('.json', '_augmented_index.pkl')
            if os.path.exists(augmented_cache_path):
                logger.info(f"Loading augmented index from {augmented_cache_path}")
                global_index = NERIndex.load(augmented_cache_path, embedding_model=emb_model)
            else:
                global_index.augment_cross_sentence(cross_cache)
                global_index.save(augmented_cache_path)
                logger.info(f"Saved augmented index to {augmented_cache_path}")

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

        if idx in completed_idxs:
            logger.info(f"[{idx+1}/{len(data)}] SKIP (already done)")
            continue

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

            ircot_answer = None
            ircot_last_round_docs = None
            if mode == 'ircot':
                retrieved_docs, ircot_last_round_docs, reasoning_traces, ircot_answer = ircot_retrieve(
                    index, question, llm_client, max_rounds=args.max_rounds)
            elif mode == 'self_ask':
                ircot_last_round_docs, retrieved_docs, reasoning_traces, ircot_answer = self_ask_retrieve(
                    index, question, llm_client, max_rounds=args.max_rounds)
            elif mode == 'iter_retgen':
                ircot_last_round_docs, retrieved_docs, reasoning_traces, ircot_answer = iter_retgen_retrieve(
                    index, question, llm_client, max_rounds=args.max_rounds)
            elif mode == 'iter_retgen_bridge':
                ircot_last_round_docs, retrieved_docs, reasoning_traces, ircot_answer = iter_retgen_bridge_retrieve(
                    index, question, llm_client, max_rounds=args.max_rounds)
            elif args.max_rounds > 1:
                retrieved_docs, reasoning_traces, round_diagnostics = iterative_retrieve(
                    index, question, llm_client, max_rounds=args.max_rounds, gold_docs=gold_docs,
                    query_entities=q_entities)
            else:
                sorted_doc_ids, sorted_doc_scores = index.retrieve(question, query_entities=q_entities)
                retrieved_docs = [index.passages[index.passage_keys[did]] for did in sorted_doc_ids]

            # For multi-round modes: R@5 measured on LAST round's 5 docs (not accumulated)
            recall_docs = ircot_last_round_docs if (mode in ('ircot', 'self_ask', 'iter_retgen', 'iter_retgen_bridge') and ircot_last_round_docs is not None) else retrieved_docs
            r1 = recall_at_k(recall_docs, gold_docs, 1)
            r2 = recall_at_k(recall_docs, gold_docs, 2)
            r5 = recall_at_k(recall_docs, gold_docs, 5)
            r10 = recall_at_k(recall_docs, gold_docs, 10)

            # For multi-round modes: use extracted answer if available, else fall back to llm_qa
            if mode in ('ircot', 'self_ask', 'iter_retgen', 'iter_retgen_bridge') and ircot_answer is not None:
                qa_answer = ircot_answer
            else:
                qa_top_k = int(os.getenv("QA_TOP_K", "15" if mode == 'ircot' else "5"))
                qa_answer = llm_qa(question, retrieved_docs[:qa_top_k], llm_client, reasoning_traces=reasoning_traces)
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
            "per_round_doc_scores": (
                getattr(iter_retgen_retrieve, '_last_per_round_doc_scores', None) if mode == 'iter_retgen'
                else getattr(iter_retgen_bridge_retrieve, '_last_per_round_doc_scores', None) if mode == 'iter_retgen_bridge'
                else None
            ),
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
                        choices=["base", "reasoning", "ircot", "self_ask", "iter_retgen", "iter_retgen_bridge"],
                        help="Reasoning mode: base, reasoning, ircot, self_ask, iter_retgen, iter_retgen_bridge")
    parser.add_argument("--global_index", action="store_true",
                        help="Build one global index from all samples' passages instead of per-sample")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to previous results JSON to resume from")
    parser.add_argument("--cross_sentence_cache", type=str, default=None,
                        help="Path to cross-sentence relations JSON for graph augmentation")
    args = parser.parse_args()
    # For reasoning/ircot, ensure max_rounds > 1
    if args.mode in ("reasoning", "ircot", "self_ask", "iter_retgen", "iter_retgen_bridge") and args.max_rounds <= 1:
        args.max_rounds = 3
    # Auto-enable HotpotQA-specific gap v2 variant (relaxed 4-hop constraints + 2-hop few-shot)
    if "hotpotqa" in os.path.basename(args.data_path).lower() and "HOTPOTQA_MODE" not in os.environ:
        os.environ["HOTPOTQA_MODE"] = "1"
        print("[INFO] auto-enabled HOTPOTQA_MODE=1 (REWRITE_SYSTEM_PROMPT_HOTPOTQA)")
    run_evaluation(args)


if __name__ == "__main__":
    main()
