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

QUERY_NER_SYSTEM = """Your task is to extract named entities from the given question.
Respond with a JSON object like {"named_entities": ["entity1", "entity2"]}."""

QUERY_NER_ONE_SHOT_INPUT = """When was Lady Godiva's birthplace abolished?"""
QUERY_NER_ONE_SHOT_OUTPUT = """{"named_entities": ["Lady Godiva"]}"""


def query_ner(question: str, llm_client) -> List[str]:
    """Extract named entities from a question using LLM."""
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
            return data.get("named_entities", [])
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

        # 3) Synonymy edges (entity-entity, embedding similarity > threshold)
        syn_threshold = 0.8
        if len(self.entity_keys) > 1 and self.entity_embeddings is not None and len(self.entity_embeddings) > 0:
            logger.info(f"  Computing synonymy edges (threshold={syn_threshold})...")
            norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normed = self.entity_embeddings / norms
            sim_matrix = normed @ normed.T

            n_before = len(edges)
            for i in range(len(self.entity_keys)):
                for j in range(i + 1, len(self.entity_keys)):
                    score = sim_matrix[i, j]
                    if score >= syn_threshold:
                        edges.append((i, j))
                        weights.append(float(score))
            logger.info(f"  {len(edges) - n_before} synonymy edges")

        if edges:
            self.graph.add_edges(edges)
            self.graph.es["weight"] = weights

    def retrieve(self, query: str, top_k: int = 5, passage_weight_scale: float = 0.05,
                 link_top_k: int = 5, pair_alpha: float = 0.5,
                 sent_alpha: float = 0.5,
                 entity_top_k: int = 5,
                 mmr_lambda: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve passages using pair matching + sentence matching → PPR.

        1. Pair matching: query vs pair embeddings ("entity1 | entity2 | sentence")
        2. Sentence matching: query vs sentence embeddings (pair's source sentence)
        3. MMR selection → top-k diverse pairs → extract entities as seeds
        4. Top-k entity filtering: only keep highest-weight entities, zero out rest
        5. Passage matching: auxiliary passage-level seeds
        6. PPR

        Returns: (sorted_doc_ids, sorted_doc_scores) indexing into self.passage_keys
        """
        # Encode query (instruction only for instruction-tuned models, e.g. GTE-Qwen2)
        # MiniLM is NOT instruction-tuned, so instruction='' to avoid polluting embeddings
        use_instruction = getattr(self.embedding_model, 'supports_instruction', False)
        sent_instr = QUERY_INSTRUCTION_SENTENCE if use_instruction else ''
        pass_instr = QUERY_INSTRUCTION_PASSAGE if use_instruction else ''
        query_emb_sentence = l2_normalize(self.embedding_model.batch_encode(
            [query], instruction=sent_instr))
        query_emb_passage = l2_normalize(self.embedding_model.batch_encode(
            [query], instruction=pass_instr))

        node_weights = np.zeros(self.graph.vcount())

        # Build sentence_id → index lookup
        sent_id_to_idx = {sid: i for i, sid in enumerate(self.sentence_ids)}

        # === Step 1: Pair matching + Sentence matching → combined score ===
        if self.pair_embeddings is not None and len(self.pair_embeddings) > 0:
            # Pair scores: query (fact instruction) vs pair embeddings
            pair_scores = (self.pair_embeddings @ query_emb_sentence.T).flatten()
            pair_scores = min_max_normalize(pair_scores)

            # Sentence scores for each pair's source sentence
            sent_scores_all = None
            if self.sentence_embeddings is not None and len(self.sentence_embeddings) > 0:
                sent_scores_all = (self.sentence_embeddings @ query_emb_sentence.T).flatten()
                sent_scores_all = min_max_normalize(sent_scores_all)

            # Combined score per pair
            combined_scores = np.zeros(len(self.pairs))
            for pi, (t1, k1, t2, k2, sent_id) in enumerate(self.pairs):
                p_score = pair_scores[pi]
                s_score = 0.0
                if sent_scores_all is not None:
                    si = sent_id_to_idx.get(sent_id, -1)
                    if si >= 0:
                        s_score = sent_scores_all[si]
                combined_scores[pi] = pair_alpha * p_score + sent_alpha * s_score

            # MMR-based diverse pair selection
            # Pre-compute pair embedding norms for similarity calculation
            pair_norms = np.linalg.norm(self.pair_embeddings, axis=1, keepdims=True)
            pair_norms = np.where(pair_norms == 0, 1, pair_norms)
            normed_pairs = self.pair_embeddings / pair_norms

            # Candidate pool: top 30 by combined score
            candidate_size = min(30, len(self.pairs))
            candidate_idxs = np.argsort(combined_scores)[::-1][:candidate_size].tolist()

            top_pair_idxs = []
            selected_embs = []
            for _ in range(link_top_k):
                if not candidate_idxs:
                    break
                if not selected_embs:
                    # First pick: highest combined score
                    best = candidate_idxs[0]
                else:
                    # MMR: balance relevance and diversity
                    selected_matrix = np.array(selected_embs)  # (n_selected, dim)
                    best_mmr = -float('inf')
                    best = candidate_idxs[0]
                    for ci in candidate_idxs:
                        relevance = combined_scores[ci]
                        # Max similarity to any already selected pair
                        sim = np.max(normed_pairs[ci] @ selected_matrix.T)
                        mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * sim
                        if mmr_score > best_mmr:
                            best_mmr = mmr_score
                            best = ci
                top_pair_idxs.append(best)
                selected_embs.append(normed_pairs[best])
                candidate_idxs.remove(best)
            ent_occur_count = np.zeros(self.graph.vcount())
            ent_max_score = np.zeros(self.graph.vcount())
            for pi in top_pair_idxs:
                t1, k1, t2, k2, sent_id = self.pairs[pi]
                score = combined_scores[pi]
                for ent_key in [k1, k2]:
                    ent_idx = self.node_name_to_idx.get(ent_key)
                    if ent_idx is not None:
                        n_passages = max(len(self.entity_to_passages[ent_key]), 1)
                        weighted_score = score / n_passages
                        node_weights[ent_idx] += weighted_score
                        ent_occur_count[ent_idx] += 1
                        if weighted_score > ent_max_score[ent_idx]:
                            ent_max_score[ent_idx] = weighted_score
            # (max + average) / 2
            for idx in self.entity_node_idxs:
                if ent_occur_count[idx] > 0:
                    avg = node_weights[idx] / ent_occur_count[idx]
                    node_weights[idx] = (ent_max_score[idx] + avg) / 2.0

        # === Step 2: Top-k entity filtering (like HippoRAG's get_top_k_weights) ===
        # Only keep entity_top_k highest-weight entity seeds, zero out the rest
        entity_weights = [(idx, node_weights[idx]) for idx in self.entity_node_idxs if node_weights[idx] > 0]
        if len(entity_weights) > entity_top_k:
            entity_weights.sort(key=lambda x: x[1], reverse=True)
            keep_idxs = set(idx for idx, _ in entity_weights[:entity_top_k])
            for idx in self.entity_node_idxs:
                if idx not in keep_idxs:
                    node_weights[idx] = 0.0

        # === Step 3: Passage matching (auxiliary, using passage instruction) ===
        if self.passage_embeddings is not None and len(self.passage_embeddings) > 0:
            passage_scores = (self.passage_embeddings @ query_emb_passage.T).flatten()
            passage_scores = min_max_normalize(passage_scores) * passage_weight_scale
            for i, p_idx in enumerate(self.passage_node_idxs):
                node_weights[p_idx] = passage_scores[i]

        if node_weights.sum() == 0:
            # Fallback: return passages by embedding similarity
            logger.warning("No seed weights, falling back to passage embedding")
            passage_scores = (self.passage_embeddings @ query_emb_passage.T).flatten()
            sorted_ids = np.argsort(passage_scores)[::-1]
            return sorted_ids, passage_scores[sorted_ids]

        # PPR (with NaN guard like HippoRAG)
        damping = 0.5
        node_weights = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        ppr_scores = self.graph.personalized_pagerank(
            vertices=range(self.graph.vcount()),
            damping=damping,
            directed=False,
            weights='weight' if 'weight' in self.graph.es.attributes() else None,
            reset=node_weights,
            implementation='prpack',
        )

        # Extract passage scores
        doc_scores = np.array([ppr_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids]

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
PIPELINE_RRF_WEIGHT = 0.5

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

def _resolve_entities_in_graph(index, entity_names: List[str], threshold: float = 0.6) -> Dict[str, int]:
    """Resolve entity names to graph vertex IDs. Exact match first, then embedding fallback."""
    resolved = {}
    unresolved = []

    for name in entity_names:
        ent_key = compute_hash(name.lower(), prefix="entity-")
        vid = index.node_name_to_idx.get(ent_key)
        if vid is not None:
            resolved[name] = vid
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
            if similarities[i][best_idx] >= threshold:
                ent_key = index.entity_keys[best_idx]
                vid = index.node_name_to_idx.get(ent_key)
                if vid is not None:
                    resolved[name] = vid
                    logger.info(f"  Entity '{name}' matched by embedding (sim={similarities[i][best_idx]:.3f}) -> vertex {vid}")
            else:
                logger.info(f"  Entity '{name}' NOT found in graph (best sim={similarities[i][best_idx]:.3f})")

    return resolved


def _degree_adaptive_weight(index, vid: int, base_weight: float) -> float:
    """Scale weight by log(degree) to resist dilution at high-degree nodes."""
    deg = index.graph.degree(vid)
    return base_weight * (1.0 + math.log(deg + 1))


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
) -> List[Tuple[int, int, float]]:
    """Run mini-PPR from bridge entities on local 5-hop subgraph.

    Returns list of (bridge_vid, seed_vid, ppr_score) for selected connections.
    """
    graph = index.graph
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


def _build_overlay_graph(index, temp_edges: List[Tuple[int, int, float]]):
    """Create a copy of the graph with temporary bridge edges added."""
    g = index.graph.copy()
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
                       gold_docs: List[str] = None):
    """Multi-round reasoning-guided retrieval with entity seed expansion + RRF.

    Ported from feature/graph-reshape ReasoningController._iterative_retrieve_single().

    Each round:
      1. Full retrieval with current query → weighted RRF
      2. If discovered entities exist: resolve → mini-PPR → bridge edges → expansion PPR → RRF
      3. LLM reasoning → rewrite query + discover bridge entities
      4. LLM reasoning reranker on final docs

    Returns: (retrieved_docs, reasoning_traces)
    """
    num_docs = len(index.passage_keys)
    rrf_k = 60
    rrf_scores = np.zeros(num_docs)
    current_query = question
    reasoning_traces = []
    round_trajectories = []  # full trajectory per round for debugging
    all_discovered = {}  # name -> vertex_id, accumulated across rounds
    base_node_weights = None
    round0_top_doc_ids = None

    for round_i in range(max_rounds):
        logger.info(f"  Round {round_i}: query='{current_query[:60]}...'")

        # Step 1: Full retrieval with current query
        sorted_doc_ids, sorted_doc_scores = index.retrieve(current_query)

        # Save base node weights from retrieve (reconstruct from retrieve's internal state)
        # We need the node_weights for expansion PPR — re-derive them
        if round_i == 0:
            # Get node weights by running the seed computation part of retrieve
            base_node_weights = _compute_base_node_weights(index, current_query)
            round0_top_doc_ids = [int(d) for d in sorted_doc_ids[:20]]

        # Step 2: Weighted RRF accumulate (later rounds weighted higher)
        round_weight = 1.0 + round_i * RRF_ROUND_BOOST
        for rank, doc_id in enumerate(sorted_doc_ids):
            rrf_scores[doc_id] += PIPELINE_RRF_WEIGHT * round_weight / (rrf_k + rank + 1)

        # Step 3: If we have discovered entities, run expansion PPR
        if all_discovered and base_node_weights is not None:
            existing_seeds = _get_existing_seed_ids(index, base_node_weights)

            # Mini-PPR filtered bridge edges
            ppr_connections = _mini_ppr_select_seeds(
                index,
                discovered_vertex_ids=all_discovered,
                existing_seed_vertex_ids=existing_seeds,
                round0_top_doc_ids=round0_top_doc_ids,
            )

            modified_weights = base_node_weights.copy()
            for name, vid in all_discovered.items():
                modified_weights[vid] += _degree_adaptive_weight(index, vid, DEFAULT_ENTITY_SEED_WEIGHT)

            # Build overlay graph with bridge edges
            temp_edges = []
            selected_pairs = set()

            # Bridge → seed edges (from mini-PPR)
            for d_vid, s_vid, score in ppr_connections:
                bridge_weight = _degree_adaptive_weight(index, d_vid, 1.0)
                temp_edges.append((d_vid, s_vid, bridge_weight))
                selected_pairs.add((d_vid, s_vid))

            # Inter-discovered edges (always connect bridge entities to each other)
            d_list = list(all_discovered.values())
            for i, v1 in enumerate(d_list):
                for v2 in d_list[i+1:]:
                    w = max(_degree_adaptive_weight(index, v1, 1.0),
                            _degree_adaptive_weight(index, v2, 1.0))
                    temp_edges.append((v1, v2, w))

            working_graph = _build_overlay_graph(index, temp_edges) if temp_edges else index.graph

            # Run expansion PPR
            modified_weights = np.where(np.isnan(modified_weights) | (modified_weights < 0), 0, modified_weights)
            if modified_weights.sum() > 0:
                # Pad modified_weights if overlay graph has more vertices
                if working_graph.vcount() > len(modified_weights):
                    padded = np.zeros(working_graph.vcount())
                    padded[:len(modified_weights)] = modified_weights
                    modified_weights = padded

                boosted_ppr = working_graph.personalized_pagerank(
                    vertices=range(working_graph.vcount()),
                    damping=EXPANSION_DAMPING,
                    directed=False,
                    weights='weight' if 'weight' in working_graph.es.attributes() else None,
                    reset=modified_weights,
                    implementation='prpack',
                )

                # Extract passage scores and rank
                boosted_doc_scores = np.array([boosted_ppr[idx] for idx in index.passage_node_idxs])
                boosted_doc_ids = np.argsort(boosted_doc_scores)[::-1]

                # Expansion PPR RRF (weight 1.0, vs pipeline 0.5)
                for rank, doc_id in enumerate(boosted_doc_ids):
                    rrf_scores[doc_id] += round_weight / (rrf_k + rank + 1)

                logger.info(
                    f"  Expansion PPR: {len(all_discovered)} entities, "
                    f"bridge edges: {len(selected_pairs)}, temp edges: {len(temp_edges)}, "
                    f"round_weight: {round_weight:.1f}"
                )

        # Current top docs from RRF
        final_sorted_ids = np.argsort(rrf_scores)[::-1]
        top_docs = [index.passages[index.passage_keys[idx]] for idx in final_sorted_ids[:10]]

        # Last round: skip reasoning
        if round_i == max_rounds - 1:
            break

        # Step 4: LLM reasoning with entity discovery
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

        # Save full trajectory for this round
        round_traj = {
            "round": round_i,
            "query": current_query,
            "analysis": reasoning_output.get("analysis", ""),
            "rewritten_query": reasoning_output.get("rewritten_query", ""),
            "discovered_entities": reasoning_output.get("discovered_entities", []),
            "should_stop": reasoning_output.get("should_stop", False),
            "resolved_entities": {},
            "top_doc_ids": [int(d) for d in final_sorted_ids[:10]],
        }

        if reasoning_output.get("should_stop", False):
            logger.info(f"  Reasoning stop at round {round_i}")
            round_trajectories.append(round_traj)
            break

        # Query rewrite
        new_query = reasoning_output.get("rewritten_query", "")
        if new_query and new_query != current_query:
            logger.info(f"  Round {round_i} rewrite: '{new_query[:60]}...'")
            current_query = new_query

        # Entity seed expansion: resolve discovered entities to graph vertices
        discovered_entities = reasoning_output.get("discovered_entities", [])
        if discovered_entities:
            logger.info(f"  Discovered entities: {discovered_entities}")
            resolved = _resolve_entities_in_graph(index, discovered_entities)
            all_discovered.update(resolved)
            logger.info(f"  Total resolved entities: {len(all_discovered)}")
            round_traj["resolved_entities"] = {name: int(vid) for name, vid in resolved.items()}

        round_trajectories.append(round_traj)

    # Final ranking from RRF
    final_sorted_ids = np.argsort(rrf_scores)[::-1]
    retrieved_docs = [index.passages[index.passage_keys[idx]] for idx in final_sorted_ids]

    # LLM reasoning reranker: reorder docs based on reasoning trajectory
    if reasoning_traces:
        reranked = _llm_reasoning_rerank(question, retrieved_docs[:10], reasoning_traces, llm_client)
        if reranked is not None:
            # Replace top docs with reranked, keep rest
            retrieved_docs = reranked + retrieved_docs[len(reranked):]

    return retrieved_docs, reasoning_traces, round_trajectories


def _compute_base_node_weights(index, query: str) -> np.ndarray:
    """Re-derive the base node weights that retrieve() would compute.

    This mirrors the seed computation in NERIndex.retrieve() so we can
    use the weights for expansion PPR in later rounds.
    """
    use_instruction = getattr(index.embedding_model, 'supports_instruction', False)
    sent_instr = QUERY_INSTRUCTION_SENTENCE if use_instruction else ''
    pass_instr = QUERY_INSTRUCTION_PASSAGE if use_instruction else ''
    query_emb_sentence = l2_normalize(index.embedding_model.batch_encode(
        [query], instruction=sent_instr))
    query_emb_passage = l2_normalize(index.embedding_model.batch_encode(
        [query], instruction=pass_instr))

    node_weights = np.zeros(index.graph.vcount())
    sent_id_to_idx = {sid: i for i, sid in enumerate(index.sentence_ids)}

    if index.pair_embeddings is not None and len(index.pair_embeddings) > 0:
        pair_scores = (index.pair_embeddings @ query_emb_sentence.T).flatten()
        pair_scores = min_max_normalize(pair_scores)

        sent_scores_all = None
        if index.sentence_embeddings is not None and len(index.sentence_embeddings) > 0:
            sent_scores_all = (index.sentence_embeddings @ query_emb_sentence.T).flatten()
            sent_scores_all = min_max_normalize(sent_scores_all)

        combined_scores = np.zeros(len(index.pairs))
        for pi, (t1, k1, t2, k2, sent_id) in enumerate(index.pairs):
            p_score = pair_scores[pi]
            s_score = 0.0
            if sent_scores_all is not None:
                si = sent_id_to_idx.get(sent_id, -1)
                if si >= 0:
                    s_score = sent_scores_all[si]
            combined_scores[pi] = 0.5 * p_score + 0.5 * s_score

        # MMR selection (same as retrieve)
        pair_norms = np.linalg.norm(index.pair_embeddings, axis=1, keepdims=True)
        pair_norms = np.where(pair_norms == 0, 1, pair_norms)
        normed_pairs = index.pair_embeddings / pair_norms

        candidate_size = min(30, len(index.pairs))
        candidate_idxs = np.argsort(combined_scores)[::-1][:candidate_size].tolist()

        top_pair_idxs = []
        selected_embs = []
        link_top_k = 5
        mmr_lambda = 0.7
        for _ in range(link_top_k):
            if not candidate_idxs:
                break
            if not selected_embs:
                best = candidate_idxs[0]
            else:
                selected_matrix = np.array(selected_embs)
                best_mmr = -float('inf')
                best = candidate_idxs[0]
                for ci in candidate_idxs:
                    relevance = combined_scores[ci]
                    sim = np.max(normed_pairs[ci] @ selected_matrix.T)
                    mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * sim
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best = ci
            top_pair_idxs.append(best)
            selected_embs.append(normed_pairs[best])
            candidate_idxs.remove(best)

        ent_occur_count = np.zeros(index.graph.vcount())
        ent_max_score = np.zeros(index.graph.vcount())
        for pi in top_pair_idxs:
            t1, k1, t2, k2, sent_id = index.pairs[pi]
            score = combined_scores[pi]
            for ent_key in [k1, k2]:
                ent_idx = index.node_name_to_idx.get(ent_key)
                if ent_idx is not None:
                    n_passages = max(len(index.entity_to_passages[ent_key]), 1)
                    weighted_score = score / n_passages
                    node_weights[ent_idx] += weighted_score
                    ent_occur_count[ent_idx] += 1
                    if weighted_score > ent_max_score[ent_idx]:
                        ent_max_score[ent_idx] = weighted_score

        for idx in index.entity_node_idxs:
            if ent_occur_count[idx] > 0:
                avg = node_weights[idx] / ent_occur_count[idx]
                node_weights[idx] = (ent_max_score[idx] + avg) / 2.0

        # Top-k entity filtering
        entity_top_k = 5
        entity_weights = [(idx, node_weights[idx]) for idx in index.entity_node_idxs if node_weights[idx] > 0]
        if len(entity_weights) > entity_top_k:
            entity_weights.sort(key=lambda x: x[1], reverse=True)
            keep_idxs = set(idx for idx, _ in entity_weights[:entity_top_k])
            for idx in index.entity_node_idxs:
                if idx not in keep_idxs:
                    node_weights[idx] = 0.0

    # Passage weights
    if index.passage_embeddings is not None and len(index.passage_embeddings) > 0:
        passage_scores = (index.passage_embeddings @ query_emb_passage.T).flatten()
        passage_scores = min_max_normalize(passage_scores) * 0.05
        for i, p_idx in enumerate(index.passage_node_idxs):
            node_weights[p_idx] = passage_scores[i]

    return node_weights


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
        self.client = OpenAI(base_url=base_url, timeout=60)
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

    data = json.load(open(args.data_path))
    if args.sample_limit and args.sample_limit < len(data):
        data = data[:args.sample_limit]
    logger.info(f"Loaded {len(data)} samples")

    # Models
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

    logger.info(f"Setting up LLM: {llm_model_name}")
    llm_client = SimpleLLM(
        model_name=llm_model_name,
        base_url=aliyun_base_url,
    )

    # Load NER cache if available
    ner_cache_path = args.ner_cache
    ner_cache = {}
    if ner_cache_path and os.path.exists(ner_cache_path):
        logger.info(f"Loading NER cache: {ner_cache_path}")
        ner_cache = json.load(open(ner_cache_path))
        logger.info(f"  {len(ner_cache)} cached entries")

    # Also try to extract entities from existing OpenIE cache
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

            # Extract entities from OpenIE results
            entities = doc.get("extracted_entities", [])
            if not entities and doc.get("extracted_triples"):
                # Fallback: extract from triples
                ent_set = set()
                for t in doc["extracted_triples"]:
                    if len(t) >= 3:
                        ent_set.add(t[0])
                        ent_set.add(t[2])
                entities = sorted(ent_set)
            openie_cache[text_key] = entities
        logger.info(f"  {len(openie_cache)} docs with entities from OpenIE")

    # Output
    save_dir = "outputs/musique_ner_pipeline_eval"
    os.makedirs(save_dir, exist_ok=True)
    mode = getattr(args, 'mode', 'base')
    if mode == 'ircot':
        output_tag = f"_ircot_rounds{args.max_rounds}"
    elif args.max_rounds > 1:
        output_tag = f"_rounds{args.max_rounds}"
    else:
        output_tag = ""
    output_path = os.path.join(save_dir, f"comparison_results{output_tag}.json")

    # Load previous results for retry mode
    prev_results = {}
    if getattr(args, 'retry_from', None) and os.path.exists(args.retry_from):
        prev_data = json.load(open(args.retry_from))
        for r in prev_data.get("results", []):
            if r.get("ner_answer") != "Error":
                prev_results[r["idx"]] = r
        logger.info(f"Retry mode: loaded {len(prev_results)} successful results from {args.retry_from}")
        failed_idxs = [r["idx"] for r in prev_data.get("results", []) if r.get("ner_answer") == "Error"]
        logger.info(f"  Will rerun {len(failed_idxs)} failed samples: {failed_idxs}")

    all_results = []
    total_start = time.time()

    for idx, sample in enumerate(data):
        # Skip successful samples in retry mode
        if idx in prev_results:
            all_results.append(prev_results[idx])
            continue

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
        gold_answers = [answer] + answer_aliases

        logger.info(f"[{idx+1}/{len(data)}] {question[:80]}...")

        try:
            # Step 1: Get NER results for each doc
            ner_results = {}
            for doc_text in docs:
                if doc_text in ner_cache:
                    ner_results[doc_text] = ner_cache[doc_text]
                elif doc_text in openie_cache:
                    ner_results[doc_text] = openie_cache[doc_text]
                else:
                    # Run LLM NER
                    entities = llm_ner(doc_text, llm_client)
                    ner_results[doc_text] = entities
                    ner_cache[doc_text] = entities

            # Step 2: Build index
            index = NERIndex(embedding_model=emb_model)
            index.build(docs, ner_results)

            # Step 3: Retrieve
            reasoning_traces = []
            round_trajectories = []
            if mode == 'ircot':
                # IRCoT: iterative retrieval with chain-of-thought
                retrieved_docs, reasoning_traces = ircot_retrieve(
                    index, question, llm_client, max_rounds=args.max_rounds)
            elif args.max_rounds > 1:
                # Graph-reshape reasoning: entity discovery + seed expansion + RRF
                retrieved_docs, reasoning_traces, round_trajectories = iterative_retrieve(
                    index, question, llm_client, max_rounds=args.max_rounds)
            else:
                # Single-round base retrieval
                sorted_doc_ids, sorted_doc_scores = index.retrieve(question)
                retrieved_docs = [index.passages[index.passage_keys[did]] for did in sorted_doc_ids]

            # Step 4: Compute recall
            r1 = recall_at_k(retrieved_docs, gold_docs, 1)
            r2 = recall_at_k(retrieved_docs, gold_docs, 2)
            r5 = recall_at_k(retrieved_docs, gold_docs, 5)
            r10 = recall_at_k(retrieved_docs, gold_docs, 10)

            # Step 5: QA
            qa_answer = llm_qa(question, retrieved_docs[:5], llm_client)

            em_correct = check_em(qa_answer, answer, answer_aliases)
            f1_score = check_f1(qa_answer, answer, answer_aliases)

        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            retrieved_docs = []
            reasoning_traces = []
            r1 = r2 = r5 = r10 = 0.0
            qa_answer = "Error"
            em_correct = False
            f1_score = 0.0

        result = {
            "idx": idx,
            "question": question,
            "gold_answer": answer,
            "gold_aliases": answer_aliases,
            "ner_answer": qa_answer,
            "ner_em": int(em_correct),
            "ner_f1": round(f1_score, 4),
            "ner_recall": {"R@1": r1, "R@2": r2, "R@5": r5, "R@10": r10},
            "reasoning_traces": reasoning_traces,
            "round_trajectories": round_trajectories,
            "n_entities": len(index.entity_keys) if hasattr(index, 'entity_keys') else 0,
            "n_sentences": len(index.sentences) if hasattr(index, 'sentences') else 0,
            "graph_nodes": index.graph.vcount() if index.graph else 0,
            "graph_edges": index.graph.ecount() if index.graph else 0,
        }
        all_results.append(result)

        em_str = "Y" if em_correct else "N"
        logger.info(f"  EM={em_str} F1={f1_score:.2f} R@5={r5:.2f} | '{qa_answer[:50]}' | Gold='{answer}'")

        # Cleanup
        del index
        gc.collect()

        # Save progress every 5 samples
        if (idx + 1) % 5 == 0 or (idx + 1) == len(data):
            n = len(all_results)
            avg_em = sum(r["ner_em"] for r in all_results) / n
            avg_f1 = sum(r["ner_f1"] for r in all_results) / n
            avg_r5 = sum(r["ner_recall"]["R@5"] for r in all_results) / n
            logger.info(f"  >>> Progress: {n}/{len(data)} | EM={avg_em:.3f} | F1={avg_f1:.3f} | R@5={avg_r5:.3f}")

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
                        "ner_em": round(avg_em, 4),
                        "ner_f1": round(avg_f1, 4),
                        "ner_recall": {
                            "R@1": round(sum(r["ner_recall"]["R@1"] for r in all_results) / n, 4),
                            "R@2": round(sum(r["ner_recall"]["R@2"] for r in all_results) / n, 4),
                            "R@5": round(avg_r5, 4),
                            "R@10": round(sum(r["ner_recall"]["R@10"] for r in all_results) / n, 4),
                        },
                    },
                    "results": all_results,
                }, f, indent=2)

    # Save NER cache
    if ner_cache:
        cache_out = os.path.join(save_dir, "ner_cache.json")
        with open(cache_out, "w") as f:
            json.dump(ner_cache, f)
        logger.info(f"Saved NER cache: {cache_out} ({len(ner_cache)} entries)")

    # Final summary
    n = len(all_results)
    avg_em = sum(r["ner_em"] for r in all_results) / n
    avg_f1 = sum(r["ner_f1"] for r in all_results) / n
    avg_r5 = sum(r["ner_recall"]["R@5"] for r in all_results) / n
    avg_r1 = sum(r["ner_recall"]["R@1"] for r in all_results) / n

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"NER Pipeline Results: {n} samples")
    print(f"  EM:   {avg_em:.4f}")
    print(f"  F1:   {avg_f1:.4f}")
    print(f"  R@1:  {avg_r1:.4f}")
    print(f"  R@5:  {avg_r5:.4f}")
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
    parser.add_argument("--retry_from", type=str, default=None,
                        help="Path to previous results JSON. Rerun only failed samples (ner_answer='Error') and keep successful ones.")
    args = parser.parse_args()
    # For reasoning/ircot, ensure max_rounds > 1
    if args.mode in ("reasoning", "ircot") and args.max_rounds <= 1:
        args.max_rounds = 3
    run_evaluation(args)


if __name__ == "__main__":
    main()
