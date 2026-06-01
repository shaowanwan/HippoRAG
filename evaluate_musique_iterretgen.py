"""
Evaluate IRCoT baseline on MuSiQue dataset.

IRCoT: iterative retrieval with chain-of-thought.
Each round: retrieve docs → LLM generates one thought → append thought to query → re-retrieve.
Stops when LLM outputs "So the answer is:" or max rounds reached.

Compares:
  1. Baseline HippoRAG (single-pass retrieval + QA)
  2. HippoRAG + IRCoT (multi-round, original IRCoT reasoning)

Usage:
    .venv/bin/python evaluate_musique_ircot.py --data_path musique.json --sample_limit 200 --max_rounds 3
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
import re as _re

import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("__main__").setLevel(logging.INFO)
logging.getLogger("src.hipporag.HippoRAG").setLevel(logging.INFO)  # see graph build progress
logger = logging.getLogger(__name__)


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
    from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
    return HippoRAG, QuerySolution, compute_mdhash_id, _get_embedding_model_class, PromptTemplateManager, RetrievalRecall, QAExactMatch, QAF1Score


def reason_step_fixed(dataset, prompt_template_manager, query, passages, thoughts, llm_client):
    """Fixed version of reason_step that handles infer() returning a plain string.

    Original reason_step in qa_utils.py has a bug: it does response_message[0]["content"]
    but CacheOpenAI.infer() returns (str, dict), not (List[dict], dict).
    """
    from src.hipporag.utils.qa_utils import merge_elements_with_same_first_line

    prompt_user = ''
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    messages = prompt_template_manager.render(name=f'ircot_{dataset}', prompt_user=prompt_user)

    try:
        # CacheOpenAI.infer() returns (message, metadata, cache_hit) due to @cache_response decorator
        result = llm_client.infer(messages=messages)
        response_message = result[0]
        if isinstance(response_message, str):
            return response_message
        return response_message[0]["content"]
    except Exception as e:
        logger.exception(f"reason_step LLM call failed: {e}")
        return ''


def save_results(output_path, config, all_results):
    n = len(all_results)
    if n == 0:
        return

    baseline_em = sum(
        1 for r in all_results
        if check_em(r["baseline_answer"], r["gold_answer"], r.get("gold_aliases", []))
    ) / n
    ircot_em = sum(
        1 for r in all_results
        if check_em(r["ircot_answer"], r["gold_answer"], r.get("gold_aliases", []))
    ) / n

    # F1
    baseline_f1_vals = [r.get("baseline_qa", {}).get("F1") for r in all_results]
    ircot_f1_vals = [r.get("ircot_qa", {}).get("F1") for r in all_results]
    baseline_f1_vals = [v for v in baseline_f1_vals if v is not None]
    ircot_f1_vals = [v for v in ircot_f1_vals if v is not None]
    baseline_f1 = round(float(np.mean(baseline_f1_vals)), 4) if baseline_f1_vals else None
    ircot_f1 = round(float(np.mean(ircot_f1_vals)), 4) if ircot_f1_vals else None

    # Recall@k
    recall_keys = ["Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20"]
    baseline_recall = {}
    ircot_recall = {}
    for key in recall_keys:
        b_vals = [r.get("baseline_retrieval", {}).get(key) for r in all_results]
        b_vals = [v for v in b_vals if v is not None]
        i_vals = [r.get("ircot_retrieval", {}).get(key) for r in all_results]
        i_vals = [v for v in i_vals if v is not None]
        if b_vals:
            baseline_recall[key] = round(float(np.mean(b_vals)), 4)
        if i_vals:
            ircot_recall[key] = round(float(np.mean(i_vals)), 4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": config,
                "summary": {
                    "n_completed": n,
                    "baseline_em": round(baseline_em, 4),
                    "ircot_em": round(ircot_em, 4),
                    "improvement_em": round(ircot_em - baseline_em, 4),
                    "baseline_f1": baseline_f1,
                    "ircot_f1": ircot_f1,
                    "improvement_f1": round(ircot_f1 - baseline_f1, 4) if baseline_f1 and ircot_f1 else None,
                    "baseline_recall": baseline_recall,
                    "ircot_recall": ircot_recall,
                },
                "results": all_results,
            },
            f,
            indent=2,
            default=make_serializable,
        )


def ircot_retrieve(hipporag, question, docs, max_rounds=3, reason_step_fn=None, prompt_template_mgr=None):
    """Run IRCoT: iterative retrieval with chain-of-thought.

    Each round:
    1. Retrieve with current query (original question + accumulated thoughts)
    2. LLM generates one thought based on retrieved docs
    3. If thought contains "So the answer is:", stop
    4. Otherwise, append thought and re-retrieve

    Returns: (final_docs, final_scores, thoughts, round_metrics)
    """
    thoughts = []
    round_metrics_list = []
    prompt_template_manager = prompt_template_mgr

    for round_i in range(max_rounds):
        # Build query: original question + all previous thoughts
        current_query = question
        if thoughts:
            current_query = question + " " + " ".join(thoughts)

        # Retrieve using HippoRAG's full pipeline
        if not hipporag.ready_to_retrieve:
            hipporag.prepare_retrieval_objects()

        results = hipporag.retrieve(queries=[current_query])
        if isinstance(results, tuple):
            query_solutions, _ = results
        else:
            query_solutions = results

        retrieved_docs = query_solutions[0].docs[:10]
        retrieved_scores = query_solutions[0].doc_scores[:10] if query_solutions[0].doc_scores is not None else []

        # Record round metrics
        round_metrics_list.append({
            "round": round_i,
            "query": current_query[:100],
            "n_thoughts": len(thoughts),
        })

        # Generate next thought using IRCoT
        thought = reason_step_fn(
            dataset="musique",
            prompt_template_manager=prompt_template_manager,
            query=question,
            passages=retrieved_docs[:5],
            thoughts=thoughts,
            llm_client=hipporag.llm_model,
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
    results = hipporag.retrieve(queries=[final_query])
    if isinstance(results, tuple):
        query_solutions, _ = results
    else:
        query_solutions = results

    final_solution = query_solutions[0]
    return final_solution, thoughts, round_metrics_list


# ──────────────────────────────────────────────────────────────────────
# IRCoT EXP1 (NER-pipeline-aligned variant)
# Mirrors evaluate_musique_ner_pipeline.py's EXP1 config:
#   - prompt: 1-shot Stanton + 2-shot MC Eiht with next-fact hint
#   - query_mode = last_sentence
#   - doc_mode   = accumulate (cap 15)
#   - per_round_k = 6
#   - stop signal: "so the answer is:" in thought, extract with regex
# Differences from Trivedi 2023 original: documented in PAPER (next-fact hint added,
# retrieval is HippoRAG NER+PPR not BM25, LLM is qwen-plus not GPT-3.5).
# ──────────────────────────────────────────────────────────────────────
IRCOT_EXP1_ONE_SHOT_DOCS = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students.\n\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton.\n\n"""
)
IRCOT_EXP1_ONE_SHOT_DEMO = (
    f'{IRCOT_EXP1_ONE_SHOT_DOCS}\n\nQuestion: '
    "When was Neville A. Stanton's employer founded?\nThought: "
    "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862.\n\n"
)
IRCOT_EXP1_TWO_SHOT_DOCS = (
    """Wikipedia Title: Smoke in tha City\nSmoke in tha City is the ninth studio album by American rapper MC Eiht. The album was released in 2004 on Tha Hall Records.\n\n"""
    """Wikipedia Title: MC Eiht\nAaron Tyler (born May 22, 1971), better known by his stage name MC Eiht, is an American rapper born in Compton, California.\n\n"""
    """Wikipedia Title: Compton, California\nCompton is a city in southern Los Angeles County, California.\n\n"""
)
IRCOT_EXP1_TWO_SHOT_DEMO = (
    f'{IRCOT_EXP1_TWO_SHOT_DOCS}\n\nQuestion: '
    "In what county is the birthplace of the performer of Smoke in tha City?\nThought: "
    "The performer of Smoke in tha City is MC Eiht. MC Eiht was born in Compton, California. Compton is located in Los Angeles County. So the answer is: Los Angeles County.\n\n"
)
IRCOT_EXP1_SYSTEM = (
    "You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. "
    "This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. "
    "Your task is to generate one thought for current step, DON'T generate the whole thoughts at once! "
    "Only conclude with \"So the answer is:\" when the final answer is explicitly supported by the retrieved passages above; "
    "if the passages don't contain enough evidence, output the next fact you need to find so the system can retrieve more."
    "\n\n"
    f"{IRCOT_EXP1_ONE_SHOT_DEMO}"
    f"{IRCOT_EXP1_TWO_SHOT_DEMO}"
)


def ircot_retrieve_exp1(hipporag, question, max_rounds=3):
    """IRCoT EXP1 on HippoRAG (mirrors NER+IRCoT EXP1 config).

    Round i:
      1. query = original question (round 0) | last_sentence(thoughts[-1]) (round 1+)
      2. retrieve top-6 docs via HippoRAG (NER+PPR)
      3. accumulate unique docs into reader prompt (cap 15)
      4. LLM generates one thought based on accumulated docs (uses EXP1 prompt)
      5. if "so the answer is:" in thought, extract answer via regex, stop

    Final answer = regex extract from stop-thought (or None if hit max_rounds).

    Returns: (final_solution_for_recall, last_round_docs, thoughts, ircot_answer)
    """
    import re as _re
    per_round_k = 6
    max_paras = 15

    thoughts = []
    accumulated_docs = []
    accumulated_set = set()
    last_solution = None
    last_round_docs = []
    ircot_answer = None

    for round_i in range(max_rounds):
        if round_i == 0 or not thoughts:
            current_query = question
        else:
            sents = _re.split(r'(?<=[.!?])\s+', thoughts[-1].strip())
            current_query = sents[-1] if sents else thoughts[-1]

        if not hipporag.ready_to_retrieve:
            hipporag.prepare_retrieval_objects()
        results = hipporag.retrieve(queries=[current_query])
        if isinstance(results, tuple):
            query_solutions, _ = results
        else:
            query_solutions = results
        last_solution = query_solutions[0]

        # last_round_docs = top-per_round_k for this round only
        last_round_docs = list(last_solution.docs[:per_round_k]) if last_solution.docs else []
        # Accumulate unique docs up to max_paras
        for doc in last_round_docs:
            if len(accumulated_docs) >= max_paras:
                break
            if doc not in accumulated_set:
                accumulated_set.add(doc)
                accumulated_docs.append(doc)

        # Generate thought using EXP1 prompt + accumulated docs
        prompt_user = ''
        for passage in accumulated_docs:
            prompt_user += f'{passage}\n\n'
        prompt_user += f'Question: {question}\nThought:' + ' '.join(thoughts)

        messages = [
            {"role": "system", "content": IRCOT_EXP1_SYSTEM},
            {"role": "user", "content": prompt_user},
        ]
        try:
            resp = hipporag.llm_model.infer(messages=messages)
            if isinstance(resp, tuple):
                resp = resp[0]
            if not isinstance(resp, str):
                resp = resp[0]["content"] if isinstance(resp, list) else str(resp)
            thought = resp.strip()
        except Exception as e:
            logger.warning(f"  IRCoT-EXP1 reason_step failed: {e}")
            thought = ''

        thoughts.append(thought)
        logger.info(
            f"  IRCoT-EXP1 round {round_i}: acc={len(accumulated_docs)} last={len(last_round_docs)} thought='{thought[:60]}...'"
        )

        if "so the answer is:" in thought.lower():
            m = _re.search(
                r'so the answer is:?\s*(.+?)(?:\.\s*$|\n|$)',
                thought, _re.IGNORECASE | _re.DOTALL)
            if m:
                ircot_answer = m.group(1).strip().strip('"').strip("'").rstrip('.').strip()
            logger.info(f"  IRCoT-EXP1 stopped at round {round_i}, extracted: '{ircot_answer}'")
            break

        if len(accumulated_docs) >= max_paras:
            logger.info(f"  IRCoT-EXP1 cap reached at {max_paras}, stopping early")
            break

    return last_solution, last_round_docs, thoughts, ircot_answer


def iter_retgen_retrieve(hipporag, question, max_rounds=3):
    """ITER-RETGEN on HippoRAG.

    Each round:
    1. Retrieve with query = original_question + " " + last_generation (only last, per Shao 2023)
    2. Run hipporag.rag_qa on retrieved docs to get a full Thought+Answer response
    3. Use this response as the next round's query expansion

    Final answer = last round's extracted answer (from QuerySolution.answer).
    No early stop; runs fixed max_rounds.

    Returns: (final_solution, generations, final_answer, round_metrics)
    """
    generations = []
    round_metrics_list = []
    final_solution = None
    final_answer = ""

    for round_i in range(max_rounds):
        current_query = question
        if generations:
            current_query = question + " " + generations[-1]

        if not hipporag.ready_to_retrieve:
            hipporag.prepare_retrieval_objects()

        results = hipporag.retrieve(queries=[current_query])
        if isinstance(results, tuple):
            query_solutions, _ = results
        else:
            query_solutions = results
        final_solution = query_solutions[0]

        qa_out = hipporag.rag_qa(queries=[final_solution])
        response_text = qa_out[1][0] if len(qa_out) > 1 and qa_out[1] else ""
        round_answer = qa_out[0][0].answer if qa_out[0] else ""

        generations.append(response_text)
        final_answer = round_answer

        round_metrics_list.append({
            "round": round_i,
            "query_len": len(current_query),
            "n_docs": len(final_solution.docs) if final_solution.docs else 0,
            "answer": round_answer[:60],
        })
        logger.info(f"  ITER-RETGEN round {round_i}: ans='{round_answer[:60]}'")

    return final_solution, generations, final_answer, round_metrics_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="musique.json")
    parser.add_argument("--sample_limit", type=int, default=200)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--openie_cache", type=str, default="outputs/musique/openie_results_ner_qwen-plus.json")
    args = parser.parse_args()

    # Import inside main to avoid multiprocessing spawn issue on macOS
    (HippoRAG, QuerySolution, compute_mdhash_id, _get_embedding_model_class,
     PromptTemplateManager, RetrievalRecall, QAExactMatch, QAF1Score) = _lazy_imports()

    # Set random seeds for reproducibility
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

    prompt_template_manager = PromptTemplateManager()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        os.environ["OPENAI_API_KEY"] = "sk-396199ed7af84eff8a0cf7a71b797601"

    # Shared sample directory for graph building — keyed by embedding model
    emb_key = embedding_model_name.split("/")[-1].replace(" ", "_")
    shared_graph_dir = os.path.join("outputs", f"musique_shared_{emb_key}")
    # Save dir keyed by dataset (musique vs hotpotqa) to avoid overwriting across data_path
    _data_name_save = os.path.splitext(os.path.basename(args.data_path))[0]
    save_dir = f"outputs/{_data_name_save}_ircot_eval"
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "method": (
            "iter_retgen" if os.getenv("ITER_RETGEN_MODE", "0") == "1"
            else "ircot_exp1" if os.getenv("IRCOT_EXP1_MODE", "0") == "1"
            else "ircot"
        ),
        "max_rounds": args.max_rounds,
        "sample_limit": args.sample_limit,
        "llm": llm_model_name,
        "embedding": embedding_model_name,
        "openie_cache": args.openie_cache,
    }

    # Load OpenIE cache
    # Key by full passage text ("title\ntext") to match the corpus doc format used downstream.
    # Refactor 2026-05-13: previously keyed by text-only (after split), which broke after we
    # started passing "title\ntext" docs to HippoRAG.index() to match evaluate_musique_reasoning.py.
    openie_text_lookup = {}
    openie_nows_lookup = {}  # whitespace-stripped fallback (HotpotQA: corpus.json single-space vs cache double-space)
    if args.openie_cache:
        logger.info(f"Using pre-computed OpenIE cache: {args.openie_cache}")
        cache_data = json.load(open(args.openie_cache))
        for doc in cache_data["docs"]:
            text_key = doc["passage"]  # full "title\ntext" — matches docs passed to hipporag.index()
            openie_text_lookup[text_key] = doc
            openie_nows_lookup[_re.sub(r"\s+", "",text_key)] = doc
        logger.info(f"Loaded {len(openie_text_lookup)} docs from OpenIE cache")

    _iter_retgen_mode = os.getenv("ITER_RETGEN_MODE", "0") == "1"
    _ircot_exp1_mode = os.getenv("IRCOT_EXP1_MODE", "0") == "1"
    if _iter_retgen_mode and _ircot_exp1_mode:
        raise ValueError("Cannot enable both ITER_RETGEN_MODE and IRCOT_EXP1_MODE")
    output_path = os.path.join(
        save_dir,
        "comparison_results_iterretgen.json" if _iter_retgen_mode else
        "comparison_results_ircot_exp1.json" if _ircot_exp1_mode else
        "comparison_results.json",
    )
    all_results = []

    # Resume
    if os.path.exists(output_path):
        try:
            existing = json.load(open(output_path))
            all_results = existing.get("results", [])
            logger.info(f"Resuming from {len(all_results)} completed samples")
        except Exception:
            all_results = []

    # Pre-load embedding model
    logger.info(f"Pre-loading embedding model: {embedding_model_name}")
    shared_embedding_model = _get_embedding_model_class(
        embedding_model_name=embedding_model_name
    )(embedding_model_name=embedding_model_name)
    logger.info("Embedding model loaded")

    # ── FULL-CORPUS HIPPORAG SETUP (one-time, NOT per-sample) ──
    # Load full MuSiQue corpus (11656 passages) — required for fair comparison with
    # NER pipeline --global_index, eval_ircot_baseline.py, and HippoRAG paper setup.
    # Auto-detect corpus path based on data_path (musique → musique_corpus, hotpotqa → hotpotqa_corpus)
    _data_name = os.path.splitext(os.path.basename(args.data_path))[0]
    corpus_path = os.getenv("CORPUS_PATH", f"reproduce/dataset/{_data_name}_corpus.json")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Full corpus not found: {corpus_path}. Cannot run paper-quality baseline. Set CORPUS_PATH env var.")
    corpus_data = json.load(open(corpus_path))
    # ⚠️ MUST include title in doc text: "{title}\n{text}"
    # OpenIE cache was extracted on title-prefixed passages; chunk_embedding must match.
    # Compare evaluate_musique_reasoning.py:290 (working baseline with R@5=0.711 on GTE).
    full_corpus_docs = [f"{c['title']}\n{c['text']}" for c in corpus_data]
    logger.info(f"Loaded full corpus: {len(full_corpus_docs)} passages from {corpus_path}")
    # Whitespace-stripped lookup so per-sample gold paragraphs (HotpotQA: double-space) map
    # to their corpus-canonical form (single-space). Without this, gold_docs_list never
    # matches retrieved docs → R@5=0 even when correct doc retrieved.
    corpus_nows_to_canonical = {_re.sub(r"\s+", "", d): d for d in full_corpus_docs}

    # Global HippoRAG save_dir (keyed by LLM + embedding so different setups don't collide)
    llm_safe = llm_model_name.replace("/", "_").replace(" ", "_")
    _graph_basename = os.getenv("HIPPORAG_GRAPH_NAME", _data_name)
    global_hipporag_dir = os.path.join("outputs", f"{_graph_basename}_hipporag_fullcorpus_{llm_safe}_{emb_key}")
    os.makedirs(global_hipporag_dir, exist_ok=True)
    logger.info(f"Global HippoRAG save_dir: {global_hipporag_dir}")

    # Build HippoRAG ONCE on full corpus
    logger.info("Building global HippoRAG instance (one-time corpus indexing)...")
    hipporag = HippoRAG(
        save_dir=global_hipporag_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url=aliyun_base_url,
        embedding_model=shared_embedding_model,
    )

    # Inject OpenIE cache (matched to full corpus) — must do BEFORE hipporag.index()
    if openie_text_lookup:
        cache_dest = hipporag.openie_results_path
        matched_docs = []
        exact_hits = 0
        nows_hits = 0
        for doc_text in full_corpus_docs:
            cached = openie_text_lookup.get(doc_text)
            if cached is not None:
                exact_hits += 1
            else:
                # Fallback: whitespace-stripped match. HotpotQA corpus.json was normalized
                # (single-space, no space before quotes); OpenIE cache built from raw
                # hotpotqa.json paragraph_text preserves double-spaces. Punctuation+letters
                # are identical so stripping whitespace recovers 100% match.
                cached = openie_nows_lookup.get(_re.sub(r"\s+", "",doc_text))
                if cached is not None:
                    nows_hits += 1
            if cached is not None:
                new_idx = compute_mdhash_id(doc_text, prefix="chunk-")
                matched_docs.append({
                    "idx": new_idx,
                    "passage": doc_text,
                    "extracted_entities": cached["extracted_entities"],
                    "extracted_triples": cached.get("extracted_triples", []),
                })
        logger.info(f"  Matched {len(matched_docs)}/{len(full_corpus_docs)} corpus docs to OpenIE cache (exact={exact_hits}, nows-fallback={nows_hits})")
        if matched_docs:
            with open(cache_dest, "w") as f:
                json.dump({"docs": matched_docs}, f)
            graph_pickle = hipporag._graph_pickle_filename
            if os.path.exists(graph_pickle):
                if os.path.getmtime(cache_dest) > os.path.getmtime(graph_pickle):
                    os.remove(graph_pickle)
                    logger.info("  Removed stale graph.pickle (older than OpenIE cache)")

    logger.info(f"Indexing full corpus ({len(full_corpus_docs)} docs) — builds graph + entity embeddings ONCE...")
    hipporag.index(docs=full_corpus_docs)
    logger.info("Global HippoRAG ready. Starting per-sample query evaluation.")

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
        # Full-corpus refactor: HippoRAG indexed ONCE outside loop, queries retrieve
        # from full 11656-passage global index. Per-sample paragraphs only used for
        # gold_docs labeling (which docs are "supporting") and gold_answers.
        # ⚠️ gold_docs MUST use same "title\ntext" format as the indexed docs, otherwise
        # recall metric string-compares fails (LLM may answer correctly but R@5=0).
        # HotpotQA: per-sample paragraph_text has double-space; corpus.json was normalized
        # to single-space. Look up canonical corpus version by whitespace-stripped match.
        gold_docs_list = []
        for para in paragraphs:
            if not para.get("is_supporting", False):
                continue
            raw = f"{para.get('title', '')}\n{para.get('paragraph_text', '')}"
            canonical = corpus_nows_to_canonical.get(_re.sub(r"\s+", "", raw), raw)
            gold_docs_list.append(canonical)
        gold_answers = [answer] + answer_aliases
        logger.info(f"[{idx+1}/{len(data)}] {question[:80]}...")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(SAMPLE_TIMEOUT)

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

        # --- IRCoT or ITER-RETGEN (mutually exclusive; skippable via SKIP_IRCOT=1) ---
        _skip_ircot = os.getenv("SKIP_IRCOT", "0") == "1"
        t0 = time.time()
        if _skip_ircot:
            ircot_answer = ""
            ircot_retrieval_eval = {}
            ircot_qa_eval = {}
            ircot_recall = {}
            thoughts = []
            ircot_rounds = []
        else:
         try:
            if _iter_retgen_mode:
                ircot_solution, thoughts, iter_final_answer, ircot_rounds = iter_retgen_retrieve(
                    hipporag, question, max_rounds=args.max_rounds,
                )
                ircot_answer = iter_final_answer
                ircot_qa_eval = {}
                recall_evaluator = RetrievalRecall(global_config=hipporag.global_config)
                ircot_recall, _ = recall_evaluator.calculate_metric_scores(
                    gold_docs=[gold_docs_list],
                    retrieved_docs=[ircot_solution.docs],
                    k_list=[1, 2, 5, 10, 20],
                )
                ircot_retrieval_eval = {}
            elif _ircot_exp1_mode:
                ircot_solution, last_round_docs, thoughts, exp1_answer = ircot_retrieve_exp1(
                    hipporag, question, max_rounds=args.max_rounds,
                )
                # If LLM never said "so the answer is:", fallback: take last thought text
                ircot_answer = exp1_answer if exp1_answer else (thoughts[-1] if thoughts else "Unknown")
                ircot_qa_eval = {}
                # R@k measured on LAST ROUND docs (consistent with EXP1's last-round-doc semantics
                # would be confusing — EXP1 uses accumulate; we use accumulated_docs for R@k)
                # Use last_solution.docs which contains the last-round retrieval result (top-k from final retrieve call)
                recall_evaluator = RetrievalRecall(global_config=hipporag.global_config)
                ircot_recall, _ = recall_evaluator.calculate_metric_scores(
                    gold_docs=[gold_docs_list],
                    retrieved_docs=[ircot_solution.docs],
                    k_list=[1, 2, 5, 10, 20],
                )
                ircot_retrieval_eval = {}
            else:
                ircot_solution, thoughts, ircot_rounds = ircot_retrieve(
                    hipporag, question, None, max_rounds=args.max_rounds,
                    reason_step_fn=reason_step_fixed, prompt_template_mgr=prompt_template_manager,
                )

                # QA with IRCoT retrieved docs
                ircot_qa_result = hipporag.rag_qa(
                    queries=[ircot_solution],
                    gold_docs=[gold_docs_list],
                    gold_answers=[gold_answers],
                )
                ircot_answer = ircot_qa_result[0][0].answer if ircot_qa_result[0] else "Unknown"
                ircot_retrieval_eval = ircot_qa_result[3] if len(ircot_qa_result) > 3 else {}
                ircot_qa_eval = ircot_qa_result[4] if len(ircot_qa_result) > 4 else {}

                # Compute recall for IRCoT retrieval
                recall_evaluator = RetrievalRecall(global_config=hipporag.global_config)
                ircot_recall, _ = recall_evaluator.calculate_metric_scores(
                    gold_docs=[gold_docs_list],
                    retrieved_docs=[ircot_solution.docs],
                    k_list=[1, 2, 5, 10, 20],
                )
         except TimeoutError:
            logger.error(f"  IRCoT timed out for sample {idx}")
            ircot_answer = "Error"
            ircot_retrieval_eval = {}
            ircot_qa_eval = {}
            ircot_recall = {}
            thoughts = []
            ircot_rounds = []
         except Exception as e:
            logger.error(f"  IRCoT failed: {e}")
            logger.error(traceback.format_exc())
            ircot_answer = "Error"
            ircot_retrieval_eval = {}
            ircot_qa_eval = {}
            ircot_recall = {}
            thoughts = []
            ircot_rounds = []
        ircot_time = time.time() - t0

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        result = {
            "idx": idx,
            "question": question,
            "gold_answer": answer,
            "gold_aliases": answer_aliases,
            "baseline_answer": baseline_answer,
            "ircot_answer": ircot_answer,
            "baseline_retrieval": baseline_retrieval,
            "ircot_retrieval": ircot_recall,
            "baseline_qa": baseline_qa,
            "ircot_qa": ircot_qa_eval,
            "ircot_thoughts": thoughts,
            "ircot_rounds": len(thoughts),
            "baseline_time": round(baseline_time, 2),
            "ircot_time": round(ircot_time, 2),
        }
        all_results.append(result)

        # Note: hipporag is now a global instance (full-corpus refactor) — do NOT del it
        gc.collect()

        b_match = "Y" if check_em(baseline_answer, answer, answer_aliases) else "N"
        i_match = "Y" if check_em(ircot_answer, answer, answer_aliases) else "N"
        logger.info(
            f"  B={b_match} '{baseline_answer[:40]}' | I={i_match} '{ircot_answer[:40]}' | Gold='{answer}' | "
            f"t={baseline_time:.1f}s/{ircot_time:.1f}s"
        )

        if (idx + 1) % 5 == 0:
            save_results(output_path, config, all_results)
            n = len(all_results)
            b_em = sum(1 for r in all_results if check_em(r["baseline_answer"], r["gold_answer"], r.get("gold_aliases", []))) / n
            i_em = sum(1 for r in all_results if check_em(r["ircot_answer"], r["gold_answer"], r.get("gold_aliases", []))) / n
            elapsed = time.time() - total_start
            eta = elapsed / n * (len(data) - n)
            logger.info(f"  >>> Progress: {n}/{len(data)} | Baseline EM={b_em:.3f} | IRCoT EM={i_em:.3f} | ETA={eta/60:.0f}min")

    save_results(output_path, config, all_results)

    # Summary
    n = len(all_results)
    if n > 0:
        b_em = sum(1 for r in all_results if check_em(r["baseline_answer"], r["gold_answer"], r.get("gold_aliases", []))) / n
        i_em = sum(1 for r in all_results if check_em(r["ircot_answer"], r["gold_answer"], r.get("gold_aliases", []))) / n
        print(f"\n{'='*60}")
        print(f"IRCoT Evaluation Complete: {n} samples")
        print(f"  Baseline EM: {b_em:.4f}")
        print(f"  IRCoT EM:    {i_em:.4f}")
        print(f"  Delta:       {i_em - b_em:+.4f}")
        print(f"  Results: {output_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
