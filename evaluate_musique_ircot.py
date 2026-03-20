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
    save_dir = "outputs/musique_ircot_eval"
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "method": "ircot",
        "max_rounds": args.max_rounds,
        "sample_limit": args.sample_limit,
        "llm": llm_model_name,
        "embedding": embedding_model_name,
        "openie_cache": args.openie_cache,
    }

    # Load OpenIE cache
    openie_text_lookup = {}
    if args.openie_cache:
        logger.info(f"Using pre-computed OpenIE cache: {args.openie_cache}")
        cache_data = json.load(open(args.openie_cache))
        for doc in cache_data["docs"]:
            if "text" in doc:
                text_key = doc["text"]
            else:
                parts = doc["passage"].split("\n", 1)
                text_key = parts[1] if len(parts) > 1 else doc["passage"]
            openie_text_lookup[text_key] = doc
        logger.info(f"Loaded {len(openie_text_lookup)} docs from OpenIE cache")

    output_path = os.path.join(save_dir, "comparison_results.json")
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
        docs = [para.get("paragraph_text", "") for para in paragraphs]
        gold_docs_list = [
            para.get("paragraph_text", "")
            for para in paragraphs
            if para.get("is_supporting", False)
        ]
        gold_answers = [answer] + answer_aliases

        # Use shared directory for graph building (all experiments share the same graphs)
        per_sample_dir = os.path.join(shared_graph_dir, f"sample_{idx:06d}")
        logger.info(f"[{idx+1}/{len(data)}] {question[:80]}...")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(SAMPLE_TIMEOUT)

        try:
            hipporag = HippoRAG(
                save_dir=per_sample_dir,
                llm_model_name=llm_model_name,
                embedding_model_name=embedding_model_name,
                llm_base_url=aliyun_base_url,
                embedding_model=shared_embedding_model,
            )
            # Inject OpenIE cache
            if openie_text_lookup:
                os.makedirs(per_sample_dir, exist_ok=True)
                cache_dest = hipporag.openie_results_path
                matched_docs = []
                for doc_text in docs:
                    if doc_text in openie_text_lookup:
                        cached = openie_text_lookup[doc_text]
                        new_idx = compute_mdhash_id(doc_text, prefix="chunk-")
                        matched_docs.append({
                            "idx": new_idx,
                            "passage": doc_text,
                            "extracted_entities": cached["extracted_entities"],
                            "extracted_triples": cached.get("extracted_triples", []),
                        })
                if matched_docs:
                    with open(cache_dest, "w") as f:
                        json.dump({"docs": matched_docs}, f)
                    # Invalidate stale graph.pickle if OpenIE cache is newer
                    graph_pickle = hipporag._graph_pickle_filename
                    if os.path.exists(graph_pickle):
                        if os.path.getmtime(cache_dest) > os.path.getmtime(graph_pickle):
                            os.remove(graph_pickle)
                            logger.debug(f"  Removed stale graph.pickle (older than OpenIE cache)")
            hipporag.index(docs=docs)
        except TimeoutError:
            logger.error(f"  Sample {idx} timed out during indexing")
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            all_results.append({
                "idx": idx, "question": question, "gold_answer": answer,
                "gold_aliases": answer_aliases,
                "baseline_answer": "Error", "ircot_answer": "Error",
                "baseline_retrieval": {}, "ircot_retrieval": {},
                "baseline_qa": {}, "ircot_qa": {},
                "error": "Timeout during indexing",
            })
            gc.collect()
            continue
        except Exception as e:
            logger.error(f"  Index failed: {e}")
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            all_results.append({
                "idx": idx, "question": question, "gold_answer": answer,
                "gold_aliases": answer_aliases,
                "baseline_answer": "Error", "ircot_answer": "Error",
                "baseline_retrieval": {}, "ircot_retrieval": {},
                "baseline_qa": {}, "ircot_qa": {},
                "error": str(e),
            })
            gc.collect()
            continue

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

        # --- IRCoT ---
        t0 = time.time()
        try:
            ircot_solution, thoughts, ircot_rounds = ircot_retrieve(
                hipporag, question, docs, max_rounds=args.max_rounds,
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

        del hipporag
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
