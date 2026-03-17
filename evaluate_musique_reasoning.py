"""
Evaluate reasoning-guided iterative retrieval vs baseline on MuSiQue dataset.

Compares:
  1. Baseline HippoRAG (single-pass)
  2. Reasoning HippoRAG (multi-round, with query rewriting + graph reshaping)

Usage:
    python evaluate_musique_reasoning.py --data_path musique.json --sample_limit 200 --max_rounds 3
"""
import json
import os
import sys
import random
import argparse
import logging
import signal
import time
import traceback
import gc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
# Only show INFO for our key modules
logging.getLogger("__main__").setLevel(logging.INFO)
logging.getLogger("src.hipporag.reasoning.controller").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def load_musique_data(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


import re as _re

def normalize_answer(s):
    """Normalize answer for EM comparison: lowercase, strip punctuation/articles/whitespace."""
    s = s.lower().strip()
    # Remove punctuation
    s = _re.sub(r'[^\w\s]', '', s)
    # Remove articles
    s = _re.sub(r'\b(a|an|the)\b', ' ', s)
    # Collapse whitespace
    s = ' '.join(s.split())
    return s


def em_match(pred, gold):
    """Check if prediction matches gold after normalization."""
    return normalize_answer(pred) == normalize_answer(gold)


def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def check_em(pred, gold_answer, gold_aliases):
    """Check EM against gold answer and all aliases."""
    all_golds = [gold_answer] + (gold_aliases if gold_aliases else [])
    return any(em_match(pred, g) for g in all_golds)


def save_results(output_path, config, all_results):
    """Incrementally save results to disk."""
    n = len(all_results)
    if n == 0:
        return

    baseline_em = sum(
        1 for r in all_results
        if check_em(r["baseline_answer"], r["gold_answer"], r.get("gold_aliases", []))
    ) / n
    reasoning_em = sum(
        1 for r in all_results
        if check_em(r["reasoning_answer"], r["gold_answer"], r.get("gold_aliases", []))
    ) / n

    # Compute average retrieval metrics across reasoning rounds
    avg_rounds = np.mean([
        r.get("reasoning_retrieval", {}).get("avg_rounds_used", 1)
        for r in all_results
    ])

    # Aggregate Recall@k from retrieval eval (final)
    recall_keys = ["Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20"]
    baseline_recall = {}
    reasoning_recall = {}
    for key in recall_keys:
        # baseline_retrieval is flat, reasoning_retrieval has "final" nested
        b_vals = []
        for r in all_results:
            br = r.get("baseline_retrieval", {})
            val = br.get("final", {}).get(key) if "final" in br else br.get(key)
            if val is not None:
                b_vals.append(val)
        r_vals = [r.get("reasoning_retrieval", {}).get("final", {}).get(key)
                  for r in all_results]
        r_vals = [v for v in r_vals if v is not None]
        if b_vals:
            baseline_recall[key] = round(float(np.mean(b_vals)), 4)
        if r_vals:
            reasoning_recall[key] = round(float(np.mean(r_vals)), 4)

    # Aggregate F1 from QA eval (final)
    baseline_f1_vals = [r.get("baseline_qa", {}).get("F1")
                        for r in all_results]
    reasoning_f1_vals = [r.get("reasoning_qa", {}).get("F1")
                         for r in all_results]
    baseline_f1_vals = [v for v in baseline_f1_vals if v is not None]
    reasoning_f1_vals = [v for v in reasoning_f1_vals if v is not None]
    baseline_f1 = round(float(np.mean(baseline_f1_vals)), 4) if baseline_f1_vals else None
    reasoning_f1 = round(float(np.mean(reasoning_f1_vals)), 4) if reasoning_f1_vals else None

    # Aggregate per-round QA metrics (EM, F1, Recall per round)
    per_round_summary = {}
    max_round_idx = 0
    for r in all_results:
        per_round_qa = r.get("reasoning_retrieval", {}).get("per_round_qa", {})
        for round_key in per_round_qa:
            ridx = int(round_key.split("_")[1])
            if ridx > max_round_idx:
                max_round_idx = ridx

    for ridx in range(max_round_idx + 1):
        round_key = f"round_{ridx}"
        round_em_vals = []
        round_f1_vals = []
        round_recall_vals = {k: [] for k in recall_keys}

        for r in all_results:
            per_round_qa = r.get("reasoning_retrieval", {}).get("per_round_qa", {})
            if round_key in per_round_qa:
                rdata = per_round_qa[round_key]
                if "ExactMatch" in rdata:
                    round_em_vals.append(rdata["ExactMatch"])
                if "F1" in rdata:
                    round_f1_vals.append(rdata["F1"])
                for k in recall_keys:
                    if k in rdata:
                        round_recall_vals[k].append(rdata[k])

        round_summary = {"n_queries": len(round_em_vals)}
        if round_em_vals:
            round_summary["ExactMatch"] = round(float(np.mean(round_em_vals)), 4)
        if round_f1_vals:
            round_summary["F1"] = round(float(np.mean(round_f1_vals)), 4)
        for k in recall_keys:
            if round_recall_vals[k]:
                round_summary[k] = round(float(np.mean(round_recall_vals[k])), 4)

        per_round_summary[round_key] = round_summary

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": config,
                "summary": {
                    "n_completed": n,
                    "baseline_em": round(baseline_em, 4),
                    "reasoning_em": round(reasoning_em, 4),
                    "improvement_em": round(reasoning_em - baseline_em, 4),
                    "baseline_f1": baseline_f1,
                    "reasoning_f1": reasoning_f1,
                    "improvement_f1": round(reasoning_f1 - baseline_f1, 4) if baseline_f1 and reasoning_f1 else None,
                    "baseline_recall": baseline_recall,
                    "reasoning_recall": reasoning_recall,
                    "per_round": per_round_summary,
                    "avg_reasoning_rounds": round(float(avg_rounds), 2),
                },
                "results": all_results,
            },
            f,
            indent=2,
            default=make_serializable,
        )


def run_evaluation(data, sample_limit, max_rounds, openie_cache=None):
    # Import inside function to avoid multiprocessing spawn issue on macOS
    from src.hipporag import HippoRAG
    import shutil

    if sample_limit and sample_limit < len(data):
        data = data[:sample_limit]

    base_save_dir = "outputs/musique_reasoning_eval"
    llm_model_name = os.getenv("LLM_MODEL_NAME", "qwen-plus")
    embedding_model_name = os.getenv(
        "EMBEDDING_MODEL_NAME", "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    )
    aliyun_base_url = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # Separate output dirs per config to avoid cache conflicts
    # Default MiniLM + qwen-plus uses base dir for backward compatibility
    default_emb = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    default_llm = "qwen-plus"
    if embedding_model_name == default_emb and llm_model_name == default_llm:
        save_dir = base_save_dir
    else:
        llm_short = llm_model_name.split("/")[-1].replace(" ", "_")
        emb_short = embedding_model_name.split("/")[-1].replace(" ", "_")
        save_dir = os.path.join(base_save_dir, f"{llm_short}__{emb_short}")
    os.makedirs(save_dir, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        # Use the key from the existing test file as fallback
        os.environ["OPENAI_API_KEY"] = "sk-396199ed7af84eff8a0cf7a71b797601"

    config = {
        "max_rounds": max_rounds,
        "sample_limit": sample_limit,
        "llm": llm_model_name,
        "embedding": embedding_model_name,
        "openie_cache": openie_cache,
    }

    # Load pre-computed OpenIE cache and build text->openie_info lookup
    openie_text_lookup = {}
    if openie_cache:
        logger.info(f"Using pre-computed OpenIE cache: {openie_cache}")
        if not os.path.isfile(openie_cache):
            raise FileNotFoundError(f"OpenIE cache not found: {openie_cache}")
        from src.hipporag.utils.misc_utils import compute_mdhash_id
        cache_data = json.load(open(openie_cache))
        for doc in cache_data["docs"]:
            # Cache may have "text" field (gpt-4o-mini) or only "passage" (qwen-plus)
            # passage format is "title\ntext", so extract text after first newline
            if "text" in doc:
                text_key = doc["text"]
            else:
                parts = doc["passage"].split("\n", 1)
                text_key = parts[1] if len(parts) > 1 else doc["passage"]
            openie_text_lookup[text_key] = doc
        logger.info(f"Loaded {len(openie_text_lookup)} docs from OpenIE cache")
    output_path = os.path.join(save_dir, "comparison_results.json")
    all_results = []

    # Check for existing progress
    if os.path.exists(output_path):
        try:
            existing = json.load(open(output_path))
            all_results = existing.get("results", [])
            logger.info(f"Resuming from {len(all_results)} completed samples")
        except Exception:
            all_results = []

    # Per-sample timeout (seconds) to prevent hanging on API calls
    SAMPLE_TIMEOUT = 300  # 5 minutes per sample

    def _sample_timeout_handler(signum, frame):
        raise TimeoutError("Sample timed out")

    total_start = time.time()

    for idx, sample in enumerate(data):
        # Skip already completed samples
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

        per_sample_dir = os.path.join(save_dir, f"sample_{idx:06d}")

        logger.info(f"[{idx+1}/{len(data)}] {question[:80]}...")

        # Set per-sample timeout to prevent hanging on API calls
        old_handler = signal.signal(signal.SIGALRM, _sample_timeout_handler)
        signal.alarm(SAMPLE_TIMEOUT)

        try:
            hipporag = HippoRAG(
                save_dir=per_sample_dir,
                llm_model_name=llm_model_name,
                embedding_model_name=embedding_model_name,
                llm_base_url=aliyun_base_url,
            )
            # Inject pre-computed OpenIE cache with correct chunk hashes
            if openie_text_lookup:
                os.makedirs(per_sample_dir, exist_ok=True)
                cache_dest = hipporag.openie_results_path
                if True:  # Always overwrite to ensure consistent OpenIE
                    matched_docs = []
                    for doc_text in docs:
                        if doc_text in openie_text_lookup:
                            cached = openie_text_lookup[doc_text]
                            # Recompute idx using the actual text (no title prefix)
                            # to match what HippoRAG's embedding_store will generate
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
                        logger.debug(f"  Injected {len(matched_docs)}/{len(docs)} cached OpenIE results")
            hipporag.index(docs=docs)
        except TimeoutError:
            logger.error(f"  Sample {idx} timed out during indexing")
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            all_results.append({
                "idx": idx, "question": question, "gold_answer": answer,
                "gold_aliases": answer_aliases,
                "baseline_answer": "Error", "reasoning_answer": "Error",
                "baseline_retrieval": {}, "reasoning_retrieval": {},
                "baseline_qa": {}, "reasoning_qa": {},
                "baseline_time": 0, "reasoning_time": 0, "error": "Timeout during indexing",
            })
            try:
                del hipporag
            except NameError:
                pass
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            continue
        except Exception as e:
            logger.error(f"  Index failed: {e}")
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            all_results.append({
                "idx": idx, "question": question, "gold_answer": answer,
                "baseline_answer": "Error", "reasoning_answer": "Error",
                "baseline_retrieval": {}, "reasoning_retrieval": {},
                "baseline_qa": {}, "reasoning_qa": {},
                "baseline_time": 0, "reasoning_time": 0, "error": str(e),
            })
            try:
                del hipporag
            except NameError:
                pass
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            continue

        # --- Baseline ---
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
        except TimeoutError:
            logger.error(f"  Baseline timed out for sample {idx}")
            baseline_answer = "Error"
            baseline_retrieval = {}
            baseline_qa = {}
        except Exception as e:
            logger.error(f"  Baseline failed: {e}")
            baseline_answer = "Error"
            baseline_retrieval = {}
            baseline_qa = {}
        baseline_time = time.time() - t0

        # Reset alarm for reasoning phase
        signal.alarm(SAMPLE_TIMEOUT)

        # --- Reasoning ---
        t0 = time.time()
        try:
            reasoning = hipporag.reasoning_rag_qa(
                queries=[question],
                gold_docs=[gold_docs_list],
                gold_answers=[gold_answers],
                max_rounds=max_rounds,
            )
            reasoning_answer = reasoning[0][0].answer if reasoning[0] else "Unknown"
            reasoning_retrieval = reasoning[3] if len(reasoning) > 3 else {}
            reasoning_qa = reasoning[4] if len(reasoning) > 4 else {}
        except TimeoutError:
            logger.error(f"  Reasoning timed out for sample {idx}")
            reasoning_answer = "Error"
            reasoning_retrieval = {}
            reasoning_qa = {}
        except Exception as e:
            logger.error(f"  Reasoning failed: {e}")
            logger.error(traceback.format_exc())
            reasoning_answer = "Error"
            reasoning_retrieval = {}
            reasoning_qa = {}
        reasoning_time = time.time() - t0

        # Clear alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        result = {
            "idx": idx,
            "question": question,
            "gold_answer": answer,
            "gold_aliases": answer_aliases,
            "baseline_answer": baseline_answer,
            "reasoning_answer": reasoning_answer,
            "baseline_retrieval": baseline_retrieval,
            "reasoning_retrieval": reasoning_retrieval,
            "baseline_qa": baseline_qa,
            "reasoning_qa": reasoning_qa,
            "baseline_time": round(baseline_time, 2),
            "reasoning_time": round(reasoning_time, 2),
        }
        all_results.append(result)

        # Free GPU memory from this sample's HippoRAG instance
        del hipporag
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Quick EM check
        b_match = "Y" if check_em(baseline_answer, answer, answer_aliases) else "N"
        r_match = "Y" if check_em(reasoning_answer, answer, answer_aliases) else "N"

        logger.info(
            f"  B={b_match} '{baseline_answer[:40]}' | R={r_match} '{reasoning_answer[:40]}' | Gold='{answer}' | "
            f"t={baseline_time:.1f}s/{reasoning_time:.1f}s"
        )

        # Save progress every 5 samples
        if (idx + 1) % 5 == 0:
            save_results(output_path, config, all_results)
            n = len(all_results)
            b_em = sum(1 for r in all_results if check_em(r["baseline_answer"], r["gold_answer"], r.get("gold_aliases", []))) / n
            r_em = sum(1 for r in all_results if check_em(r["reasoning_answer"], r["gold_answer"], r.get("gold_aliases", []))) / n
            elapsed = time.time() - total_start
            eta = elapsed / n * (len(data) - n)
            logger.info(f"  >>> Progress: {n}/{len(data)} | Baseline EM={b_em:.3f} | Reasoning EM={r_em:.3f} | ETA={eta/60:.0f}min")

    # Final save
    save_results(output_path, config, all_results)

    # --- Summary ---
    n = len(all_results)
    baseline_em = sum(
        1 for r in all_results
        if check_em(r["baseline_answer"], r["gold_answer"], r.get("gold_aliases", []))
    ) / n
    reasoning_em = sum(
        1 for r in all_results
        if check_em(r["reasoning_answer"], r["gold_answer"], r.get("gold_aliases", []))
    ) / n

    total_time = time.time() - total_start

    # Recompute per-round summary for printing
    recall_keys = ["Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20"]

    # Recompute per-round metrics for printing
    max_round_idx = 0
    for r in all_results:
        per_round_qa = r.get("reasoning_retrieval", {}).get("per_round_qa", {})
        for round_key in per_round_qa:
            ridx = int(round_key.split("_")[1])
            if ridx > max_round_idx:
                max_round_idx = ridx

    per_round_print = {}
    for ridx in range(max_round_idx + 1):
        round_key = f"round_{ridx}"
        round_em_vals = []
        round_f1_vals = []
        round_recall_vals = {k: [] for k in recall_keys}

        for r in all_results:
            per_round_qa = r.get("reasoning_retrieval", {}).get("per_round_qa", {})
            if round_key in per_round_qa:
                rdata = per_round_qa[round_key]
                if "ExactMatch" in rdata:
                    round_em_vals.append(rdata["ExactMatch"])
                if "F1" in rdata:
                    round_f1_vals.append(rdata["F1"])
                for k in recall_keys:
                    if k in rdata:
                        round_recall_vals[k].append(rdata[k])

        per_round_print[round_key] = {
            "n": len(round_em_vals),
            "EM": np.mean(round_em_vals) if round_em_vals else float('nan'),
            "F1": np.mean(round_f1_vals) if round_f1_vals else float('nan'),
            "recall": {k: np.mean(v) if v else float('nan') for k, v in round_recall_vals.items()},
        }

    # Baseline recall/F1 for comparison
    # baseline_retrieval is flat (Recall@k at top level), not nested under "final"
    b_recall_print = {}
    for key in recall_keys:
        b_vals = []
        for r in all_results:
            br = r.get("baseline_retrieval", {})
            val = br.get("final", {}).get(key) if "final" in br else br.get(key)
            if val is not None:
                b_vals.append(val)
        if b_vals:
            b_recall_print[key] = np.mean(b_vals)

    b_f1_vals = [r.get("baseline_qa", {}).get("F1") for r in all_results]
    b_f1_vals = [v for v in b_f1_vals if v is not None]
    baseline_f1_print = np.mean(b_f1_vals) if b_f1_vals else float('nan')

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Samples: {n}")
    print(f"Max reasoning rounds: {max_rounds}")

    # Per-round table
    print(f"\n{'':=<70}")
    print(f"{'Round':<10} {'N':>5} {'EM':>8} {'F1':>8} {'R@1':>8} {'R@2':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8}")
    print(f"{'':=<70}")

    # Baseline row
    print(f"{'Baseline':<10} {n:>5} {baseline_em:>8.4f} {baseline_f1_print:>8.4f}", end="")
    for key in recall_keys:
        val = b_recall_print.get(key, float('nan'))
        print(f" {val:>8.4f}", end="")
    print()

    # Per-round rows
    for ridx in range(max_round_idx + 1):
        round_key = f"round_{ridx}"
        rd = per_round_print[round_key]
        print(f"{'Round ' + str(ridx):<10} {rd['n']:>5} {rd['EM']:>8.4f} {rd['F1']:>8.4f}", end="")
        for key in recall_keys:
            val = rd["recall"].get(key, float('nan'))
            print(f" {val:>8.4f}", end="")
        print()

    # Final row (reasoning final)
    r_f1_vals = [r.get("reasoning_qa", {}).get("F1") for r in all_results]
    r_f1_vals = [v for v in r_f1_vals if v is not None]
    r_f1_print = np.mean(r_f1_vals) if r_f1_vals else float('nan')
    r_recall_print = {}
    for key in recall_keys:
        r_vals = [r.get("reasoning_retrieval", {}).get("final", {}).get(key)
                  for r in all_results]
        r_vals = [v for v in r_vals if v is not None]
        if r_vals:
            r_recall_print[key] = np.mean(r_vals)

    print(f"{'Final':<10} {n:>5} {reasoning_em:>8.4f} {r_f1_print:>8.4f}", end="")
    for key in recall_keys:
        val = r_recall_print.get(key, float('nan'))
        print(f" {val:>8.4f}", end="")
    print()
    print(f"{'':=<70}")

    avg_rounds_print = np.mean([
        r.get("reasoning_retrieval", {}).get("avg_rounds_used", 1)
        for r in all_results
    ])
    print(f"\nAvg reasoning rounds: {avg_rounds_print:.2f}")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="musique.json")
    parser.add_argument("--sample_limit", type=int, default=200)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openie_cache", type=str, default=None,
                        help="Path to pre-computed OpenIE results JSON (e.g., gpt-4o-mini). "
                             "Skips LLM-based NER/triple extraction during indexing.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_musique_data(args.data_path)
    logger.info(f"Loaded {len(data)} samples")

    run_evaluation(data, args.sample_limit, args.max_rounds, openie_cache=args.openie_cache)


if __name__ == "__main__":
    main()
