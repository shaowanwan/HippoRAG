"""
NER pipeline + oracle bridge injection (NO LLM reasoning loop).

Tests whether bridge entities ALONE — without multi-round LLM reasoning — provide
the EM gain seen in PID 41732 (canonical NER+reasoning, EM=0.396).

Design (Decision c):
  - Source of bridges: PID 41732 output, per-query final union (round_diagnostics[-1].discovered_entities_total).
  - Pipeline: canonical NERIndex from evaluate_musique_ner_pipeline.py (imported, NOT modified).
  - Flow per query:
      1. Build NERIndex (same code path as canonical 41732).
      2. baseline = index.retrieve(question) → llm_qa.
      3. Look up bridges for this query from 41732 cache.
      4. Resolve bridges → vertex IDs (canonical _resolve_entities_in_graph).
      5. extra_node_weights[vid] += _degree_adaptive_weight(vid, 0.5) for each resolved bridge.
      6. oracle = index.retrieve(question, extra_node_weights=ews)  ← single PPR, no overlay edges, no query rewrite.
      7. llm_qa(question, oracle_docs[:5]).

Comparison:
  - NER baseline (single retrieve, no bridges)  ← from 41732 baseline_em ≈ 0.260
  - NER + oracle bridges (this script)          ← NEW
  - NER + reasoning full (multi-round LLM)      ← from 41732 ner_em ≈ 0.396

Usage:
    .venv/bin/python evaluate_musique_ner_oracle_bridges.py \
        --data_path musique.json --sample_limit 30 \
        --bridge_source outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260421_143341_41732.json
"""
import json
import os
import sys
import argparse
import logging
import time
import gc
import traceback
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# NO_SIM_FACTOR=1 matches "decision c" (no query-pair sim available without LLM round).
os.environ.setdefault("NO_SIM_FACTOR", "1")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Import canonical NER pipeline as-is. Do not modify it.
import evaluate_musique_ner_pipeline as ner

# Pickle compatibility: the global_ner_index.pkl was saved while NERIndex was in
# the __main__ module (evaluate_musique_ner_pipeline.py run directly). When we
# unpickle from this script, pickle looks for `__main__.NERIndex` and fails.
# Expose NERIndex (and a few related classes) in this __main__ namespace so the
# pickle resolves. No code is modified — only an alias.
import sys as _sys
_main_module = _sys.modules['__main__']
for _cls_name in dir(ner):
    if not _cls_name.startswith('_'):
        _obj = getattr(ner, _cls_name)
        if isinstance(_obj, type) and not hasattr(_main_module, _cls_name):
            setattr(_main_module, _cls_name, _obj)


def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def load_bridge_source(path):
    """Load PID 41732 (or equivalent) result file, return {idx: bridges_list}.

    Uses round_diagnostics[-1].discovered_entities_total when non-empty,
    otherwise falls back to union of new_discovered_entities across rounds.
    """
    data = json.load(open(path))
    out = {}
    for r in data["results"]:
        idx = r["idx"]
        rds = r.get("round_diagnostics", [])
        if not rds:
            out[idx] = []
            continue
        # Last round's total is the union of all rounds at end.
        last_total = rds[-1].get("discovered_entities_total", []) or []
        if last_total:
            out[idx] = list(last_total)
            continue
        # Fallback: union of new_discovered_entities from all rounds.
        union = set()
        for rd in rds:
            for e in rd.get("new_discovered_entities", []) or []:
                union.add(e)
        out[idx] = sorted(union)
    return out


def save_results(output_path, config, all_results):
    n = len(all_results)
    if n == 0:
        return

    def _em(field):
        return sum(
            1 for r in all_results
            if ner.check_em(r[field], r["gold_answer"], r.get("gold_aliases", []))
        ) / n

    def _avg(key):
        vals = [r.get(key) for r in all_results]
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "n_completed": n,
        "baseline_em": round(_em("baseline_answer"), 4),
        "oracle_em": round(_em("oracle_answer"), 4),
        "improvement_em_oracle_vs_baseline": round(_em("oracle_answer") - _em("baseline_answer"), 4),
        "baseline_f1": _avg("baseline_f1"),
        "oracle_f1": _avg("oracle_f1"),
        "baseline_recall_at_1": _avg("baseline_r1"),
        "baseline_recall_at_2": _avg("baseline_r2"),
        "baseline_recall_at_5": _avg("baseline_r5"),
        "oracle_recall_at_1": _avg("oracle_r1"),
        "oracle_recall_at_2": _avg("oracle_r2"),
        "oracle_recall_at_5": _avg("oracle_r5"),
        "bridges_resolved_mean": _avg("n_bridges_resolved"),
        "bridges_provided_mean": _avg("n_bridges_provided"),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"config": config, "summary": summary, "results": all_results},
            f, indent=2, default=make_serializable,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="musique.json")
    parser.add_argument("--sample_limit", type=int, default=30)
    parser.add_argument(
        "--bridge_source",
        type=str,
        default="outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260421_143341_41732.json",
    )
    parser.add_argument(
        "--ner_cache",
        type=str,
        default="outputs/musique_ner_pipeline_eval/ner_cache.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output dir. Default: per-sample → musique_ner_oracle_bridges_eval, "
             "global → musique_ner_oracle_bridges_global_eval",
    )
    parser.add_argument("--global_index", action="store_true",
                        help="Use a single global NER index across ALL samples "
                             "(matches PID 41732 setup). Default: per-sample index.")
    parser.add_argument(
        "--global_index_pickle",
        type=str,
        default="outputs/musique_ner_pipeline_eval/global_ner_index.pkl",
        help="Path to cached global NER index pickle. Loaded directly if exists.",
    )
    parser.add_argument(
        "--openie_cache",
        type=str,
        default="outputs/musique/openie_results_ner_qwen-plus.json",
        help="OpenIE cache (fallback for entity extraction when ner_cache misses).",
    )
    parser.add_argument("--seed_weight", type=float, default=0.5,
                        help="DEFAULT_ENTITY_SEED_WEIGHT for bridge injection (canonical default 0.5).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    # Imports (lazy, mirror canonical script)
    from src.hipporag.embedding_model import _get_embedding_model_class

    # Data
    all_data = json.load(open(args.data_path))
    if args.sample_limit and args.sample_limit < len(all_data):
        data = all_data[:args.sample_limit]
    else:
        data = all_data
    logger.info(f"Loaded {len(data)} samples")

    # Bridge cache
    if not os.path.exists(args.bridge_source):
        logger.error(f"Bridge source not found: {args.bridge_source}")
        sys.exit(1)
    bridges_by_idx = load_bridge_source(args.bridge_source)
    logger.info(f"Loaded bridges for {len(bridges_by_idx)} samples from {args.bridge_source}")

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

    logger.info(f"LLM: {llm_model_name}")
    llm_client = ner.SimpleLLM(model_name=llm_model_name, base_url=aliyun_base_url)

    # NER cache (per-doc entity extraction reuse, same as canonical 41732)
    ner_cache = {}
    if args.ner_cache and os.path.exists(args.ner_cache):
        logger.info(f"Loading NER cache: {args.ner_cache}")
        ner_cache = json.load(open(args.ner_cache))
        logger.info(f"  {len(ner_cache)} cached entries")

    # OpenIE cache (fallback for entity extraction; mirrors canonical line 1644-1666)
    openie_cache = {}
    if args.openie_cache and os.path.exists(args.openie_cache):
        logger.info(f"Loading OpenIE cache: {args.openie_cache}")
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

    # Global index (mirrors canonical line 1688-1721, matches PID 41732 setup)
    global_index = None
    if args.global_index:
        if os.path.exists(args.global_index_pickle):
            logger.info(f"Loading cached global NER index: {args.global_index_pickle}")
            global_index = ner.NERIndex.load(args.global_index_pickle, embedding_model=emb_model)
            logger.info(
                f"  Loaded: {global_index.graph.vcount()} nodes, "
                f"{global_index.graph.ecount()} edges"
            )
        else:
            logger.info("Building global NER index from ALL samples (not just sample_limit)...")
            full_data = json.load(open(args.data_path))
            all_docs = []
            seen = set()
            for sample in full_data:
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
                    entities = ner.llm_ner(doc_text, llm_client)
                    global_ner_results[doc_text] = entities
                    ner_cache[doc_text] = entities
            global_index = ner.NERIndex(embedding_model=emb_model)
            global_index.build(all_docs, global_ner_results)
            logger.info(
                f"Built global index: {global_index.graph.vcount()} nodes, "
                f"{global_index.graph.ecount()} edges"
            )

    # Output dir: auto-pick based on global flag if not specified
    if args.output_dir:
        save_dir = args.output_dir
    elif args.global_index:
        save_dir = "outputs/musique_ner_oracle_bridges_global_eval"
    else:
        save_dir = "outputs/musique_ner_oracle_bridges_eval"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "comparison_results.json")

    config = {
        "method": "ner_oracle_bridges_single_round",
        "decision": "c (single PPR, no overlay edges, no query rewrite)",
        "global_index": bool(args.global_index),
        "bridge_source": args.bridge_source,
        "sample_limit": args.sample_limit,
        "llm": llm_model_name,
        "embedding": embedding_model_name,
        "seed_weight": args.seed_weight,
        "NO_SIM_FACTOR": os.environ.get("NO_SIM_FACTOR", "0"),
    }

    # Resume
    all_results = []
    completed_idxs = set()
    if os.path.exists(output_path):
        try:
            prev = json.load(open(output_path))
            all_results = prev.get("results", [])
            completed_idxs = {r["idx"] for r in all_results}
            logger.info(f"Resuming from {len(completed_idxs)} samples")
        except Exception:
            all_results = []
            completed_idxs = set()

    total_start = time.time()

    for idx, sample in enumerate(data):
        if idx in completed_idxs:
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

        bridges = bridges_by_idx.get(idx, [])
        logger.info(f"[{idx+1}/{len(data)}] bridges={len(bridges)} : {question[:70]}...")

        try:
            # 1. Build/select NERIndex (matches canonical line 1761-1776)
            if global_index is not None:
                # Use shared global index for all samples (matches PID 41732 setup)
                index = global_index
            else:
                # Per-sample index: build from this sample's docs.
                # Mirrors canonical fallback order: ner_cache → openie_cache → llm_ner.
                ner_results = {}
                for doc_text in docs:
                    if doc_text in ner_cache:
                        ner_results[doc_text] = ner_cache[doc_text]
                    elif doc_text in openie_cache:
                        ner_results[doc_text] = openie_cache[doc_text]
                    else:
                        entities = ner.llm_ner(doc_text, llm_client)
                        ner_results[doc_text] = entities
                        ner_cache[doc_text] = entities

                index = ner.NERIndex(embedding_model=emb_model)
                index.build(docs, ner_results)

            # 2. Baseline: single retrieve, no bridges
            base_sorted_ids, _ = index.retrieve(question)
            base_docs = [index.passages[index.passage_keys[did]] for did in base_sorted_ids]
            baseline_answer = ner.llm_qa(question, base_docs[:5], llm_client)
            base_em = int(ner.check_em(baseline_answer, answer, answer_aliases))
            base_f1 = round(ner.compute_f1(baseline_answer, answer), 4)
            base_r1 = ner.recall_at_k(base_docs, gold_docs, 1)
            base_r2 = ner.recall_at_k(base_docs, gold_docs, 2)
            base_r5 = ner.recall_at_k(base_docs, gold_docs, 5)

            # 3-5. Oracle: resolve bridges, build extra_node_weights, single retrieve
            if bridges:
                resolved = ner._resolve_entities_in_graph(index, bridges)
                extra_node_weights = np.zeros(index.graph.vcount())
                for name, (vid, sim) in resolved.items():
                    # decision c: single round, no decay, sim forced to 1.0 via NO_SIM_FACTOR
                    extra_node_weights[vid] += ner._degree_adaptive_weight(
                        index, vid, args.seed_weight, sim=1.0
                    )
                n_resolved = len(resolved)
            else:
                resolved = {}
                extra_node_weights = None
                n_resolved = 0

            # 6. Oracle retrieve — single PPR, no overlay graph, no query rewrite
            oracle_sorted_ids, _ = index.retrieve(
                question,
                extra_node_weights=extra_node_weights,
                working_graph=None,
            )
            oracle_docs = [index.passages[index.passage_keys[did]] for did in oracle_sorted_ids]
            oracle_answer = ner.llm_qa(question, oracle_docs[:5], llm_client)
            oracle_em = int(ner.check_em(oracle_answer, answer, answer_aliases))
            oracle_f1 = round(ner.compute_f1(oracle_answer, answer), 4)
            oracle_r1 = ner.recall_at_k(oracle_docs, gold_docs, 1)
            oracle_r2 = ner.recall_at_k(oracle_docs, gold_docs, 2)
            oracle_r5 = ner.recall_at_k(oracle_docs, gold_docs, 5)

        except Exception as e:
            logger.error(f"  Failed sample {idx}: {e}")
            traceback.print_exc()
            baseline_answer = "Error"
            oracle_answer = "Error"
            base_em = oracle_em = 0
            base_f1 = oracle_f1 = 0.0
            base_r1 = base_r2 = base_r5 = 0.0
            oracle_r1 = oracle_r2 = oracle_r5 = 0.0
            resolved = {}
            n_resolved = 0

        result = {
            "idx": idx,
            "question": question,
            "gold_answer": answer,
            "gold_aliases": answer_aliases,
            "baseline_answer": baseline_answer,
            "oracle_answer": oracle_answer,
            "baseline_em": base_em,
            "oracle_em": oracle_em,
            "baseline_f1": base_f1,
            "oracle_f1": oracle_f1,
            "baseline_r1": base_r1,
            "baseline_r2": base_r2,
            "baseline_r5": base_r5,
            "oracle_r1": oracle_r1,
            "oracle_r2": oracle_r2,
            "oracle_r5": oracle_r5,
            "n_bridges_provided": len(bridges),
            "n_bridges_resolved": n_resolved,
            "bridges_provided": bridges,
            "bridges_resolved": {name: int(vid) for name, (vid, _) in resolved.items()},
        }
        all_results.append(result)

        b_mark = "Y" if base_em else "N"
        o_mark = "Y" if oracle_em else "N"
        logger.info(
            f"  B={b_mark} '{baseline_answer[:35]}' | O={o_mark} '{oracle_answer[:35]}' | "
            f"Gold='{answer}' | bridges {n_resolved}/{len(bridges)}"
        )

        if (idx + 1) % 5 == 0:
            save_results(output_path, config, all_results)
            n = len(all_results)
            b_em = sum(r["baseline_em"] for r in all_results) / n
            o_em = sum(r["oracle_em"] for r in all_results) / n
            elapsed = time.time() - total_start
            eta = elapsed / max(n - len(completed_idxs), 1) * (len(data) - n)
            logger.info(
                f"  >>> {n}/{len(data)} | Baseline EM={b_em:.3f} | Oracle EM={o_em:.3f} | "
                f"ETA={eta/60:.0f}min"
            )

        gc.collect()

    # Persist NER cache if we added new entries
    if args.ner_cache:
        try:
            with open(args.ner_cache, "w") as f:
                json.dump(ner_cache, f)
            logger.info(f"NER cache saved: {len(ner_cache)} entries")
        except Exception as e:
            logger.warning(f"Could not save NER cache: {e}")

    save_results(output_path, config, all_results)

    n = len(all_results)
    if n > 0:
        b_em = sum(r["baseline_em"] for r in all_results) / n
        o_em = sum(r["oracle_em"] for r in all_results) / n
        b_r5 = float(np.mean([r["baseline_r5"] for r in all_results]))
        o_r5 = float(np.mean([r["oracle_r5"] for r in all_results]))
        bridges_avg = float(np.mean([r["n_bridges_resolved"] for r in all_results]))
        print(f"\n{'='*60}")
        print(f"NER + Oracle Bridges (single round): {n} samples")
        print(f"  Baseline EM:  {b_em:.4f}    R@5: {b_r5:.4f}")
        print(f"  Oracle  EM:   {o_em:.4f}    R@5: {o_r5:.4f}")
        print(f"  Delta EM:     {o_em - b_em:+.4f}")
        print(f"  Delta R@5:    {o_r5 - b_r5:+.4f}")
        print(f"  Avg bridges resolved: {bridges_avg:.1f}")
        print(f"  Results: {output_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
