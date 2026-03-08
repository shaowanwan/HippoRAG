"""
Demo: Reasoning-guided iterative retrieval with HippoRAG.

This demonstrates the multi-round retrieval where an LLM reasons about
intermediate results to rewrite queries, adjust entity weights, and
add temporary graph edges between rounds.

Usage:
    python demo_reasoning.py
"""
import json
import logging
import os

logging.basicConfig(level=logging.INFO)


def main():
    from src.hipporag import HippoRAG
    # --- Corpus ---
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is born in Minsk.",
        "Montebello is a part of Rockland County.",
    ]

    save_dir = "outputs/reasoning_demo"
    llm_model_name = "qwen-plus"
    embedding_model_name = "Transformers/sentence-transformers/all-MiniLM-L6-v2"
    aliyun_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', "sk-396199ed7af84eff8a0cf7a71b797601")

    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url=aliyun_base_url,
    )

    # Index
    hipporag.index(docs=docs)

    # --- Queries ---
    queries = [
        "What county is Erik Hort's birthplace a part of?",  # Multi-hop
    ]
    gold_docs = [
        [
            "Erik Hort's birthplace is Montebello.",
            "Montebello is a part of Rockland County.",
        ],
    ]
    gold_answers = [
        ["Rockland County"],
    ]

    # --- Standard single-pass retrieval (baseline) ---
    print("\n" + "=" * 60)
    print("BASELINE: Single-pass retrieval")
    print("=" * 60)
    baseline_results = hipporag.rag_qa(
        queries=queries, gold_docs=gold_docs, gold_answers=gold_answers
    )
    print(f"Baseline QA: {baseline_results[-1]}")

    # --- Reasoning-guided iterative retrieval ---
    print("\n" + "=" * 60)
    print("REASONING: Multi-round iterative retrieval (max 3 rounds)")
    print("=" * 60)
    reasoning_results = hipporag.reasoning_rag_qa(
        queries=queries,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
        max_rounds=3,
    )
    solutions, responses, metadata, retrieval_eval, qa_eval = reasoning_results

    print(f"\nRetrieval eval (per-round + final):")
    print(json.dumps(retrieval_eval, indent=2))

    print(f"\nQA eval:")
    print(json.dumps(qa_eval, indent=2))

    print(f"\nFinal answer: {solutions[0].answer}")
    print(f"Gold answer: {gold_answers[0]}")


if __name__ == "__main__":
    main()
