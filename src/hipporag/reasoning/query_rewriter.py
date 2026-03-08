import json
import logging
import signal
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Timeout for reasoning LLM calls (seconds)
REASONING_TIMEOUT = 60

REWRITE_SYSTEM_PROMPT = """You are a retrieval reasoning assistant. Given an original query, the documents retrieved so far, and optionally a reasoning trace, your job is to:

1. Analyze what information has been found and what is still missing.
2. Decide whether the retrieved documents already contain sufficient information to answer the query.
3. If not, rewrite the query to better target the missing information.

Respond in JSON format:
{
    "analysis": "Brief analysis of what's found vs missing",
    "rewritten_query": "The rewritten query targeting missing info",
    "should_stop": false
}

Rules:
- If the retrieved documents already contain sufficient information to answer the query, set should_stop=true and leave rewritten_query empty.
- Rewritten query should be a natural language question, not keywords.
- Focus on what information is missing and craft the query to find it.
"""


def build_rewrite_prompt(
    original_query: str,
    current_query: str,
    retrieved_docs: List[str],
    round_idx: int,
    previous_traces: List[str] = None,
) -> List[dict]:
    """Build the LLM prompt for query rewriting."""
    docs_text = ""
    for i, doc in enumerate(retrieved_docs[:10]):
        docs_text += f"[Doc {i+1}] {doc[:500]}\n\n"

    user_content = f"""Original query: {original_query}
Current query (round {round_idx}): {current_query}

Retrieved documents so far:
{docs_text}"""

    if previous_traces:
        user_content += "\nPrevious reasoning traces:\n"
        for t in previous_traces[-3:]:
            user_content += f"- {t}\n"

    user_content += "\nAnalyze and provide your reasoning output in JSON."

    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class QueryRewriter:
    """Uses LLM reasoning to rewrite queries for the next retrieval round."""

    def __init__(self, llm_model):
        self.llm_model = llm_model

    def reason_and_rewrite(
        self,
        original_query: str,
        current_query: str,
        retrieved_docs: List[str],
        round_idx: int,
        previous_traces: List[str] = None,
    ) -> dict:
        """Call LLM to reason about retrieval results and rewrite query.

        Returns:
            dict with keys: analysis, rewritten_query, should_stop
        """
        messages = build_rewrite_prompt(
            original_query=original_query,
            current_query=current_query,
            retrieved_docs=retrieved_docs,
            round_idx=round_idx,
            previous_traces=previous_traces,
        )

        try:
            def _timeout_handler(signum, frame):
                raise TimeoutError("Reasoning LLM call timed out")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(REASONING_TIMEOUT)
            try:
                response, metadata, cache_hit = self.llm_model.infer(messages)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            result = self._parse_response(response)
            return result
        except TimeoutError:
            logger.warning(f"QueryRewriter timed out after {REASONING_TIMEOUT}s")
            return {
                "analysis": "Timed out",
                "rewritten_query": current_query,
                "should_stop": True,
            }
        except Exception as e:
            logger.error(f"QueryRewriter failed: {e}")
            return {
                "analysis": f"Error: {e}",
                "rewritten_query": current_query,
                "should_stop": True,
            }

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response JSON, with fallbacks for malformed output."""
        defaults = {
            "analysis": "",
            "rewritten_query": "",
            "should_stop": False,
        }

        try:
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(text)
            for key in defaults:
                if key not in parsed:
                    parsed[key] = defaults[key]
            return parsed
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to parse rewriter response: {e}")
            defaults["analysis"] = response[:200] if response else "Parse failed"
            defaults["should_stop"] = True
            return defaults
