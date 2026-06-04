"""
One-shot fix script.

Recovers NER+ITER (73034) and NER+IRCoT (73197) ner_em values that were
underestimated by the NEW regex extraction bug introduced in the 5/19 stashed
evaluate_musique_ner_pipeline.py.

Background:
  At 5/19 the llm_qa Answer-extraction was changed from
    response.split('Answer:')[1].strip()
  to
    re.search(r'Answer:?\\s*(.+?)(?:\\.\\s*$|\\n|$)', text, IGNORECASE | DOTALL)
  The new regex was case-insensitive and the colon was optional, so it
  matched "the answer" inside the Thought section before the real
  "Answer: X" label, causing wrong extractions.

  Current canonical evaluate_musique_ner_pipeline.py uses OLD split (no bug).

  73034/73197 result files only stored extracted answers; raw LLM responses
  were not saved EXCEPT in `reasoning_traces[last_round]` for ITER (which
  contains the final-round "Thought: ... Answer: X" string). That allows
  partial recovery.

Outputs:
  - Adds `ner_em_old_extract`, `ner_answer_old_extract`, and
    `ner_em_old_extract_recovery_method` per sample.
  - Adds `summary.ner_em_old_extract_avg` and `summary.extract_bug_note`.
  - Adds `config.extract_bug_5_19` describing the bug.
  Files are overwritten in place.
"""
import json
import re
import os
import sys
from typing import List, Optional, Tuple


def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s


def _em(pred: str, gold: str, aliases: Optional[List[str]]) -> int:
    p = _normalize(pred)
    for g in [gold] + (aliases or []):
        if p == _normalize(g):
            return 1
    return 0


def _extract_old_split(raw: str) -> Optional[str]:
    """4/21-style extraction: split on 'Answer:' (case-sensitive, colon required).

    Used by ITER + reasoning paths where llm_qa prompt formats responses as
    "Thought: ... Answer: X".
    """
    if not raw or 'Answer:' not in raw:
        return None
    try:
        ans = raw.split('Answer:')[1].strip()
        # Take up to first newline to avoid bleeding into next line of thought
        ans = ans.split('\n')[0].strip()
        ans = ans.rstrip('.').strip()
        return ans
    except Exception:
        return None


def _extract_ircot_so_the_answer(raw: str) -> Optional[str]:
    """IRCoT-style extraction: find 'so the answer is:' (case-insensitive),
    take content up to first newline. Mirrors the 4/21 split() spirit but
    targets the IRCoT prompt's expected phrasing
    (per IRCOT_SYSTEM in the 5/19 stashed pipeline).
    """
    if not raw:
        return None
    lower = raw.lower()
    idx = lower.find('so the answer is')
    if idx < 0:
        return None
    # Skip the phrase + optional colon + whitespace
    tail = raw[idx + len('so the answer is'):]
    if tail.startswith(':'):
        tail = tail[1:]
    tail = tail.lstrip()
    # Up to first newline
    ans = tail.split('\n')[0].strip()
    # Strip surrounding quotes and trailing period
    ans = ans.strip('"').strip("'").rstrip('.').strip()
    return ans or None


def _recover_one(result: dict) -> Tuple[Optional[str], str]:
    """Return (recovered_answer, recovery_method).

    Tries Answer: pattern first (ITER / reasoning format), then
    'so the answer is:' (IRCoT format). Walks traces in reverse so the
    final-round response is preferred.
    """
    traces = result.get('reasoning_traces') or []
    for i in range(len(traces) - 1, -1, -1):
        ans = _extract_old_split(traces[i])
        if ans:
            return ans, f"trace[{i}]_old_split_Answer"
    # Fall back: IRCoT 'so the answer is:' pattern
    for i in range(len(traces) - 1, -1, -1):
        ans = _extract_ircot_so_the_answer(traces[i])
        if ans:
            return ans, f"trace[{i}]_old_split_SoTheAnswerIs"
    return None, "no_recovery"


def process(path: str) -> None:
    print(f"\n=== {os.path.basename(path)} ===")
    with open(path) as f:
        data = json.load(f)

    results = data.get('results', [])
    n = len(results)
    if n == 0:
        print("  empty file, skipped")
        return

    n_stored_em = 0
    n_recovered_em = 0
    n_no_recovery = 0
    n_recovered_flip = 0  # NEW=0 → OLD=1

    for r in results:
        gold = r.get('gold_answer', '')
        aliases = r.get('gold_aliases', [])
        stored_ner_em = int(r.get('ner_em', 0))

        recovered_ans, method = _recover_one(r)
        if recovered_ans is None:
            # No raw response in traces → keep stored value as best guess
            r['ner_answer_old_extract'] = r.get('ner_answer', '')
            r['ner_em_old_extract'] = stored_ner_em
            r['ner_em_old_extract_recovery_method'] = 'no_raw_in_traces_kept_stored'
            n_no_recovery += 1
            n_recovered_em += stored_ner_em
        else:
            new_em = _em(recovered_ans, gold, aliases)
            r['ner_answer_old_extract'] = recovered_ans
            r['ner_em_old_extract'] = new_em
            r['ner_em_old_extract_recovery_method'] = method
            n_recovered_em += new_em
            if stored_ner_em == 0 and new_em == 1:
                n_recovered_flip += 1
        n_stored_em += stored_ner_em

    summary = data.setdefault('summary', {})
    summary['ner_em_old_extract_avg'] = round(n_recovered_em / n, 4)
    summary['ner_em_stored_avg'] = round(n_stored_em / n, 4)
    summary['ner_em_recovered_delta'] = round((n_recovered_em - n_stored_em) / n, 4)
    summary['ner_em_recovered_flips'] = n_recovered_flip
    summary['ner_em_no_recovery_count'] = n_no_recovery
    summary['extract_bug_note'] = (
        "ner_em stored value was extracted with a buggy regex introduced in "
        "5/19 NER pipeline (see config.extract_bug_5_19). ner_em_old_extract_avg "
        "is re-computed from reasoning_traces[last_round_with_'Answer:'] using "
        "the 4/21-style split('Answer:')[1].split('\\n')[0]. Samples without "
        "an Answer-bearing trace fall back to the stored value (lower bound)."
    )

    config = data.setdefault('config', {})
    config['extract_bug_5_19'] = {
        "introduced_in_commit": "4eebfc6 (stash@{0}, 5/19)",
        "fixed_in_commit": "8e98b23 (only-right-baseline, 5/19+)",
        "buggy_regex": r"re.search(r'Answer:?\s*(.+?)(?:\.\s*$|\n|$)', text, IGNORECASE|DOTALL)",
        "fixed_regex": "response.split('Answer:')[1].strip()",
        "symptom": "regex matched 'answer' inside Thought (e.g. 'the answer is X') "
                   "before the real 'Answer: Y' label, extracting wrong content.",
        "affected_field_in_this_file": "ner_em / ner_answer (and possibly baseline_em / baseline_answer; "
                                       "baseline raw responses were not saved → cannot be recovered)",
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  n_samples              : {n}")
    print(f"  ner_em (stored, NEW)   : {n_stored_em/n:.4f}")
    print(f"  ner_em (OLD re-extract): {n_recovered_em/n:.4f}   "
          f"(+{(n_recovered_em-n_stored_em)/n*100:.1f} pp)")
    print(f"  recovered flips (NEW=0 → OLD=1): {n_recovered_flip}")
    print(f"  no_recovery (no 'Answer:' in any trace): {n_no_recovery}")
    print(f"  saved fields:")
    print(f"    per-result: ner_answer_old_extract, ner_em_old_extract, ner_em_old_extract_recovery_method")
    print(f"    summary   : ner_em_old_extract_avg, ner_em_stored_avg, ner_em_recovered_delta,")
    print(f"                ner_em_recovered_flips, ner_em_no_recovery_count, extract_bug_note")
    print(f"    config    : extract_bug_5_19 (full bug description)")


def main():
    targets = [
        "outputs/musique_ner_pipeline_eval/comparison_results_iterretgen_rounds3_20260519_003951_73034.json",
        "outputs/musique_ner_pipeline_eval/comparison_results_ircot_rounds3_20260519_004222_73197.json",
    ]
    for p in targets:
        if not os.path.exists(p):
            print(f"MISSING: {p}")
            continue
        process(p)


if __name__ == "__main__":
    main()
