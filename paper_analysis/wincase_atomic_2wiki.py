"""Win-case sub-classification on 2WikiMultiHopQA: of the questions where BRGD is correct and
ITER-RETGEN is wrong, classify HOW ITER failed. Same taxonomy as MuSiQue, but:
  - bridge/hop entities come from the 2Wiki `evidences` reasoning chain (joined by id);
  - B (retrieval failure) = bridge_in_trace AND R@5 < 1.0  (2Wiki is retrieval-bound).
Run from project root: .venv/bin/python paper_analysis/wincase_atomic_2wiki.py
"""
import json, re, string
from collections import Counter
from openai import OpenAI

BRGD = "outputs/2wikimultihopqa_ner_pipeline_eval/comparison_results_rounds3_20260519_225606_28808.json"
ITER = "outputs/2wikimultihopqa_ner_pipeline_eval/comparison_results_iterretgen_rounds3_20260606_155204_15542.json"
ROOT = "2wikimultihopqa.json"
ORIG = "reproduce/dataset/2wikimultihopqa.json"
KEY = "sk-396199ed7af84eff8a0cf7a71b797601"
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OUT = "outputs/wincase_atomic_2wiki.json"

C_SYS = """ITER-RETGEN retrieved gold passages but its final answer does not match the gold.
Using the gold passages, classify the ROOT cause. Reply ONLY one word:
REASONING     = the answer is a real entity/fact PRESENT in the gold passages, but the WRONG one
                (it had the evidence yet reasoned to the wrong conclusion).
HALLUCINATION = the answer is NOT supported by the gold passages (fabricated / ungrounded).
REFUSAL       = it abstains ("not specified", "unknown", "none", or empty).
FORMAT        = the answer is semantically the SAME as the gold (paraphrase / verbose), just EM=0."""
AD_SYS = """ITER-RETGEN never produced the needed bridge/intermediate entity. Reply ONLY one letter:
A = it refused or produced no useful reasoning toward the bridge.
D = it reasoned but drifted off-topic in the first round and never recovered."""


def sq(a):
    a = a or ''; ex = set(string.punctuation)
    a = ''.join(c for c in a.lower() if c not in ex); a = re.sub(r'\b(a|an|the)\b', ' ', a)
    return ' '.join(a.split())


def main():
    cli = OpenAI(api_key=KEY, base_url=BASE, timeout=60)
    root = json.load(open(ROOT)); oid = {o['_id']: o for o in json.load(open(ORIG))}
    brgd = {r['idx']: r for r in json.load(open(BRGD))['results']}
    it = {r['idx']: r for r in json.load(open(ITER))['results']}
    wins = [i for i in brgd if i in it and brgd[i]['ner_em'] and not it[i]['ner_em']]
    print(f'{len(wins)} BRGD-win cases (2Wiki)')

    def llm(sys_p, usr, pat, dft):
        try:
            r = cli.chat.completions.create(model='qwen-plus', temperature=0, max_tokens=8,
                                            messages=[{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': usr}])
            m = re.search(pat, r.choices[0].message.content.strip().upper())
            return m.group(1) if m else dft
        except Exception:
            return dft

    cnt = Counter(); recs = []
    for i in wins:
        m = root[i]; o = oid.get(m['id'], {}); itr = it[i]
        final = sq(m.get('answer', ''))
        hop = set()
        for ev in (o.get('evidences') or []):
            if isinstance(ev, (list, tuple)) and len(ev) >= 3:
                for e in (ev[0], ev[2]):
                    if sq(e) and sq(e) != final: hop.add(sq(e))
        trace = ' '.join(str(x) for x in (itr.get('reasoning_traces') or []))
        in_trace = any(h and h in sq(trace) for h in hop)
        r5 = (itr.get('ner_recall') or {}).get('R@5', 0)
        gold_text = ' | '.join(p['paragraph_text'] for p in m['paragraphs'] if p.get('is_supporting'))

        if not in_trace:
            code = llm(AD_SYS, f"Question: {m['question']}\nGold: {m['answer']}\nNeeded bridge: {list(hop)[:8]}\n"
                               f"ITER trace: {trace[:1200]}\nITER answer: {itr.get('ner_answer')}", r'(A|D)', 'A')
        elif r5 < 1.0:
            code = 'B'
        else:
            code = 'C-' + llm(C_SYS, f"Gold passages: {gold_text[:1500]}\nGold answer: {m['answer']}\n"
                              f"ITER answer: {itr.get('ner_answer')}", r'(REASONING|HALLUCINATION|REFUSAL|FORMAT)', 'REASONING').lower()
        cnt[code] += 1
        recs.append({'idx': i, 'code': code, 'r5': r5, 'bridge_in_trace': in_trace,
                     'iter_answer': itr.get('ner_answer'), 'gold': m.get('answer')})

    n = len(wins)
    order = ['A', 'D', 'B', 'C-reasoning', 'C-hallucination', 'C-refusal', 'C-format']
    LBL = {'A': 'A  never produced bridge (refused)', 'D': 'D  drifted',
           'B': 'B  had bridge, retrieval R@5<1 (retrieval failure)',
           'C-reasoning': 'C  reasoning failure (grounded but wrong)', 'C-hallucination': 'C  hallucination (ungrounded)',
           'C-refusal': 'C  refusal', 'C-format': 'C  format-only (EM=0, semantically ok)'}
    print(f'\nWin-case atomic classification (2Wiki, n={n}):')
    for c in order:
        if cnt.get(c): print(f'  {LBL[c]:55} {cnt[c]:3} ({cnt[c]/n*100:.1f}%)')
    cqa = sum(cnt[c] for c in order if c.startswith('C-'))
    print(f'  --> A+D (no bridge)={cnt["A"]+cnt["D"]} | B (retrieval)={cnt["B"]} | C (QA)={cqa}')
    json.dump({'n': n, 'counts': dict(cnt), 'records': recs}, open(OUT, 'w'), indent=1)
    print('saved', OUT)


if __name__ == "__main__":
    main()
