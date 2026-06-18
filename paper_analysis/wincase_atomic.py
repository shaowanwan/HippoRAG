"""Win-case sub-classification: of the MuSiQue questions where BRGD is correct and ITER-RETGEN
is wrong, classify HOW ITER-RETGEN failed (atomic reason). No graph step — purely ITER's failure.

  bridge_in_trace = a needed intermediate (hop) answer appears in ITER's reasoning trace
  iter_r5         = ITER dense retrieval R@5 (fraction of gold fetched)

  not bridge_in_trace          -> A (never produced the bridge) or D (drifted)  [LLM picks A/D]
  bridge_in_trace & R@5 == 0   -> B  : had the bridge entity but its DENSE retrieval fetched ZERO
                                       gold passages -> signal dilution (knows the entity, dense
                                       cannot route to it).
  bridge_in_trace & R@5 > 0    -> C  : had the bridge AND some gold, but answered wrong. LLM splits:
                                       REASONING (wrong but grounded in passages) / HALLUCINATION
                                       (ungrounded) / REFUSAL (abstains) / FORMAT (semantically same, EM=0)

Run from project root:  .venv/bin/python paper_analysis/wincase_atomic.py
"""
import json, re, string
from collections import Counter
from openai import OpenAI

BRGD = "outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260421_143341_41732.json"
ITER = "outputs/musique_ner_pipeline_eval/comparison_results_iterretgen_rounds3_20260606_155206_15547.json"
KEY = "sk-396199ed7af84eff8a0cf7a71b797601"
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OUT = "outputs/wincase_atomic.json"

C_SYS = """ITER-RETGEN retrieved gold passages but its final answer does not match the gold.
Using the gold passages, classify the ROOT cause. Reply ONLY one word:
REASONING     = the answer is a real entity/fact PRESENT in the gold passages, but the WRONG one
                (it had the evidence yet reasoned to the wrong hop / wrong conclusion).
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
    mq = {i: m for i, m in enumerate(json.load(open('musique.json')))}
    brgd = {r['idx']: r for r in json.load(open(BRGD))['results']}
    it = {r['idx']: r for r in json.load(open(ITER))['results']}
    wins = [i for i in brgd if i in it and brgd[i]['ner_em'] and not it[i]['ner_em']]
    print(f'{len(wins)} BRGD-win cases')

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
        m = mq[i]; itr = it[i]; qd = m.get('question_decomposition', []); final = sq(m.get('answer', ''))
        hops = [s['answer'] for s in qd if sq(s['answer']) != final]
        trace = ' '.join(str(x) for x in (itr.get('reasoning_traces') or []))
        in_trace = any(sq(h) and sq(h) in sq(trace) for h in hops)
        r5 = (itr.get('ner_recall') or {}).get('R@5', 0)
        gold_text = ' | '.join(p['paragraph_text'] for p in m['paragraphs'] if p.get('is_supporting'))

        if not in_trace:
            code = llm(AD_SYS, f"Question: {m['question']}\nGold: {m['answer']}\nNeeded bridge: {hops}\n"
                               f"ITER trace: {trace[:1200]}\nITER answer: {itr.get('ner_answer')}", r'(A|D)', 'A')
        elif r5 == 0:
            code = 'B'
        else:
            code = llm(C_SYS, f"Gold passages: {gold_text[:1500]}\nGold answer: {m['answer']}\n"
                              f"ITER answer: {itr.get('ner_answer')}", r'(REASONING|HALLUCINATION|REFUSAL|FORMAT)', 'REASONING')
            code = 'C-' + code.lower()
        cnt[code] += 1
        recs.append({'idx': i, 'code': code, 'r5': r5, 'bridge_in_trace': in_trace,
                     'iter_answer': itr.get('ner_answer'), 'gold': m.get('answer')})

    n = len(wins)
    order = ['A', 'D', 'B', 'C-reasoning', 'C-hallucination', 'C-refusal', 'C-format']
    LBL = {'A': 'A  never produced bridge (refused)', 'D': 'D  drifted',
           'B': 'B  had bridge, dense retrieved 0 gold (signal dilution)',
           'C-reasoning': 'C  reasoning failure (grounded but wrong)',
           'C-hallucination': 'C  hallucination (ungrounded)',
           'C-refusal': 'C  refusal', 'C-format': 'C  format-only (EM=0, semantically ok)'}
    print(f'\nWin-case atomic classification (n={n}):')
    for c in order:
        if cnt.get(c): print(f'  {LBL[c]:55} {cnt[c]:3} ({cnt[c]/n*100:.1f}%)')
    cqa = sum(cnt[c] for c in order if c.startswith('C-'))
    print(f'  --> A+D (no bridge)={cnt["A"]+cnt["D"]} | B (retrieval/dilution)={cnt["B"]} | C (QA)={cqa}')
    json.dump({'n': n, 'counts': dict(cnt), 'records': recs}, open(OUT, 'w'), indent=1)
    print('saved', OUT)


if __name__ == "__main__":
    main()
