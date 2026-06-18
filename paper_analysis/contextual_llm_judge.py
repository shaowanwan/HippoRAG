"""LLM-judge validation: are contextual bridges relevant to the answer reasoning chain?
Semantic cross-check of the graph-connectivity evidence/distractor split.
Sample of N queries (qwen-plus). Run from project root.
"""
import json, re, string, sys
from openai import OpenAI

RUN = "outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260421_143341_41732.json"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
KEY = "sk-396199ed7af84eff8a0cf7a71b797601"
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
SYS = ('You judge whether candidate entities are relevant to a multi-hop question. An entity is '
       'RELEVANT if it is part of the topic, evidence, or intermediate reasoning that connects the '
       'question to its answer; UNRELATED if it is a distractor not needed for this question.')


def sq(a):
    a = a or ''; ex = set(string.punctuation)
    a = ''.join(c for c in a.lower() if c not in ex); a = re.sub(r'\b(a|an|the)\b', ' ', a)
    return ' '.join(a.split())


def main():
    cli = OpenAI(api_key=KEY, base_url=BASE, timeout=60)
    mq = {i: m for i, m in enumerate(json.load(open('musique.json')))}
    d = json.load(open(RUN))['results']
    rel = unrel = err = nq = 0
    for r in d[:N]:
        m = mq[r['idx']]; qd = m.get('question_decomposition', [])
        final = sq(m.get('answer', '')); inter = [sq(s['answer']) for s in qd if sq(s['answer']) != final]; q = sq(m.get('question', ''))
        raw = set()
        for rd in (r.get('round_diagnostics') or []): raw |= set(rd.get('new_discovered_entities') or [])
        ctx = [n for n in {x for x in raw if x}
               if not (sq(n) == final or sq(n) in final or final in sq(n)
                       or any(sq(n) == h or sq(n) in h or h in sq(n) for h in inter if h) or sq(n) in q)]
        if not ctx: continue
        nq += 1
        lst = '\n'.join(f'{i+1}. {e}' for i, e in enumerate(ctx))
        usr = (f"Question: {m['question']}\nFinal answer: {m['answer']}\n\nCandidate entities discovered "
               f"while answering:\n{lst}\n\nFor each, reply 'relevant' or 'unrelated'. Output ONLY JSON "
               f'like {{"1":"relevant","2":"unrelated"}}.')
        try:
            resp = cli.chat.completions.create(model='qwen-plus', temperature=0, max_tokens=400,
                                               messages=[{'role': 'system', 'content': SYS}, {'role': 'user', 'content': usr}])
            j = json.loads(re.search(r'\{.*\}', resp.choices[0].message.content, re.DOTALL).group())
            for i in range(len(ctx)):
                v = str(j.get(str(i + 1), '')).lower()
                if 'rel' in v and 'unrel' not in v: rel += 1
                elif 'unrel' in v: unrel += 1
                else: err += 1
        except Exception:
            err += len(ctx)
    tot = rel + unrel
    print(f'LLM judge on {nq} queries, {tot} contextual bridges:')
    print(f'  RELEVANT (evidence-chain): {rel} ({rel/tot*100:.1f}%)')
    print(f'  UNRELATED (distractor):    {unrel} ({unrel/tot*100:.1f}%)')
    print(f'  parse errors: {err}')


if __name__ == "__main__":
    main()
