"""2Wiki failure breakdown: classify BRGD's failures (ITER-win cases) with the same taxonomy as
ITER's failures (paper_analysis/wincase_atomic_2wiki.py), and plot the two side by side.
Reads ITER-failure counts from outputs/wincase_atomic_2wiki.json (run that first).
Run from project root: .venv/bin/python paper_analysis/failure_breakdown_2wiki.py
"""
import json, re, string
from collections import Counter
from openai import OpenAI

BRGD = "outputs/2wikimultihopqa_ner_pipeline_eval/comparison_results_rounds3_20260519_225606_28808.json"
ITER = "outputs/2wikimultihopqa_ner_pipeline_eval/comparison_results_iterretgen_rounds3_20260606_155204_15542.json"
ROOT = "2wikimultihopqa.json"
ORIG = "reproduce/dataset/2wikimultihopqa.json"
ITER_FAIL_JSON = "outputs/wincase_atomic_2wiki.json"   # produced by wincase_atomic_2wiki.py
KEY = "sk-396199ed7af84eff8a0cf7a71b797601"
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OUT = "outputs/failure_breakdown_2wiki"

C_SYS = ("A retrieved-gold answer is wrong. Reply ONE word: REASONING (grounded wrong) / "
         "HALLUCINATION (ungrounded) / REFUSAL (abstain) / FORMAT (semantically same, EM=0).")


def sq(a):
    a = a or ''; ex = set(string.punctuation)
    a = ''.join(c for c in a.lower() if c not in ex); a = re.sub(r'\b(a|an|the)\b', ' ', a)
    return ' '.join(a.split())


def classify_brgd_failures():
    cli = OpenAI(api_key=KEY, base_url=BASE, timeout=60)
    root = json.load(open(ROOT)); oid = {o['_id']: o for o in json.load(open(ORIG))}
    brgd = {r['idx']: r for r in json.load(open(BRGD))['results']}
    it = {r['idx']: r for r in json.load(open(ITER))['results']}
    fails = [i for i in brgd if i in it and not brgd[i]['ner_em'] and it[i]['ner_em']]
    cnt = Counter()
    for i in fails:
        m = root[i]; o = oid.get(m['id'], {}); b = brgd[i]; final = sq(m.get('answer', ''))
        hop = set()
        for ev in (o.get('evidences') or []):
            if isinstance(ev, (list, tuple)) and len(ev) >= 3:
                for e in (ev[0], ev[2]):
                    if sq(e) and sq(e) != final: hop.add(sq(e))
        disc = set()
        for rd in (b.get('round_diagnostics') or []): disc |= set(sq(x) for x in (rd.get('new_discovered_entities') or []))
        trace = ' '.join(str(x) for x in (b.get('reasoning_traces') or []))
        in_b = any(h and (h in sq(trace) or any(h in dd or dd in h for dd in disc)) for h in hop)
        r5 = (b.get('ner_recall') or {}).get('R@5', 0)
        gold = ' | '.join(p['paragraph_text'] for p in m['paragraphs'] if p.get('is_supporting'))
        if not in_b:
            code = 'A'
        elif r5 < 1.0:
            code = 'B'
        else:
            try:
                r = cli.chat.completions.create(model='qwen-plus', temperature=0, max_tokens=8,
                    messages=[{'role': 'system', 'content': C_SYS},
                              {'role': 'user', 'content': f"Gold: {gold[:1400]}\nGold answer: {m['answer']}\nBRGD answer: {b.get('ner_answer')}"}])
                mt = re.search(r'(REASONING|HALLUCINATION|REFUSAL|FORMAT)', r.choices[0].message.content.upper())
                code = 'C-' + (mt.group(1).lower() if mt else 'reasoning')
            except Exception:
                code = 'C-reasoning'
        cnt[code] += 1
    return dict(cnt), len(fails)


def main():
    iter_counts = json.load(open(ITER_FAIL_JSON))['counts']
    brgd_counts, n_brgd = classify_brgd_failures()
    n_iter = sum(iter_counts.values())
    print('ITER failures:', iter_counts)
    print('BRGD failures:', brgd_counts)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import numpy as np
    cols = [('B', 'Retrieval (R@5<1)', '#4C72B0'), ('C-reasoning', 'Reasoning error', '#DD8452'),
            ('C-hallucination', 'Hallucination', '#C44E52'), ('C-refusal', 'Refusal', '#8172B3'),
            ('C-format', 'Format only (EM=0)', '#937860'), ('A', 'No bridge produced', '#BFBFBF')]
    groups = [(f'ITER-RETGEN fails\n(n={n_iter}, BRGD wins)', iter_counts),
              (f'BRGD fails\n(n={n_brgd}, ITER wins)', brgd_counts)]
    x = np.arange(len(groups)); fig, ax = plt.subplots(figsize=(5.6, 4.6)); bottom = np.zeros(len(groups))
    for code, lbl, c in cols:
        vals = np.array([g[1].get(code, 0) / sum(g[1].values()) * 100 for g in groups])
        ax.bar(x, vals, 0.5, bottom=bottom, color=c, edgecolor='white', label=lbl)
        for i, v in enumerate(vals):
            if v > 4: ax.text(x[i], bottom[i] + v / 2, f'{v:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white' if c != '#BFBFBF' else '#333')
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel("% of that method's head-to-head losses", fontsize=10); ax.set_ylim(0, 100)
    ax.set_title('Failure breakdown on 2WikiMultiHopQA', fontsize=11)
    ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.spines[['top', 'right']].set_visible(False); plt.tight_layout()
    plt.savefig(OUT + '.pdf', bbox_inches='tight'); plt.savefig(OUT + '.png', dpi=200, bbox_inches='tight')
    print('saved', OUT)


if __name__ == "__main__":
    main()
