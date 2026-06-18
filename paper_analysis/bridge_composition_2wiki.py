"""Bridge composition on 2WikiMultiHopQA (mirror of the MuSiQue Figure 4).
hop-intermediate is defined from the original 2Wiki `evidences` reasoning-chain triples
(joined to the run by id). Contextual is split into evidence/distractor by graph connectivity.
Run from project root: .venv/bin/python paper_analysis/bridge_composition_2wiki.py
"""
import json, re, string, pickle, sys, os, logging
logging.disable(logging.INFO); sys.path.insert(0, os.getcwd())
import numpy as np
import evaluate_musique_ner_pipeline as ner
sys.modules['__main__'].NERIndex = ner.NERIndex
from sentence_transformers import SentenceTransformer

INDEX = "outputs/2wikimultihopqa_ner_pipeline_eval/global_ner_index.pkl"
RUN = "outputs/2wikimultihopqa_ner_pipeline_eval/comparison_results_rounds3_20260519_225606_28808.json"
ROOT = "2wikimultihopqa.json"
ORIG = "reproduce/dataset/2wikimultihopqa.json"
OUT = "outputs/bridge_composition_2wiki"


class EmbWrap:
    def __init__(self): self.m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    def batch_encode(self, t, **k): return np.asarray(self.m.encode(t, show_progress_bar=False))


def sq(a):
    a = a or ''; ex = set(string.punctuation)
    a = ''.join(c for c in a.lower() if c not in ex); a = re.sub(r'\b(a|an|the)\b', ' ', a)
    return ' '.join(a.split())


def main():
    idx = pickle.load(open(INDEX, 'rb')); idx.embedding_model = EmbWrap()
    g = idx.graph; pidx = set(idx.passage_node_idxs)
    p2v = {sq(g.vs[v]['content']): v for v in pidx if g.vs[v]['content']}
    root = json.load(open(ROOT))
    oid = {o['_id']: o for o in json.load(open(ORIG))}
    d = json.load(open(RUN))['results']

    names = set()
    for r in d:
        for rd in (r.get('round_diagnostics') or []):
            names |= set(rd.get('new_discovered_entities') or [])
    resolved = ner._resolve_entities_in_graph(idx, [n for n in names if n], threshold=0.55)
    n2v = {n: v[0] for n, v in resolved.items()}

    C = {'Answer': 0, 'Hop-intermediate': 0, 'Query': 0, 'Contextual (evidence)': 0, 'Contextual (distractor)': 0}
    for r in d:
        m = root[r['idx']]; o = oid.get(m['id'], {})
        final = sq(m.get('answer', '')); aliases = [sq(a) for a in (m.get('answer_aliases') or [])]
        q = sq(m.get('question', ''))
        # hop entities = subj/obj in evidences, excluding the answer
        hop = set()
        for ev in (o.get('evidences') or []):
            if isinstance(ev, (list, tuple)) and len(ev) >= 3:
                for e in (ev[0], ev[2]):
                    if sq(e) and sq(e) != final: hop.add(sq(e))
        gv = set(p2v[sq(p['paragraph_text'])] for p in m['paragraphs'] if p.get('is_supporting') and sq(p['paragraph_text']) in p2v)
        raw = set()
        for rd in (r.get('round_diagnostics') or []): raw |= set(rd.get('new_discovered_entities') or [])
        for name in {x for x in raw if x}:
            b = sq(name)
            if b == final or b in final or final in b or any(b == a or b in a or a in b for a in aliases if a): C['Answer'] += 1
            elif any(b == h or b in h or h in b for h in hop if h): C['Hop-intermediate'] += 1
            elif b in q: C['Query'] += 1
            else:
                vid = n2v.get(name)
                if vid is not None and (set(g.neighbors(vid)) & gv): C['Contextual (evidence)'] += 1
                else: C['Contextual (distractor)'] += 1
    tot = sum(C.values())
    print('2Wiki counts:', C, 'total', tot)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    order = ['Contextual (distractor)', 'Hop-intermediate', 'Contextual (evidence)', 'Query', 'Answer']
    vals = [C[k] / tot * 100 for k in order]
    cols = ['#E0A890', '#4C72B0', '#DD8452', '#55A868', '#C44E52']; hatch = ['////', '', '', '', '']
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars = ax.bar(range(len(order)), vals, color=cols, edgecolor='white', width=0.66)
    for i, bar in enumerate(bars):
        bar.set_hatch(hatch[i]); ax.text(bar.get_x() + bar.get_width() / 2, vals[i] + 0.8, f'{vals[i]:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['Contextual\n(distractor)', 'Hop-\nintermediate', 'Contextual\n(evidence)', 'Query', 'Answer'], fontsize=9.5)
    ax.set_ylabel('% of bridges', fontsize=11); ax.set_ylim(0, max(vals) + 6)
    ax.set_title(f'Composition of bridge entities (2WikiMultiHopQA, {tot} bridges / 1000 q)', fontsize=10.5)
    ax.spines[['top', 'right']].set_visible(False); plt.tight_layout()
    plt.savefig(OUT + '.pdf', bbox_inches='tight'); plt.savefig(OUT + '.png', dpi=200, bbox_inches='tight')
    print('saved', OUT)


if __name__ == "__main__":
    main()
