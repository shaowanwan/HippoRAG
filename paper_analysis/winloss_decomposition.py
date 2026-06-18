"""BRGD vs ITER-RETGEN per-question win/loss (NER backbone) + stacked-bar figure.
Win = exactly one method correct (per-question EM). Run from project root.
"""
import json
import numpy as np

PAIRS = {
    'MuSiQue': ('outputs/musique_ner_pipeline_eval/comparison_results_rounds3_20260421_143341_41732.json',
               'outputs/musique_ner_pipeline_eval/comparison_results_iterretgen_rounds3_20260606_155206_15547.json'),
    '2WikiMultiHopQA': ('outputs/2wikimultihopqa_ner_pipeline_eval/comparison_results_rounds3_20260519_225606_28808.json',
                        'outputs/2wikimultihopqa_ner_pipeline_eval/comparison_results_iterretgen_rounds3_20260606_155204_15542.json'),
}


def em_map(path, key='ner_em'):
    return {r['idx']: r[key] for r in json.load(open(path))['results']}


def main():
    stats = {}
    for ds, (bf, itf) in PAIRS.items():
        b, it = em_map(bf), em_map(itf)
        ids = [i for i in b if i in it]
        bc = sum(1 for i in ids if b[i] and it[i])
        bw = sum(1 for i in ids if b[i] and not it[i])
        iw = sum(1 for i in ids if it[i] and not b[i])
        bwr = sum(1 for i in ids if not b[i] and not it[i])
        stats[ds] = dict(bc=bc, bw=bw, iw=iw, bwr=bwr, net=bw - iw)
        print(f'{ds} (n={len(ids)}): both-correct={bc} BRGD-win={bw} ITER-win={iw} both-wrong={bwr} net=+{bw-iw}')

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ds = list(stats.keys()); x = np.arange(len(ds))
    segs = [('Both correct', 'bc', '#55A868'), ('BRGD win', 'bw', '#4C72B0'),
            ('ITER-RETGEN win', 'iw', '#DD8452'), ('Both wrong', 'bwr', '#BFBFBF')]
    fig, ax = plt.subplots(figsize=(6.6, 4.4)); bottom = np.zeros(len(ds))
    for name, k, c in segs:
        vals = np.array([stats[d][k] for d in ds])
        ax.bar(x, vals, 0.5, bottom=bottom, color=c, edgecolor='white', label=name)
        for i, v in enumerate(vals):
            if v > 25:
                ax.text(x[i], bottom[i] + v / 2, str(v), ha='center', va='center', fontsize=10,
                        fontweight='bold', color='white' if c != '#BFBFBF' else '#333')
        bottom += vals
    for i, d in enumerate(ds):
        ax.text(x[i], 1015, f"net +{stats[d]['net']}", ha='center', fontsize=11, fontweight='bold', color='#333')
    ax.set_xticks(x); ax.set_xticklabels(ds, fontsize=11)
    ax.set_ylabel('Questions (out of 1000)', fontsize=11); ax.set_ylim(0, 1060)
    ax.set_title('BRGD vs ITER-RETGEN: per-question outcomes (NER backbone)', fontsize=11)
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.spines[['top', 'right']].set_visible(False); plt.tight_layout()
    plt.savefig('outputs/winloss_decomposition.pdf', bbox_inches='tight')
    plt.savefig('outputs/winloss_decomposition.png', dpi=200, bbox_inches='tight')
    print('saved outputs/winloss_decomposition')


if __name__ == "__main__":
    main()
