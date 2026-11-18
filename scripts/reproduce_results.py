#!/usr/bin/env python3
"""Recompute descriptive endpoints from anonymous numeric records; no API/GPU."""
import csv, json, math, random
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def mean(xs): return sum(xs) / len(xs)
def ranks(xs):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    out = [0.] * len(xs); i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]: j += 1
        for k in range(i, j): out[order[k]] = (i + 1 + j) / 2
        i = j
    return out

def corr(xs, ys):
    mx, my = mean(xs), mean(ys)
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / den if den else None

def main():
    with (ROOT/'results/numeric/paper_endpoints.csv').open() as f: rows = list(csv.DictReader(f))
    paired = [r for r in rows if r['resolution_delta'] != '']
    xs = [float(r['score_delta']) for r in paired]
    ys = [float(r['resolution_delta']) for r in paired]
    signs = Counter(((x>0)-(x<0), (y>0)-(y<0)) for x,y in zip(xs,ys))
    transitions = list(csv.DictReader((ROOT/'results/numeric/concern_transitions.csv').open()))
    result = {
        'cohort_papers': len(rows), 'eligible_papers': len(paired),
        'eligible_concerns': sum(int(r['eligible_concerns']) for r in rows),
        'primary_score_mean_delta': mean([float(r['score_delta']) for r in rows]),
        'eligible_score_mean_delta': mean(xs), 'resolution_proxy_mean_delta': mean(ys),
        'spearman_rho': corr(ranks(xs), ranks(ys)),
        'direction_counts_score_then_proxy': {f'{a:+d},{b:+d}':n for (a,b),n in sorted(signs.items())},
        'concern_transition_total':sum(int(r['count']) for r in transitions),
        'limits':'Same-family AI proxies; not expert-confirmed resolution. No manuscript or review text is included.'
    }
    if (result['cohort_papers'],result['eligible_papers'],result['eligible_concerns']) != (300,284,873):
        raise ValueError('Release coverage mismatch')
    if signs[(-1,1)] != 25 or signs[(0,1)] != 20 or signs[(1,1)] != 10:
        raise ValueError('Release direction counts mismatch')
    output=ROOT/'reproduced';output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
