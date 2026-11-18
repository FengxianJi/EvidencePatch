#!/usr/bin/env python3
"""Post-hoc conditional alignment; no subgroup selection by significance."""
import csv, json, math, random, hashlib, runpy
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
helpers=runpy.run_path(str(ROOT/'evaluation/analyze_primary300_score_resolution_alignment_v2.py'))
rho=helpers['spearman']
paths=[ROOT/'dataset/primary300_analysis_bundle_v1/paper_scores.csv',ROOT/'results/primary300_resolution_proxy_m3_v3/corrected_paper_metrics.csv',ROOT/'results/primary300_resolution_proxy_m3_v3/corrected_concern_records.csv']
load=lambda p:list(csv.DictReader(p.open(encoding='utf-8-sig')))
scores={r['cohort_id']:r for r in load(paths[0])}; resolutions=load(paths[1]); concerns=defaultdict(list)
for r in load(paths[2]):
 if r['paired_analysis_eligible']=='True':concerns[r['case_id']].append(r)
rows=[]
for r in resolutions:
 if not r['C5_minus_C0']:continue
 s=scores[r['case_id']]
 if s['source_sha256']!=r['source_sha256'] or s['review_text_sha256']!=r['review_text_sha256']:raise ValueError('provenance mismatch')
 rows.append(dict(score=float(s['targeted_concern_evidence_minus_baseline']),proxy=float(r['C5_minus_C0']),placebo_score=float(s['targeted_concern_evidence_minus_placebo']),base=float(s['baseline_mean_score']),count=int(r['valid_repairable_concern_count']),chars=int(s['targeted_concern_evidence_chars']),types={x['repairability'] for x in concerns[r['case_id']]}))
groups=[('all',lambda r:True),('baseline_lt3',lambda r:r['base']<3),('baseline_3_to_lt4',lambda r:3<=r['base']<4),('eligible_concerns_1_2',lambda r:r['count']<=2),('eligible_concerns_3plus',lambda r:r['count']>=3),('has_existing_evidence_concern',lambda r:'existing_evidence' in r['types']),('has_clarification_concern',lambda r:'clarification' in r['types']),('has_reanalysis_concern',lambda r:'reanalysis' in r['types'])]
rng=random.Random(20261118); out=[]
for name,fn in groups:
 g=[r for r in rows if fn(r)]
 xs=[r['score'] for r in g];ys=[r['proxy'] for r in g]
 if len(set(xs))<2 or len(set(ys))<2:continue
 draws=[]
 for _ in range(3000):
  ix=[rng.randrange(len(g)) for _ in g];a=[xs[i] for i in ix];b=[ys[i] for i in ix]
  if len(set(a))>1 and len(set(b))>1:draws.append(rho(a,b))
 draws.sort();up=[r for r in g if r['proxy']>0];flat=[r for r in g if r['proxy']==0]
 avg=lambda z:sum(z)/len(z) if z else None
 out.append(dict(group=name,n=len(g),spearman=rho(xs,ys),ci95_pointwise=[draws[int(.025*len(draws))],draws[int(.975*len(draws))]],proxy_up_papers=len(up),score_up_among_proxy_up=sum(r['score']>0 for r in up),score_up_rate_proxy_up=avg([r['score']>0 for r in up]),score_up_rate_proxy_flat=avg([r['score']>0 for r in flat]),mean_score_delta_proxy_up=avg([r['score'] for r in up]),mean_score_delta_proxy_flat=avg([r['score'] for r in flat]),spearman_C5_minus_placebo=rho([r['placebo_score'] for r in g],ys)))
result=dict(status='exploratory_posthoc',bootstrap_replicates=3000,seed=20261118,inputs_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},groups=out,limits=['Subgroups overlap; pointwise intervals are not adjusted for multiple comparisons.','Concern types and resolution labels are M3 judgments; no human ground truth.','No causal attribution; no confirmatory subgroup discovery.','Paper strata defined using baseline scores can induce mathematical coupling and regression to mean.'])
outdir=ROOT/'results/primary300_score_resolution_subgroups_v1';outdir.mkdir(exist_ok=True)
(outdir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# Conditional score / resolution-proxy alignment','','Exploratory post-hoc analysis, 284 eligible papers. Subgroups overlap; bootstrap intervals are pointwise and not multiplicity-adjusted.','','| Subgroup | n | Spearman rho | Pointwise 95% CI | Proxy-up papers | Score-up among proxy-up |','|---|---:|---:|---|---:|---:|']
for r in out:lines.append(f"| {r['group']} | {r['n']} | {r['spearman']:.3f} | [{r['ci95_pointwise'][0]:.3f}, {r['ci95_pointwise'][1]:.3f}] | {r['proxy_up_papers']} | {r['score_up_among_proxy_up']} |")
lines+=['','These outcomes measure holistic score and a selected concern-resolution proxy, not the relationship between all review content and score. Inspect the JSON for the descriptive score-up frequencies. Intervals do not establish no association; a positive subgroup interval would still require multiple-comparison control and held-out confirmation.','']
(outdir/'report.md').write_text('\n'.join(lines))
print('\n'.join(lines))
