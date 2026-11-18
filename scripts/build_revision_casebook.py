#!/usr/bin/env python3
"""Render an HTML before/after casebook from saved synthetic adapter runs."""
import hashlib,json
from html import escape
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'examples/revision_casebook';PAGES=OUT/'pages';PAGES.mkdir(exist_ok=True)
NAMES={'frozen_original':'Frozen original','one_shot_generic':'One-shot generic LLM','iterative_generic':'Iterative generic LLM','drpg_style':'DRPG-style adaptation','retrieval_only':'Retrieval only','peerify_verify':'Peerify-style verification','random_budget':'Random budget','cost_greedy':'Cost greedy','all_feasible':'All feasible within budget','oracle_selector':'Estimated-utility oracle','evidencepatch_risk_aware':'EvidencePatch: risk-aware','evidencepatch_bm25_risk_aware':'EvidencePatch: BM25 + risk-aware','evidencepatch_section_context':'EvidencePatch: section context'}
CAUTIONS={
'frozen_original':'No revision; reference condition only.',
'one_shot_generic':'Drafts a response; evidence sufficiency still needs a separate judgment.',
'iterative_generic':'Uses audit feedback to revise the draft; iteration alone does not create missing measurements.',
'drpg_style':'Explicitly decomposes concerns in a single prompted pipeline; not the original DRPG implementation.',
'retrieval_only':'An exact passage can be irrelevant: RGB storage does not measure label-noise robustness.',
'peerify_verify':'Returns a diagnostic, not a patch; no-op score copies in the study do not establish verification accuracy.',
'random_budget':'Respects feasibility but ignores relevance and scientific value.',
'cost_greedy':'Uses supplied utility/cost estimates; a cheap distractor can win when those estimates are wrong.',
'all_feasible':'Uses feasible actions within the budget; it is not an unlimited insert-everything condition.',
'oracle_selector':'Optimizes estimated utilities in a small candidate set, not future expert outcomes.',
'evidencepatch_risk_aware':'Estimates relevance/risk before selecting; estimates can still be wrong.',
'evidencepatch_bm25_risk_aware':'Receives exact-span candidates and full review context; no returned pointer constitutes a new experiment.',
'evidencepatch_section_context':'Adds retrieved source sections to the estimator; more context does not guarantee semantic validity.'}
STYLE='''body{margin:0;background:#edf2f6;color:#182d43;font:21px/1.55 Arial,sans-serif}main{width:1380px;margin:auto;padding:38px 36px 45px}h1{font-size:34px;line-height:1.2;margin:8px 0 15px}h2{font-size:27px;margin:0 0 12px}h3{font-size:21px;margin:0 0 9px}.tag{color:#1d6f72;font-size:17px;font-weight:bold;letter-spacing:1px}.scope{background:#fff7dc;border:1px solid #e3cf8e;padding:16px 20px;border-radius:12px;margin:20px 0}article{background:white;border:1px solid #cbd9e5;border-radius:16px;padding:24px;margin:22px 0}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}.panel{padding:18px;background:#f5f7f9;border:1px solid #d7e0e7;border-radius:10px}.after{background:#eef7f3}.meta{font-size:18px;color:#4a6277;margin-top:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:19px/1.5 'DejaVu Sans Mono',monospace;margin:0}a{color:#165a8c}.decision{background:#e8f1fa;padding:16px;margin:14px 0;border-radius:9px}table{border-collapse:collapse;width:100%;font-size:18px}td,th{border:1px solid #cbd9e5;padding:8px;text-align:left}footer{font-size:17px;color:#526b80;margin-top:25px}ul{padding-left:25px}'''
def page(title,body):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>'+escape(title)+'</title><style>'+STYLE+'</style></head><body><main><div class="tag">EVIDENCEPATCH / EXECUTED MECHANISM CASEBOOK</div><h1>'+escape(title)+'</h1><div class="scope">Authored synthetic paper and review. These are saved outputs from actual adapters, not the 100/300-paper study and not new efficacy evidence. No review scores are assigned.</div>'+body+'<footer>Full inputs and outputs: ../input.json and ../outputs.json. Every displayed response is traceable to a saved method output. Scientific missing evidence remains missing.</footer></main></body></html>'
def preview(text,limit=1250):
 return text if len(text)<=limit else text[:limit]+'\n\n[Display excerpt; complete output is in outputs.json.]'
def card(name,x,ctx):
 appendix=x.get('appendix_markdown','')
 after=appendix if appendix else '[No manuscript change.]'
 if name=='peerify_verify':after='Diagnostic output; manuscript unchanged.\n\n'+json.dumps(x.get('verifications',[]),indent=2,ensure_ascii=False)
 badge='LLM calls: '+str(x.get('calls',0))+' | Appendix characters: '+str(len(appendix))
 if 'actions' in x:badge+=' | Selected actions: '+(', '.join(x['actions']) or 'none')
 return '<article><h2>'+escape(NAMES[name])+'</h2><div class="columns"><div class="panel"><h3>BEFORE / shared frozen manuscript</h3><pre>'+escape(ctx['manuscript'])+'</pre><h3>AI-review-style feedback (authored)</h3><pre>'+escape(ctx['reviews'])+'</pre></div><div class="panel after"><h3>AFTER / appended response or diagnostic</h3><pre>'+escape(preview(after))+'</pre></div></div><p class="meta">'+escape(badge)+'</p><div class="decision">What this shows: '+escape(CAUTIONS[name])+'</div></article>'
def main():
 ctx=json.loads((OUT/'input.json').read_text());saved=json.loads((OUT/'outputs.json').read_text());outputs=saved['methods']
 if set(outputs)!=set(NAMES):raise RuntimeError('Expected all ten controls and all three variants before rendering')
 groups=[('generation','Three ways to generate a response',['one_shot_generic','iterative_generic','drpg_style']),('retrieval_verification','No change, retrieval, and verification',['frozen_original','retrieval_only','peerify_verify']),('selection','Four policies on the same candidate pool',['random_budget','cost_greedy','all_feasible','oracle_selector']),('evidencepatch','EvidencePatch variants on the shared example',['evidencepatch_risk_aware','evidencepatch_bm25_risk_aware','evidencepatch_section_context'])]
 for filename,title,methods in groups:
  (PAGES/(filename+'.html')).write_text(page(title,''.join(card(n,outputs[n],ctx) for n in methods)))
 for n,x in outputs.items():(PAGES/(n+'.html')).write_text(page(NAMES[n],card(n,x,ctx)))
 x=outputs['evidencepatch_bm25_risk_aware'];ratings=x.get('ratings',[])
 tbody='<table><tr><th>Candidate</th><th>Source text</th><th>Decision</th><th>Estimated relevance / risk</th></tr>'
 for c in ctx['candidates']:
  r=next((z for z in ratings if z['candidate_id']==c['candidate_id']),{})
  tbody+='<tr><td>'+escape(c['candidate_id'])+'</td><td>'+escape(c['quote'])+'</td><td>'+escape(r.get('decision','missing'))+'</td><td>'+escape(str(r.get('p_quote_relevant'))+' / '+str(r.get('risk')))+'</td></tr>'
 tbody+='</table>'
 body=card('evidencepatch_bm25_risk_aware',x,ctx)+'<article><h2>Inspect the estimates before the action</h2>'+tbody+'<p>All three passages are exact source strings. Exactness alone cannot distinguish the seed-setting answer from the RGB distractor.</p></article>'
 for case in saved['boundary_cases']:
  brief={k:v for k,v in case.items() if k!='output'};o=case['output']
  brief['actions']=o.get('actions',[]);brief['appendix_markdown']=o.get('appendix_markdown','');brief['interpretation']=o.get('interpretation','Evidence pointers only; inspect full ratings in outputs.json.')
  body+='<article><h2>'+escape(case['case'])+'</h2><pre>'+escape(json.dumps(brief,indent=2))+'</pre></article>'
 (PAGES/'evidencepatch_focus.html').write_text(page('EvidencePatch close-up: evidence, distractors, missing experiments, budget',body))
 links=''.join('<li><a href="pages/'+file+'.html">'+title+'</a></li>' for file,title,_ in groups)
 links+='<li><a href="pages/evidencepatch_focus.html">EvidencePatch detailed boundary cases</a></li>'
 links+=''.join('<li><a href="pages/'+n+'.html">'+name+'</a></li>' for n,name in NAMES.items())
 (OUT/'index.html').write_text(page('Before / after: all method outputs','<p>Open a comparison or an individual method below.</p><ul>'+links+'</ul>').replace('href="pages/','href="pages/'))
 forms=json.loads((OUT/'edit_surfaces.json').read_text())
 body=''
 for form in forms['forms']:
  body+='<article><h2>'+escape(form['name'])+'</h2><div class="columns"><div class="panel"><h3>BEFORE</h3><pre>'+escape(form['before'])+'</pre></div><div class="panel after"><h3>AFTER</h3><pre>'+escape(form['after'])+'</pre></div></div><p>'+escape(form['effect'])+'</p></article>'
 rendered=page('Two editing surfaces: rewriting and adding a navigation anchor',body)
 rendered=rendered.replace('Authored synthetic paper and review. These are saved outputs from actual adapters, not the 100/300-paper study and not new efficacy evidence. No review scores are assigned.',escape(forms['scope']))
 (PAGES/'edit_surfaces.html').write_text(rendered)
 print('Rendered HTML casebook for',len(outputs),'methods and four boundary cases.')
if __name__=='__main__':main()
