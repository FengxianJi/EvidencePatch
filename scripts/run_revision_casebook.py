#!/usr/bin/env python3
"""Run real adapters on an authored synthetic manuscript. No study-score claims.
Offline controls run by default. Paid model methods require --allow-api and --api-config.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from method import get_method, list_methods
from planner import budget
from verifier.provenance import normalized_contains
OUT=ROOT/'examples/revision_casebook'
PROPOSED=['evidencepatch_risk_aware','evidencepatch_bm25_risk_aware','evidencepatch_section_context']
API_METHODS=['one_shot_generic','iterative_generic','drpg_style','peerify_verify',*PROPOSED]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--allow-api',action='store_true');p.add_argument('--api-config',type=Path)
 a=p.parse_args();ctx=json.loads((OUT/'input.json').read_text());ctx['planner_api']=budget
 saved_path=OUT/'outputs.json'
 saved=json.loads(saved_path.read_text()) if saved_path.exists() else {'scope':'Authored synthetic mechanism demonstration; actual adapter outputs, no experimental scores.','input_sha256':hashlib.sha256((OUT/'input.json').read_bytes()).hexdigest(),'methods':{}}
 if saved['input_sha256']!=hashlib.sha256((OUT/'input.json').read_bytes()).hexdigest():raise ValueError('Input changed; choose a separate output version')
 if a.allow_api:
  if not a.api_config:p.error('--api-config is required with --allow-api')
  from executor.minimax import MiniMaxClient
  config=json.loads(a.api_config.read_text());key=Path(config['key_file']).expanduser()
  if not key.is_absolute():key=a.api_config.resolve().parent/key
  config['key_file']=str(key);logs=ROOT/'local_runs';logs.mkdir(exist_ok=True)
  client=MiniMaxClient(config,ROOT,logs/'casebook_api_usage.jsonl')
  calls=[0]
  def call(**kw):
   calls[0]+=1
   if calls[0]>8:raise RuntimeError('Casebook client-call cap reached')
   result=client.complete([{'role':'system','content':kw['system']},{'role':'user','content':kw['user']}],kw['purpose'],kw['paper_id'],request_options={'max_tokens':2600,'thinking':{'type':'disabled'}})
   return {'content':result['content']}
  ctx['llm_call']=call
 for name in [*list_methods(),*PROPOSED]:
  if name in saved['methods']:continue
  if name in API_METHODS and not a.allow_api:continue
  run=importlib.import_module('method.methods.'+name+'.method').run if name in PROPOSED else get_method(name)
  output=run(copy.copy(ctx))
  saved['methods'][name]=output;saved_path.write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n')
  print(name,'saved; appendix chars',len(output.get('appendix_markdown','')),flush=True)
 # Four distinct EvidencePatch boundary situations, including the exact-source prefilter.
 from method.methods.evidencepatch_bm25_risk_aware.method import run
 cases=[]
 for name,candidates,b in [('missing_experiment_no_candidate',[],1),('bad_quote_rejected',[{'candidate_id':'bad','quote':'We achieved 99 percent robustness.','claim':'How robust is the model?','cost':1}],1)]:
  accepted=[x for x in candidates if normalized_contains(ctx['manuscript'],x['quote'])]
  output=run({**ctx,'candidates':accepted,'budget':b})
  cases.append({'case':name,'candidate_count_before_provenance_filter':len(candidates),'candidate_count_after_provenance_filter':len(accepted),'output':output})
 if 'evidencepatch_bm25_risk_aware' in saved['methods']:
  cases.append({'case':'evidence_present_limited_budget','output':saved['methods']['evidencepatch_bm25_risk_aware']})
  # Reuse saved estimates, recompute a zero-budget plan locally; not another model call.
  cases.append({'case':'zero_budget_plan','output':{'actions':budget.greedy_cost_plan([budget.PatchCandidate(candidate_id='seed',cost=1,expected_benefit=.9)],0),'budget':0,'interpretation':'Budget feasibility check with an illustrative utility; no model rescore.'}})
 saved['boundary_cases']=cases;saved_path.write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n')
 print('Completed methods:',len(saved['methods']),flush=True)
if __name__=='__main__':main()
