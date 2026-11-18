#!/usr/bin/env python3
"""Run one method on user-provided material; default example makes no API call."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from method import get_method, list_methods
from planner import budget

def main():
    p=argparse.ArgumentParser(description=__doc__)
    proposed=['evidencepatch_risk_aware','evidencepatch_bm25_risk_aware','evidencepatch_section_context']
    p.add_argument('--input',type=Path,default=ROOT/'examples/toy_context.json')
    p.add_argument('--method',choices=list_methods()+proposed,default='retrieval_only')
    p.add_argument('--api-config',type=Path)
    p.add_argument('--allow-api',action='store_true',help='Explicitly allow paid calls with provided API config')
    args=p.parse_args()
    ctx=json.loads(args.input.read_text());ctx['planner_api']=budget
    if args.allow_api:
        if not args.api_config:p.error('--allow-api requires --api-config')
        from executor.minimax import MiniMaxClient
        config=json.loads(args.api_config.read_text())
        key_path=Path(config['key_file']).expanduser()
        if not key_path.is_absolute():key_path=args.api_config.resolve().parent/key_path
        config['key_file']=str(key_path)
        logs=ROOT/'local_runs';logs.mkdir(exist_ok=True)
        client=MiniMaxClient(config,ROOT,logs/'api_usage.jsonl')
        def call(**kw):
            return client.complete([{'role':'system','content':kw['system']},
                                    {'role':'user','content':kw['user']}],kw['purpose'],kw['paper_id'])
        ctx['llm_call']=call
    if args.method in proposed:
        from importlib import import_module
        run=import_module('method.methods.'+args.method+'.method').run
    else:run=get_method(args.method)
    print(json.dumps(run(ctx),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
