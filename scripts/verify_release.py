#!/usr/bin/env python3
"""Verify public artifact checksums, numeric schemas, and absence of key files."""
import ast, csv, hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for name,expected in manifest['sha256'].items():
        p=ROOT/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Artifact checksum mismatch: '+name)
    for path in ROOT.rglob('*.py'):
        if '.git' not in path.parts:ast.parse(path.read_text(),filename=str(path))
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.parts or '__pycache__' in path.parts:continue
        if path.suffix=='.key' or path.name=='.env':raise ValueError('Private file detected')
    with (ROOT/'results/numeric/paper_endpoints.csv').open() as f:
        r=csv.DictReader(f)
        if set(r.fieldnames)!=set(['baseline_score','score_delta','score_delta_vs_placebo','resolution_delta','eligible_concerns']):
            raise ValueError('Unexpected numeric schema')
        rows=list(r)
        if len(rows)!=300:raise ValueError('Expected 300 anonymous rows')
        for row in rows:
            for value in row.values():
                if value!='':float(value)
    print(f"Verified {len(manifest['sha256'])} release artifacts, Python syntax, and anonymous numeric schema")

if __name__=='__main__':main()
