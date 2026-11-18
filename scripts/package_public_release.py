#!/usr/bin/env python3
"""Build deterministic archives of tracked public code/results and an optional blog snapshot.
This never traverses untracked inputs, API logs, ignored caches, or the research workspace.
"""
from __future__ import annotations
import argparse,gzip,hashlib,io,json,re,subprocess,tarfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def tracked_snapshot(repo:Path):
    status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    if status.strip():raise RuntimeError('Commit tracked/untracked public changes before packaging: '+str(repo))
    rev=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    names=subprocess.check_output(['git','-C',str(repo),'ls-files','-z']).decode().split('\0')
    files={}
    for name in names:
        if not name:continue
        p=repo/name
        if p.is_symlink():raise RuntimeError('Symlinks require explicit release review: '+name)
        if p.suffix=='.key' or p.name=='.env' or any(x in p.relative_to(repo).parts for x in ['__pycache__','node_modules','local_runs','.venv']):raise RuntimeError('Private/cache path in tracked files: '+name)
        b=p.read_bytes()
        if re.search(rb'(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',b):raise RuntimeError('Token-like data in '+name)
        files[name]=b
    return rev,files

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',default='v0.1.0');p.add_argument('--blog-repo',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'dist')
    a=p.parse_args()
    if not re.fullmatch(r'v\d+\.\d+\.\d+',a.version):p.error('Use a semantic version such as v0.1.0')
    rev,files=tracked_snapshot(ROOT)
    versions={'code_commit':rev}
    if a.blog_repo:
        blogrev,blogfiles=tracked_snapshot(a.blog_repo.resolve())
        versions['blog_commit']=blogrev
        files.update({'blog-site/'+k:v for k,v in blogfiles.items()})
    files['README_BUNDLE.md']=('''# EvidencePatch public resources bundle

The code, public numeric results, figures, and authored casebook are at the archive root.
If present, `blog-site/` is the complete tracked source snapshot of the final bilingual blog.
No original manuscripts/reviews, credentials, raw study API responses, or unfinished training artifacts are included.

```bash
python3 scripts/verify_release.py
python3 scripts/reproduce_results.py
python3 scripts/run_method.py
```

See `docs/RELEASE_SCOPE.md` for available reproduction levels and license-file status.
The casebook is an authored synthetic demonstration, not research-paper evidence.
This archive packages the declared public release scope, not all original study inputs.
''').encode()
    manifest={'version':a.version,**versions,'scope':'Tracked public release plus optional tracked final blog snapshot',
              'files':{name:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for name,b in sorted(files.items())}}
    files['BUNDLE_INFO.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    stem='EvidencePatch-'+a.version+'-public-resources';prefix='EvidencePatch-'+a.version+'/'
    zpath=out/(stem+'.zip');tpath=out/(stem+'.tar.gz')
    for path in [zpath,tpath]:
        if path.exists():raise RuntimeError('Refusing to overwrite existing archive: '+str(path))
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,b in sorted(files.items()):
            info=zipfile.ZipInfo(prefix+name,(2026,11,18,0,0,0));info.external_attr=0o100644<<16
            archive.writestr(info,b,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with tpath.open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w') as archive:
                for name,b in sorted(files.items()):
                    info=tarfile.TarInfo(prefix+name);info.size=len(b);info.mode=0o644;info.mtime=0
                    archive.addfile(info,io.BytesIO(b))
    sums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [zpath,tpath]}
    (out/'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n,h in sums.items()))
    (out/'bundle_inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Verify every ZIP member against its intended bytes before reporting success.
    with zipfile.ZipFile(zpath) as archive:
        if archive.testzip() is not None:raise RuntimeError('ZIP CRC check failed')
        for name,b in files.items():
            if archive.read(prefix+name)!=b:raise RuntimeError('ZIP data mismatch')
    print(json.dumps({'version':a.version,**versions,'files':len(files),'archives':{p.name:p.stat().st_size for p in [zpath,tpath]},'output':str(out)},indent=2))
if __name__=='__main__':main()
