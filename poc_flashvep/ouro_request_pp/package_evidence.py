"""Preserve executable commands, source hashes, and evidence tests."""
import datetime
import json
import shlex
import shutil
import subprocess
import sys
from .contract import ROOT,PROJECT,sha,write


def main():
    snapshot=ROOT/'burner_restore_snapshot.log'
    if not snapshot.exists() and (ROOT/'burner_restore.log').exists():
        shutil.copyfile(ROOT/'burner_restore.log',snapshot)
    commands=['#!/usr/bin/env bash','# Command ledger. Review the GPU lease before any GPU command.',
              '# Named runs refuse to overwrite existing evidence. These are recorded invocations, not an automatic rerun.',
              'set -euo pipefail',f'cd {shlex.quote(str(PROJECT))}']
    for f in sorted((ROOT/'commands').glob('*.json')):
        d=json.loads(f.read_text())
        if 'argv' not in d:continue
        commands.append('\n# '+f.name)
        env={'CUDA_VISIBLE_DEVICES':'4,5,6,7',**d.get('environment',{})}
        prefix=['env']+[f'{k}={v}' for k,v in env.items()]
        commands.append(shlex.join(prefix+d['argv']))
    (ROOT/'commands.sh').write_text('\n'.join(commands)+'\n')
    result=subprocess.run([sys.executable,'-m','unittest','poc_flashvep.ouro_request_pp.test_routes',
                           'poc_flashvep.ouro_request_pp.test_evidence','-v'],cwd=PROJECT,
                          text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (ROOT/'tests.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout
    files=[]
    for directory in [ROOT,PROJECT/'poc_flashvep/ouro_request_pp']:
        for f in sorted(directory.rglob('*')):
            if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.sqlite':continue
            if f.name in ['artifact_manifest.json','burner_restore.log']:continue
            files.append(dict(path=str(f.relative_to(PROJECT)),bytes=f.stat().st_size,sha256=sha(f)))
    write(ROOT/'artifact_manifest.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=PROJECT,text=True).strip(),
        base_SHA=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        mutable_exclusion='burner_restore.log can grow after the burner starts',files=files))
    print('PACKAGED',len(files),'files. Evidence tests passed.')


if __name__=='__main__':main()
