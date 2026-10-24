"""Hash final deliverables after all experiments and resource changes finish."""
from .common import ROOT,REPO,FORK,digest,write
import subprocess
import time

def main():
    assert not subprocess.check_output(['git','diff','--binary'],cwd=FORK)
    items=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(ROOT)
        if relative.parts[0] in ('raw','distributed','state'):continue
        if path.name in ('burner.log','artifact_hash_manifest.json'):continue
        if path.suffix in ('.sqlite','.nsys-rep'):continue
        items.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=digest(path)))
    write(ROOT/'artifact_hash_manifest.json',dict(sealed_wall_s=time.time(),files=items,
        runtime_clean=True,unrelated_index_sha256=subprocess.check_output('git diff --cached --binary | sha256sum',shell=True,cwd=REPO,text=True).strip().split()[0]))
    print('SEALED',len(items))

if __name__=='__main__':main()
