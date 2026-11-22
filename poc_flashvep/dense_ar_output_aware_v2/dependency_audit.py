"""Check controller dependencies against the explicitly requested publication base."""
import subprocess
import time
from .common import ROOT,REPO,BASE,write,digest,preserve

def main():
    preserve();rows=[]
    packages=('dense_ar_slo_layer_pd','dense_ar_native_reproduction','dense_ar_bi0_axis_rootcause',
        'dense_ar_bi0_shortanswer_pd','dense_ar_longbench400_pd')
    for package in packages:
        for path in sorted((REPO/'poc_flashvep'/package).glob('*.py')):
            relative=str(path.relative_to(REPO));actual=subprocess.check_output(['git','hash-object',str(path)],cwd=REPO,text=True).strip()
            expected=subprocess.run(['git','rev-parse',f'{BASE}:{relative}'],cwd=REPO,capture_output=True,text=True)
            baseline=expected.stdout.strip() if expected.returncode==0 else None
            rows.append(dict(path=relative,SHA256=digest(path),git_blob=actual,base_git_blob=baseline,base_identical=actual==baseline))
    result=dict(time=time.time(),base=BASE,rows=rows,all_base_identical=all(r['base_identical'] for r in rows),
        scope='Read-only controller dependency audit. Private GPU worker source capsules remain the exact runtime authority. '
            'No user checkout,index or historical module is modified.')
    write(ROOT/'analysis/controller_dependency_base_audit_v2.json',result)
    print('CONTROLLER_DEPENDENCIES',len(rows),'MISMATCHES',[r['path'] for r in rows if not r['base_identical']],flush=True)

if __name__=='__main__':main()
