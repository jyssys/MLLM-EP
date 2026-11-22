"""Freeze the common QPS grid and a shared immutable baseline source capsule."""
import zipfile
from .common import ROOT,REPO,write,digest
from .experiment import create


def main():
    folder=ROOT/'references/source_capsules';folder.mkdir(exist_ok=True)
    capsule=folder/'motivation_baseline_v1.zip';assert not capsule.exists()
    files=[REPO/'poc_flashvep/__init__.py']
    for package in ('dense_ar_longbench400_pd','dense_ar_bi0_shortanswer_pd','dense_ar_slo_layer_pd'):
        files.extend(p for p in (REPO/'poc_flashvep'/package).glob('*') if p.suffix in ('.py','.cc','.h'))
    with zipfile.ZipFile(capsule,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:bundle.write(path,str(path.relative_to(REPO)))
    grid=[.4,.7,1.,1.4,1.8,2.2,2.8,3.5,4.5];jobs=[]
    for i,qps in enumerate(grid):
        for method in (('Native','Agg') if i%2==0 else ('Agg','Native')):
            name=f'Motive400_{method}_q{qps:g}_seed2026112161_v1'
            create(name,method,'longbench_balanced400_v1.json',pair=(0,1),qps=qps,seed=2026112161,
                execution_family='OFFICIAL_DEFAULT',continuous_source_credit_limit=0,
                phase='MOTIVATION_DISCOVERY',source_capsule=capsule.name,source_capsule_SHA256=digest(capsule))
            jobs.append(name)
    write(ROOT/'jobs/motivation400_discovery_v1.json',dict(jobs=jobs,stop_on_failure=True))
    write(ROOT/'configs/motivation400_grid_frozen_v1.json',dict(QPS=grid,primary_pair=[0,1],n=400,
        seed=2026112161,source_capsule_SHA256=digest(capsule),important_confirmation_QPS=[1.8,2.8,4.5],
        methods=['Agg','Native'],execution_family='Installed standard continuous scheduler,default compile/graphs.',
        source_credit_limit='Uncapped standard baseline. No partition adapter.',
        scope='Grid follows the64-request pilots. QPS4 raises Native TTFT p95 to7.373s while Agg p95 is1.845s. '
            'This is grid discovery,not a final crossover claim. Frozen requests and SLO stay unchanged. '
            'The source capsule prevents later implementation work from changing these baseline runs.'))
    print('MOTIVATION_DISCOVERY_JOBS',len(jobs))


if __name__=='__main__':main()
