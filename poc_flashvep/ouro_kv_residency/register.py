"""Register corpus families, controls, tolerances, and immutable job plans."""
import json
import subprocess
import urllib.request
from .common import ROOT,FORK,MODEL,FIXTURES,write,digest,UUIDS,UPSTREAM

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    FIXTURES.mkdir(parents=True,exist_ok=True)
    if (ROOT/'source_lock.json').exists():
        assert json.loads((ROOT/'source_lock.json').read_text())['model_revision']==MODEL.name
        print('ALREADY_REGISTERED. Existing contracts remain unchanged.')
        return
    families=[('TRAIN',11),('TRAIN',1661),('VALIDATION',84),('VALIDATION',1342),('FINAL',2701),('FINAL',2600)]
    rows=[]
    for split,book in families:
        path=ROOT/'corpus'/f'pg{book}.txt'
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            url=f'https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt'
            path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        rows.append(dict(split=split,family=f'gutenberg:{book}',path=str(path.relative_to(ROOT)),
            sha256=digest(path),source=f'https://www.gutenberg.org/ebooks/{book}'))
    write(ROOT/'source_family_manifest.json',dict(registered_before_tests=True,families=rows,
        procedure='Use unique contiguous book passages. No repeated-token padding. Token slicing is a hardware context control.'))
    write(ROOT/'prospective_contract.json',dict(model_revision=MODEL.name,source=UPSTREAM,
        native_baseline='UNMODIFIED RLT Async + Mixed + FA4 + CUDA Graph',
        precision='BF16',loops=4,context_limit=65536,kv_namespaces=96,
        evidence=['LIVE_MEASURED_OPERATOR','LIVE_MEASURED_SERVING','MODEL_UPPER_BOUND','SIMULATED','NOT_RUN'],
        operator_tolerance=dict(relative_L2=.03,absolute_max=.125,finite=True,
            meaning='Freeze before TRAIN GPU tests. Absolute error uses original output scale. No post-hoc relaxation.'),
        layout_tolerance=dict(KV_byte_exact=True,logit_relative_L2=.03,output_token_exact=True),
        contention_threshold_fraction=.03,profiling='Never sum kernel service into clean E2E shares.',
        no_gates_on_single_negative=True,mandatory='Physical CP2 operator, followed by feasible CP4/8 operator.',
        phase5='Require incremental effect versus the identical generic depth-agnostic CP implementation.',
        final='No full-model serving or accuracy claim for standalone operators.'))
    facts=dict(pin=UPSTREAM,path=str(FORK),diff=subprocess.check_output(['git','diff','--binary'],cwd=FORK,text=True),
        model_revision=MODEL.name,model_config=json.loads((MODEL/'config.json').read_text()),
        GPUs=UUIDS,baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    assert not facts['diff']
    write(ROOT/'source_lock.json',facts)
    for name,command in [('topology',['nvidia-smi','topo','-m']),('nvlink',['nvidia-smi','nvlink','-s']),
        ('gpu_identity',['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,memory.total','--format=csv']),
        ('numa',['lscpu'])]:
        result=subprocess.run(command,text=True,capture_output=True)
        (ROOT/f'{name}.txt').write_text(result.stdout+result.stderr)
    jobs=[]
    # Independent starts rotate physical devices. Over-capacity cases stop before allocation.
    for i,(p,n) in enumerate(( (p,n) for p in (1024,2048,4096,8192,16384,32768,65408) for n in (1,2,4,8) )):
        if n*((p+128+15)//16)*4>18432:
            continue
        for start in range(2):
            jobs.append(dict(id=f'grid_P{p}_N{n}_s{start}',kind='timing',gpu=(i+start*3)%8,
                P=p,N=n,cap=64,repeats=2,split='TRAIN' if start==0 else 'VALIDATION',timeout_s=3600))
    write(ROOT/'plans'/'grid.json',dict(gpus=list(range(8)),jobs=jobs))
    write(ROOT/'plans'/'smoke.json',dict(gpus=[0],jobs=[dict(id='native_smoke_P2048_N1',gpu=0,
        kind='timing',P=2048,N=1,cap=16,repeats=2,split='TRAIN',timeout_s=900)]))

if __name__=='__main__':main()
