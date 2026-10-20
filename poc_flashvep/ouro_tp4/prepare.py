"""Freeze the prior R4 cases and token feedback before a new GPU run."""
import datetime
import gzip
import subprocess
from .contract import *


def main():
    assert not (ROOT/'workload.json').exists()
    frozen=[]
    for f in sorted((OLD/'raw/primary_loop_pp4_v5_r0').glob('*.json.gz')):
        with gzip.open(f,'rt') as stream:d=json.load(stream)
        if d['depth']!=4:continue
        c=d['case'];requests=[]
        for r in d['outputs']:
            requests.append(dict(id=r['id'],input_ids=r['input_ids'],teacher_ids=r['ids']))
        frozen.append(dict(index=len(frozen),R=4,C=c['C'],P=c['P'],O=c['O'],requests=requests,
            source=str(f.relative_to(PROJECT)),source_sha256=sha(f)))
    assert len(frozen)==21
    write(ROOT/'workload.json',dict(cases=frozen,scope='Hardware-only frozen feedback. Not accuracy or free-running generation.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=PROJECT).strip()))
    (ROOT/'spec_frozen.md').write_text(SPEC.read_text())
    write(ROOT/'preflight.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        requested_spec='/home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_tp4_parallelism_discovery_spec.md',
        actual_spec=str(SPEC),spec_sha256=sha(SPEC),model=str(MODEL),revision=REVISION,limits=TOL,
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=PROJECT).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True,cwd=PROJECT).strip(),
        old_source_hashes={str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'poc_flashvep/ouro_request_pp').glob('*.py')},
        old_report_hashes={name:sha(OLD/name) for name in ['analysis.md','benchmark_summary.md','correctness.md','timing_contract.md','reproduce.md']},
        physical_GPUs=GPUS,push_authority='Final user instruction explicitly requests GitHub push.'))
    env={k:os.environ.get(k) for k in ['NCCL_ALGO','NCCL_PROTO','NCCL_P2P_DISABLE','NCCL_IB_DISABLE']}
    write(ROOT/'environment_preflight.json',dict(NCCL_overrides=env,
        gpu=subprocess.check_output(['nvidia-smi','-i','4,5,6,7','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv'],text=True),
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        nvlink=subprocess.check_output(['nvidia-smi','nvlink','-s','-i','4,5,6,7'],text=True)))
    print('FROZEN_R4',len(frozen),flush=True)


if __name__=='__main__':main()
