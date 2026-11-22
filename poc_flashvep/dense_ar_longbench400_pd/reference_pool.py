"""Run eight independent natural-length references under the global guard."""
import fcntl
import argparse
import json
import os
import signal
import subprocess
import time
import zipfile
from .common import ROOT,SOURCE,REPO,PYTHON,QUIET_LOCK,initialize,preserve,resources,write,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='v2');args=parser.parse_args()
    assert args.version.isalnum(),args.version
    initialize()
    while not (ROOT/'state/candidate_input_census_complete.json').exists():
        from .common import dispatch_allowed
        assert dispatch_allowed()
        time.sleep(3)
    folder=ROOT/'raw'/f'CandidateNaturalReference_{args.version}';assert not folder.exists(),'Preserve prior evidence and use a new run version'
    folder.mkdir();owner=resources();children=[];handles=[];interrupted=[False]
    def stop(*_):interrupted[0]=True;write(folder/'STOP_AFTER_REQUEST.json',dict(time=time.time(),reason='Owned controller signal. Preserve each completed request.'))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        preserve();owner.stop(list(range(8)))
        try:
            archive=folder/'source.zip'
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(SOURCE.glob('*.py')):bundle.write(path,str(path.relative_to(REPO)))
            write(folder/'source_provenance.json',dict(source_archive_SHA256=digest(archive),
                candidate_pool_SHA256=digest(ROOT/'data/candidate_pool_original.json'),
                model_revision='b968826d9c46dd6066d109eabc6255188de91218',
                original_model_forward=True,numerical_mode='BI0',scope='Eight independent C1 references,not eight timing replications. No cross-GPU throughput comparison.'))
            inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total','--format=csv'],text=True)
            write(folder/'GPU_inventory.json',dict(text=inventory))
            for gpu in range(8):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),VLLM_BATCH_INVARIANT='0',
                    VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',
                    PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'])
                command=[str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.reference_worker','--shard',str(gpu),'--version',args.version]
                handle=(ROOT/'logs'/f'CandidateReference_GPU{gpu}_{args.version}.log').open('w');handles.append(handle)
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child)
                write(folder/f'command_GPU{gpu}.json',dict(command=command,pid=child.pid,GPU=gpu,time=time.time(),
                    source_archive_SHA256=digest(archive),environment={key:env[key] for key in ('CUDA_VISIBLE_DEVICES','VLLM_BATCH_INVARIANT','OMP_NUM_THREADS','PATH')}))
            write(ROOT/'state/active_reference_jobs.json',dict(time=time.time(),pids=[p.pid for p in children],GPUs=list(range(8))))
            began=time.monotonic()
            while any(p.poll() is None for p in children):
                failed=[p for p in children if p.poll() not in (None,0)]
                if failed:stop()
                if time.monotonic()-began>5*3600:stop()
                if interrupted[0] and time.monotonic()-began>5*3600+300:raise TimeoutError('Reference checkpoint drain timeout')
                time.sleep(2)
            exits={gpu:p.returncode for gpu,p in enumerate(children)}
            write(folder/'worker_exits.json',dict(passed=all(code==0 for code in exits.values()),returncodes=exits))
            assert all(code==0 for code in exits.values()),exits
            write(ROOT/'state/candidate_natural_reference_complete.json',dict(time=time.time(),passed=True,version=args.version,folder=str(folder)))
        finally:
            for child in children:
                if child.poll() is None:
                    assert os.getsid(child.pid)==child.pid
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            for handle in handles:handle.close()
            preserve()
            # Four scoped supervisors allow later pair leases without stopping unrelated pairs.
            for pair in ((0,1),(2,3),(4,5),(6,7)):owner.restore(list(pair))
            write(ROOT/'state/reference_resource_release.json',dict(time=time.time(),GPUs=list(range(8)),restored=True))


if __name__=='__main__':main()
