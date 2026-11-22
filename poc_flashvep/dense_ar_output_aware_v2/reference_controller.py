"""Generate expansion lengths under a durable, owned eight-GPU lease."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import ROOT,PYTHON,QUIET_LOCK,write,digest,initialize,preserve,resources,dispatch_allowed

def main():
    initialize();assert dispatch_allowed()
    census=json.loads((ROOT/'state/candidate_input_census_complete.json').read_text());assert census['passed']
    folder=ROOT/'raw/CandidateNaturalReference_v2extra';assert not folder.exists()
    folder.mkdir();private=ROOT/'references/worker_source_root_v1'
    capsule=ROOT/'references/source_capsules/v2_rooted_eager_prefill_native_decode_v1.zip'
    write(folder/'source_provenance.json',dict(SHA256=digest(capsule),candidate_SHA256=census['SHA256'],
        reference_scope='Original installed Full36 forward,Mode0 prompt,native full-decode CUDA graph,C1,natural EOS,cap2048. Length reference only.'))
    owner=resources();children=[];handles=[];stop=[False]
    def request_stop(*_):
        stop[0]=True;write(folder/'STOP_AFTER_REQUEST.json',dict(time=time.time(),reason='Owned checkpoint stop'))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,request_stop)
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX);preserve();owner.stop(list(range(8)))
        try:
            for gpu in range(8):
                env={k:v for k,v in os.environ.items() if not k.startswith(('DENSE_','DUETSERVE_'))}
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(private),VLLM_BATCH_INVARIANT='0',
                    VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'])
                command=[str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.reference_worker','--shard',str(gpu),'--version','v2extra']
                handle=(ROOT/'logs'/f'candidate_reference_gpu{gpu}_v2extra.log').open('w');handles.append(handle)
                child=subprocess.Popen(command,cwd=private,env=env,stdout=handle,stderr=subprocess.STDOUT,
                    start_new_session=True,preexec_fn=lambda gpu=gpu:os.sched_setaffinity(0,set(range(gpu*20,(gpu+1)*20))))
                children.append(child);write(folder/f'command_gpu{gpu}.json',dict(command=command,pid=child.pid,gpu=gpu,
                    source_SHA256=digest(capsule),time=time.time(),CPU_affinity=list(range(gpu*20,(gpu+1)*20))))
            began=time.monotonic()
            while any(p.poll() is None for p in children):
                # A completed shard releases its device immediately. Do not
                # leave that GPU idle while a longer natural shard drains.
                owner.restore([gpu for gpu,p in enumerate(children) if p.poll()==0])
                if any(p.poll() not in (None,0) for p in children) and not stop[0]:request_stop()
                if (not dispatch_allowed() or time.monotonic()-began>3*3600) and not stop[0]:request_stop()
                if stop[0] and time.monotonic()-began>3*3600+300:raise TimeoutError('Checkpoint drain timeout')
                write(ROOT/'state/reference_controller.json',dict(time=time.time(),pid=os.getpid(),status='RUNNING',
                    workers=[dict(gpu=i,pid=p.pid,returncode=p.poll()) for i,p in enumerate(children)]))
                time.sleep(5)
            exits={i:p.returncode for i,p in enumerate(children)};passed=all(x==0 for x in exits.values())
            write(folder/'worker_exits.json',dict(passed=passed,returncodes=exits))
            assert passed,exits
            write(ROOT/'state/candidate_natural_reference_complete.json',dict(time=time.time(),passed=True,version='v2extra'))
        finally:
            for child in children:
                if child.poll() is None:
                    assert os.getsid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(45)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            for handle in handles:handle.close()
            preserve()
            for pair in ((0,1),(2,3),(4,5),(6,7)):owner.restore(list(pair))
            write(ROOT/'state/reference_resource_release.json',dict(time=time.time(),restored=True))

if __name__=='__main__':main()
