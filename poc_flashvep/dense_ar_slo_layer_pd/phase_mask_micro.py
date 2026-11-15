"""Require byte equality before using graph-safe phase normalization."""
import fcntl,subprocess
from .common import *


def main():
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX);owner=resources();owner.stop(list(range(8)))
        folder=ROOT/'raw/phase_mask_norm_micro_v1'
        if (folder/'receipt.json').exists():return
        assert not folder.exists();folder.mkdir();started=time.time()
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',VLLM_BATCH_INVARIANT='1')
            r=subprocess.run([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_slo_layer_pd.phase_mask_micro','--worker'],
                cwd=REPO,env=env,capture_output=True,text=True,timeout=240)
            write(folder/'receipt.json',dict(start=started,end=time.time(),returncode=r.returncode,
                stdout=r.stdout,stderr=r.stderr,source_SHA256=digest(__file__),evidence='CORRECTNESS_ONLY'))
            assert r.returncode==0,r.stderr
        finally:
            if not owner.processes():owner.restore(list(range(8)))


def worker():
    import torch
    from vllm import _custom_ops as ops
    from .phase_mask_norm import load
    torch.set_num_threads(1);module=load();checks=[]
    for rows in (1,2,8,16,64,128,255,256,384,512,1024):
        for seed in range(4):
            torch.manual_seed(14000+seed)
            for kind in ('PREFILL','DECODE','MIXED'):
                labels=torch.zeros(rows,device='cuda',dtype=torch.int32)
                if kind=='DECODE':labels.fill_(1)
                elif kind=='MIXED':labels[torch.randperm(rows,device='cuda')[:min(rows//2,32)]]=1
                for scale in (.01,1.,100.):
                    a=torch.randn((rows,4096),device='cuda',dtype=torch.bfloat16)*scale
                    b=torch.randn_like(a)*scale;w=torch.randn(4096,device='cuda',dtype=torch.bfloat16)
                    expected_a=a.clone();expected_b=b.clone()
                    for phase in (0,1):
                        indices=torch.where(labels==phase)[0]
                        chunks=[indices[i:i+255] for i in range(0,len(indices),255)] if phase else [indices]
                        for idx in chunks:
                            if not len(idx):continue
                            count=len(idx);size=max(256,count) if phase==0 else count
                            x=torch.zeros((size,4096),device='cuda',dtype=torch.bfloat16);r=torch.zeros_like(x)
                            x[:count]=a[idx];r[:count]=b[idx]
                            ops.fused_add_rms_norm(x,r,w,1e-6)
                            expected_a[idx]=x[:count];expected_b[idx]=r[:count]
                    module.masked_norm(a,b,w,labels,1e-6);torch.cuda.synchronize()
                    passed=torch.equal(a,expected_a) and torch.equal(b,expected_b)
                    checks.append(dict(rows=rows,seed=seed,phase_pattern=kind,scale=scale,passed=passed))
    passed=all(c['passed'] for c in checks)
    write(ROOT/'analysis/phase_mask_norm_GPU_gate.json',dict(checks=checks,passed=passed,
        kernel_source_SHA256=digest(SOURCE/'phase_mask_norm.py'),original_body_SHA256=digest(SOURCE/'exact_phase_norm.py'),
        GPU_UUID=str(torch.cuda.get_device_properties(0).uuid),evidence='CORRECTNESS_ONLY',
        scope='Every tested BF16 normalized and residual element matches the original phase-specific operation. This is not a model, CUDA Graph, Serving, or speedup qualification.'))
    assert passed,'Masked normalization differs from the original phase reductions'
    print('PHASE_MASK_GPU_PASS',len(checks),flush=True)


if __name__=='__main__':
    import sys
    if '--worker' in sys.argv:worker()
    else:main()
