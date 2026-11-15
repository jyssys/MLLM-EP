"""Check exact rows and isolated event enclosures under the shared GPU lease."""
import fcntl,subprocess
from .common import *


def main():
    initialize()
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX);owner=resources();owner.stop(list(range(8)))
        folder=ROOT/'raw/exact_phase_norm_micro_v2'
        assert not folder.exists(),'Keep the prior diagnostic'
        folder.mkdir();started=time.time()
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',VLLM_BATCH_INVARIANT='1',TORCH_CUDA_ARCH_LIST='9.0',MAX_JOBS='1')
            child=subprocess.run([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_slo_layer_pd.norm_micro','--worker'],cwd=REPO,env=env,
                capture_output=True,text=True,timeout=180)
            write(folder/'command_and_result.json',dict(start=started,end=time.time(),returncode=child.returncode,
                stdout=child.stdout,stderr=child.stderr,source_SHA256=digest(__file__),
                evidence='CORRECTNESS_ONLY',scope='Isolated GPU0 kernel qualification. All approved burners are quiet. No Serving rate or Decode-SLO result.'))
            assert child.returncode==0,child.stderr
        finally:
            if not owner.processes():owner.restore(list(range(8)))


def worker():
    import torch
    from vllm import _custom_ops as ops
    from .exact_phase_norm import load
    from .analyze import percentile
    from .phase_stable_norm import stable_fused
    from types import SimpleNamespace
    torch.set_num_threads(1);torch.manual_seed(1401);module=load();checks=[];timings=[]
    def reference(a,r,w):
        x=torch.zeros((256,4096),device='cuda',dtype=torch.bfloat16);v=torch.zeros_like(x)
        x[:len(a)].copy_(a);v[:len(a)].copy_(r);ops.fused_add_rms_norm(x,v,w,1e-6)
        return x[:len(a)],v[:len(a)]
    for rows in (1,2,8,16,64,128,255):
        for seed in range(10):
            generator=torch.Generator(device='cuda').manual_seed(1000+seed)
            a=torch.randn((rows,4096),device='cuda',dtype=torch.bfloat16,generator=generator)
            r=torch.randn_like(a);w=torch.randn((4096,),device='cuda',dtype=torch.bfloat16,generator=generator)
            for scale in (.01,1.,100.):
                x=a*scale;v=r*scale;expected_x,expected_r=reference(x,v,w)
                module.phase_norm(x,v,w,1e-6);torch.cuda.synchronize()
                passed=torch.equal(x,expected_x) and torch.equal(v,expected_r)
                checks.append(dict(rows=rows,seed=seed,scale=scale,passed=passed,
                    norm_mismatches=int((x!=expected_x).sum()),residual_mismatches=int((v!=expected_r).sum())))
        x=torch.randn((rows,4096),device='cuda',dtype=torch.bfloat16);r=torch.randn_like(x)
        w=torch.ones((4096,),device='cuda',dtype=torch.bfloat16)
        norm=SimpleNamespace(weight=w,variance_epsilon=1e-6,pass_weight_add=True,variance_size_override=None)
        def original(module,a,v):ops.fused_add_rms_norm(a,v,module.weight,module.variance_epsilon);return a,v
        for kind in ('padded_reference','explicit_256_thread'):
            elapsed=[]
            for _ in range(30):
                a=x.clone();v=r.clone();torch.cuda.synchronize()
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record()
                stable_fused(original,norm,a,v,[(0,rows,'PREFILL')],
                    module.phase_norm if kind=='explicit_256_thread' else None)
                end.record();end.synchronize();elapsed.append(begin.elapsed_time(end))
            timings.append(dict(rows=rows,kind=kind,event_enclosure_p50_ms=percentile(elapsed,.5),
                event_enclosure_p95_ms=percentile(elapsed,.95),samples=len(elapsed)))
    passed=all(r['passed'] for r in checks)
    write(ROOT/'analysis/exact_phase_norm_GPU_gate.json',dict(checks=checks,passed=passed,timings=timings,
        GPU_UUID=str(torch.cuda.get_device_properties(0).uuid),torch=torch.__version__,
        kernel_source_SHA256=digest(SOURCE/'exact_phase_norm.py'),evidence='CORRECTNESS_ONLY',
        scope='Compare every valid BF16 norm and residual element against the original256-row padded operation. Isolated stream-event enclosures include allocation and launch gaps. They are not kernel-union times or Serving speedups.'))
    assert passed,'The explicit launch did not reproduce the original valid rows exactly'
    print('EXACT_PHASE_NORM_GPU_PASS',len(checks))


if __name__=='__main__':
    import sys
    if '--worker' in sys.argv:worker()
    else:main()
