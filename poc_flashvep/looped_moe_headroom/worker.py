"""One model, one physical GPU, one immutable run directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .common import ROOT, write, append, digest

def timed(fn, repeats=20, warmups=5):
    import torch
    for _ in range(warmups):
        fn()
    rows = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        rows.append((time.perf_counter()-start)*1000)
        del out
    return rows

def p0(a, target):
    import torch
    from .adapters import quality
    from .probes import Controller
    metadata = a.metadata()
    text = 'The capital of France is Paris. A scientist measures the temperature and records the result. '
    ids = a.encode(text*12, 128)
    ref = a.forward(ids)
    assert torch.isfinite(ref).all()
    with Controller(a, 'probe') as c:
        other = a.forward(ids)
        frames = c.frames
        calls = dict(c.calls)
    result = quality(other, ref, ids)
    result['max_logit_error'] = float((ref-other).abs().max())
    assert torch.equal(ref, other), result
    for key in a.recurrent:
        assert calls[key] == a.loops, (key, calls[key], a.loops)
    with Controller(a, 'random_drop', q=0.) as c:
        q0 = a.forward(ids)
    assert torch.equal(ref, q0), 'q=0 must preserve exact native output'
    plain_gen = a.generate(ids, 8, stop_eos=False)
    with Controller(a, 'probe'):
        probe_gen = a.generate(ids, 8, stop_eos=False)
    assert plain_gen['ids'] == probe_gen['ids']
    metadata.update(calls_per_site=calls, observed_expert_outputs_dtype=sorted(set(r['dtype'] for r in frames)))
    write(target/'metadata.json', metadata)
    write(target/'baseline_validation.json', dict(status='PASS', input_ids=ids[0].tolist(),
        no_op=result, q0_max_error=float((ref-q0).abs().max()),
        generation=plain_gen, probe_generation=probe_gen, exact_tokens=True,
        total_site_executions=sum(calls.values()), forward_calls=1,
        shifted_ce=True, padding_mask_used=False))
    if a.name.startswith('ifm_'):
        # Official functional fallback comparison on the same pretrained model.
        attentions = [m for m in a.model.modules() if type(m).__name__ == 'CausalSoftmaxAttention']
        previous = [m.backend for m in attentions]
        small = ids[:, :32]
        flash = a.forward(small)
        try:
            for m in attentions:
                m.backend = None
            math_logits = a.forward(small)
        finally:
            for m, backend in zip(attentions, previous):
                m.backend = backend
        comparison = quality(flash, math_logits, small)
        comparison['relative_l2'] = float((flash.float()-math_logits.float()).norm()/math_logits.float().norm())
        # This is a predeclared backend check, not a relaxed no-op criterion.
        comparison['thresholds'] = dict(relative_l2=.05, top1_agreement=.90)
        comparison['pass'] = comparison['relative_l2'] <= .05 and comparison['top1_agreement'] >= .90
        write(target/'official_attention_backend_check.json', comparison)
        if not comparison['pass']:
            from .backend_diagnostic import run
            diagnostic=run(a,ids)
            write(target/'fp32_attention_diagnostic.json',diagnostic)
            assert diagnostic['pass_independent_checks'],diagnostic
    return dict(status='PASS', model=a.name, parameter_count=metadata['parameter_count'],
        recurrent_sites=len(a.recurrent), loops=a.loops)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['ifm_ouro','ifm_huginn','foil1'])
    p.add_argument('--phase', required=True, choices=['p0','p1','p2','timing','generation','profile','contention','micro'])
    p.add_argument('--run-id', required=True)
    p.add_argument('--job')
    args = p.parse_args()
    target = ROOT/'raw'/args.run_id
    target.mkdir(parents=True, exist_ok=True)
    assert not (target/'complete.json').exists(), 'A completed run cannot be overwritten'
    assert not (target/'failure.json').exists(), 'Use a new run ID after a failure'
    assert not (target/'request.json').exists(), 'Review an interrupted run. Use a new run ID, not an append resume.'
    started = time.time()
    write(target/'request.json', dict(vars(args), start=started, pid=os.getpid(),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), command=sys.argv,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p.relative_to(Path(__file__).parent)):digest(p) for p in Path(__file__).parent.rglob('*.py')}))
    try:
        from .adapters import Adapter
        adapter = Adapter(args.model)
        write(target/'runtime_metadata.json',adapter.metadata())
        parameter_state={name:(id(p),p.data_ptr(),p._version,p.dtype) for name,p in adapter.model.named_parameters()}
        if args.phase == 'p0':
            result = p0(adapter, target)
        elif args.phase in ('profile','contention','micro'):
            job=json.loads(Path(args.job).read_text())
            if args.phase=='micro':
                from .timing import micro_job
                result=micro_job(adapter,target,job)
            else:
                from .profiling import heavy,contention
                result={'profile':heavy,'contention':contention}[args.phase](adapter,target,job)
        elif args.phase == 'generation':
            from .generation import run
            result=run(adapter,target,json.loads(Path(args.job).read_text()))
        else:
            from .study import run
            result = run(adapter, args.phase, target, json.loads(Path(args.job).read_text()) if args.job else {})
        assert parameter_state=={name:(id(p),p.data_ptr(),p._version,p.dtype) for name,p in adapter.model.named_parameters()}, 'Parameter storage or values changed'
        result['parameter_storage_and_versions_unchanged']=True
        write(target/'complete.json', dict(result, elapsed_s=time.time()-started))
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write(target/'failure.json', dict(type=type(exc).__name__, error=str(exc), trace=traceback.format_exc(),
            elapsed_s=time.time()-started))
        raise

if __name__ == '__main__':
    main()
