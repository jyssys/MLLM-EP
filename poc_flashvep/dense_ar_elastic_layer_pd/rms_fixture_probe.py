"""Replay the first differing normalization with identical model operands."""
import argparse
from .common import *
from .watch import GPUWatch
from .diagnosis import tensor_difference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());folder=ROOT/'raw'/cfg['name']
    watch=GPUWatch(folder,cfg['gpus'],cfg['name']);watch.start()
    try:
        import torch,vllm
        from safetensors import safe_open
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.batch_invariant import init_batch_invariance
        init_batch_invariance();torch.set_num_threads(1)
        fixtures=[]
        for arm,role in (('Native','decode'),('Continuous','replica1')):
            fixtures.append(torch.load(ROOT/f'raw/numeric_{arm}_eager_O16_N8_v3/numeric_probe_{role}.pt',weights_only=False))
        name='model.layers.23.post_attention_layernorm.weight'
        index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
        with safe_open(MODEL/index[name],framework='pt',device='cpu') as source:weight=source.get_tensor(name).cuda()
        module='model.layers.23.';native,candidate=fixtures
        x=native['submodules'][module+'self_attn.o_proj']['outputs'][0].cuda()
        residual=native['submodules'][module+'input_layernorm']['outputs'][1].cuda()
        assert torch.equal(x.cpu(),candidate['submodules'][module+'self_attn.o_proj']['outputs'][0])
        assert torch.equal(residual.cpu(),candidate['submodules'][module+'input_layernorm']['outputs'][1])
        expected=[f['submodules'][module+'post_attention_layernorm']['outputs'][0] for f in fixtures]
        rows=[];profile=None
        if cfg.get('primitive_profile'):
            for count in (1,513):
                a=x.repeat(count,1);r=residual.repeat(count,1);ops.fused_add_rms_norm(a,r,weight,1e-6)
            torch.cuda.synchronize()
            profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
            profile.__enter__()
        for count in (1,2,4,8,16,32,64,128,255,256,512,513,1024,8000,32000):
            a=x.repeat(count,1);r=residual.repeat(count,1)
            ops.fused_add_rms_norm(a,r,weight,1e-6);torch.cuda.synchronize()
            rows.append(dict(rows=count,Native=tensor_difference(a[0].cpu(),expected[0]),
                Continuous=tensor_difference(a[0].cpu(),expected[1])))
        write(folder/'rms_fixture.json',dict(rows=rows,identical_original_operands=True,
            batch_invariant_env=os.environ.get('VLLM_BATCH_INVARIANT'),
            extension_path=str(Path(vllm.__file__).parent/'_C_stable_libtorch.abi3.so'),
            evidence='CORRECTNESS_ONLY',scope='Exact saved layer23 operands. Other rows repeat the victim only for a primitive shape test. This is not a serving workload.'))
        if profile is not None:
            profile.__exit__(None,None,None);profile.export_chrome_trace(str(folder/'rms_primitive_diagnostic_trace.json'))
        print(json.dumps(rows,indent=2),flush=True)
        write(folder/'worker_exits.json',dict(passed=True));write(folder/'status.json',dict(status='COMPLETE'))
    finally:watch.stop()


if __name__=='__main__':main()
