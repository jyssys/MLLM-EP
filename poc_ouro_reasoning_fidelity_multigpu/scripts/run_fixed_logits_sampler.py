"""Isolate seeded sampling on frozen logits. No model or performance claim."""
import os
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
    import numpy as np
    import torch
    import vllm.v1.sample.ops.topk_topp_sampler as ops
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    source=ROOT/'results/native_fidelity_TP1_v1/teacher_60.step0.rank0.npz'
    frozen=np.load(source)['logits']
    assert frozen.shape[0]==1
    logits=torch.as_tensor(frozen,device='cuda',dtype=torch.float32)
    sampler=ops.TopKTopPSampler(use_fp64_gumbel=False)
    noise=[]
    original=ops.empty_exponential_noise_like
    def capture(probs,use_fp64_gumbel):
        value=original(probs,use_fp64_gumbel)
        noise[:]=[value]
        return value
    ops.empty_exponential_noise_like=capture
    records=[]
    try:
        for batch in [1,4]:
            for repeat in range(2):
                generators={i:torch.Generator(device='cuda').manual_seed(SEEDS[0]) for i in range(batch)}
                items=[]
                for step in range(128):
                    before=[int(g.get_offset()) for g in generators.values()]
                    sampled,_=sampler(logits.repeat(batch,1),generators,None,
                                      torch.full((batch,),.7,device='cuda'))
                    value=noise[0].detach().cpu().numpy()
                    items.append(dict(step=step,token_ids=sampled.tolist(),offset_before=before,
                        offset_after=[int(g.get_offset()) for g in generators.values()],
                        noise_row_sha256=[hashlib.sha256(row.tobytes()).hexdigest() for row in value]))
                records.append(dict(batch=batch,repeat=repeat,steps=items))
    finally:
        ops.empty_exponential_noise_like=original
    same_repeat=all(records[i]['steps']==records[i+1]['steps'] for i in [0,2])
    same_first_row=all(a['noise_row_sha256'][0]==b['noise_row_sha256'][0]
        and a['token_ids'][0]==b['token_ids'][0]
        for a,b in zip(records[0]['steps'],records[2]['steps']))
    write(output/'fixed_logits_sampler.json',dict(success=True,records=records,
        source_sha256=sha(source),torch=torch.__version__,sampler_source_sha256=sha(ops.__file__),
        GPU_uuid=str(torch.cuda.get_device_properties(0).uuid),physical_gpu=4,
        same_process_repeats_equal=same_repeat,B1_B4_first_row_equal=same_first_row,
        model_forward='NOT_RUN',actual_request_generator_offsets='NOT_CAPTURED',
        scope='Fixed-logit diagnostic. This result does not establish model-output restart determinism.'))


if __name__=='__main__':
    main()
