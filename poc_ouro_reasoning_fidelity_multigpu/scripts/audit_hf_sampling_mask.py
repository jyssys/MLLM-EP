"""Test the actual HF mask fallback used by the unpadded B1 sampling probe."""
from types import SimpleNamespace
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    import inspect
    import torch
    import transformers
    from transformers.generation.utils import GenerationMixin
    assert transformers.__version__=='4.57.1'
    config=SimpleNamespace(_pad_token_tensor=torch.tensor(2),_eos_token_tensor=torch.tensor([2]))
    records=[]
    for row in read(ROOT/'results/fidelity_completion_inputs.json'):
        inputs=torch.tensor([row['prompt_ids']])
        mask=GenerationMixin._prepare_attention_mask_for_generation(None,inputs,config,{})
        assert bool(torch.all(mask==1))
        records.append(dict(id=row['id'],contains_chat_end_id=bool(torch.any(inputs==2)),
            mask_shape=list(mask.shape),all_ones=True,unpadded_B1=True))
    source=inspect.getfile(GenerationMixin)
    write(ROOT/'results/HF_sampling_mask_audit.json',dict(records=records,
        source=source,source_sha256=sha(source),version=transformers.__version__,
        conclusion='With pad=eos=2, the inspected fallback returns an all-ones input mask. It does not mask ChatML end markers.',
        scope='Direct CPU execution of the installed generation mask helper. Not benchmark accuracy.'))

if __name__=='__main__':
    main()
