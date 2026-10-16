"""Check the installed unpadded-mask helper without a model forward."""
import inspect
from types import SimpleNamespace
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    import torch
    import transformers
    from transformers.generation.utils import GenerationMixin
    assert transformers.__version__ == '4.57.1'
    helper = GenerationMixin._prepare_attention_mask_for_generation
    config = SimpleNamespace(_pad_token_tensor=torch.tensor(2), _eos_token_tensor=torch.tensor([2]))
    questions = read(ROOT/'results/frozen_questions.json')
    selected = [row for row in questions if row['dataset'] == 'aime24' and row['id'] in ['60', '61']]
    records = []
    for row in selected:
        for arm in ['P0', 'P1']:
            tokens = torch.tensor([row['arms'][arm]['ids']], dtype=torch.long)
            mask = helper(None, tokens, config, {})
            records.append(dict(id=row['id'], arm=arm, token_count=tokens.numel(),
                prompt_contains_EOS_delimiters=bool((tokens == 2).any()),
                all_input_positions_visible=bool((mask == 1).all()),
                mask_shape=list(mask.shape), mask_dtype=str(mask.dtype)))
    assert len(records) == 4 and all(row['all_input_positions_visible'] for row in records)
    target = ROOT/'results/HF_unpadded_attention_mask_audit.json'
    assert not target.exists()
    write(target, dict(transformers=transformers.__version__, torch=torch.__version__, records=records,
        source_path=inspect.getsourcefile(helper), source_sha256=sha(inspect.getsourcefile(helper)),
        helper_source=inspect.getsource(helper), model_forward='NOT_RUN', GPU_used=False,
        full_generation_attention_mask='NOT_CAPTURED', execution_parameters_changed=False,
        interpretation='The installed helper returns an all-ones mask for these unpadded B1 prompts when pad equals EOS.',
        limitation='This helper check does not capture every mask inside the running reference model.'))


if __name__ == '__main__':
    main()
