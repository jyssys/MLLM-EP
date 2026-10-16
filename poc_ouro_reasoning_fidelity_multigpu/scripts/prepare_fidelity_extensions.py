"""Declare cross-backend tolerances and longer completion probes before comparison."""
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    questions=read(ROOT/'results/frozen_questions.json')
    rows=[r for r in questions if r['dataset']=='math100'][:2]
    values=[dict(id=r['id'],dataset=r['dataset'],prompt_ids=r['arms']['P1']['ids']) for r in rows]
    path=ROOT/'results/fidelity_completion_inputs.json'
    assert not path.exists()
    write(path,values)
    write(ROOT/'fidelity_extension_lock.json',dict(
        reason='Two metadata-ordered MATH cases supplement the long AIME capped probes. No result-based sample selection.',
        inputs_sha256=sha(path),cap=8192,temperature=0.,EOS_enabled=True,
        primary_lock_sha256=sha(ROOT/'protocol_lock.json'),
        tolerance_rule=dict(logits_relative_L2=dict(floor=.04,ceiling=.10),
            hidden_relative_L2=dict(floor=.06,ceiling=.10),
            KV_relative_L2=dict(floor=.025,ceiling=.05),
            next_token_JSD=dict(floor=.005,ceiling=.05),
            calibration_multiplier=2.,
            rule='For each metric, freeze max(floor, 2*calibration maximum). A result above the ceiling blocks confirmation.',
            scope='Diagnostic cross-backend BF16 calibration. Not a quality or bitwise parity assertion.'),
        consequential_drift_rule='Investigate large-margin top1 changes, mask/cache inconsistencies, or failed confirmation limits. Small-margin natural branch changes are not alone a model bug.'))

if __name__=='__main__':
    main()
