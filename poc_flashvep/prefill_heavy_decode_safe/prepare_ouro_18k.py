"""Search between the valid 16K Native arm and the 24K TPOT failure."""
import copy,json
from .common import *


def main():
    initialize();length=18432
    for split in ('calibration','discovery'):
        source=ROOT/'data'/f'extended_natural_{split}_ouro_L24576.json'
        rows=copy.deepcopy(json.loads(source.read_text()))
        for r in rows:r['input_ids']['ouro']=r['input_ids']['ouro'][:length];r['length']=length
        path=ROOT/'data'/f'extended_natural_{split}_ouro_L18432.json'
        if path.exists():assert json.loads(path.read_text())==rows
        else:write(path,rows)
    cfg=json.loads((ROOT/'configs'/'extended_ouro_L24576_native_reference_state8.json').read_text())
    data=json.loads((ROOT/'data'/'extended_natural_calibration_ouro_L18432.json').read_text())
    state='extended_ouro_L18432_native_reference_state8'
    cfg.update(name=state,length=length,state_capacity_tokens=length,chunk=length,token_budget=2*length,
        D_token_budget=length,P_prefill_graph_rows=2*length,D_prefill_graph_rows=length,
        dataset='extended_natural_calibration_ouro_L18432.json',prompt_ids=[r['prompt_id'] for r in data[:8]],
        required_qualification='extended_ouro_L24576_native_reference_state8',
        phase='NATURAL_18K_NATIVE_FULL_STATE_REFERENCE',
        selection_reason='The 24576/16 Native pilot has TPOT p95 55.14415284618735 ms, above the unchanged 53.70702898374411 ms request limit. Test an intermediate length. Do not relax the SLO or reduce Output below 16.')
    save(cfg)
    cap=json.loads((ROOT/'configs'/'extended_capacity_ouro_L24576_O16_B12_start1.json').read_text())
    name='extended_capacity_ouro_L18432_O16_B12_start1'
    cap.update(name=name,length=length,state_capacity_tokens=length,chunk=length,
        token_budget=2*length,P_prefill_graph_rows=2*length,D_prefill_graph_rows=length,
        dataset='extended_natural_calibration_ouro_L18432.json',prompt_ids=[r['prompt_id'] for r in data[:2]],
        required_qualification=state,phase='NATURAL_18K_MATCHED_NATIVE_FINITE_CAPACITY')
    save(cap)
    write(ROOT/'state/extended_18k_plan.json',dict(jobs=[state,name],state8=state,capacity=name,
        scope='Intermediate legal context. Independent calibration and confirmation IDs. No new kernel, precision, SLO, or model change.'))
    print(state,name)


if __name__=='__main__':main()
