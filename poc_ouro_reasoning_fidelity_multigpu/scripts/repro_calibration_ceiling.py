"""Detect the exact calibration-state ceiling exceedance. This is not a quality verdict."""
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_ouro_reasoning_fidelity_multigpu.scripts.analyze_fidelity import compare

def main():
    row=read(ROOT/'results/fidelity_inputs.json')[0]
    assert row['id']=='60' and row['split']=='calibration'
    result=compare(row,1,1024)[0]
    ceiling=read(ROOT/'fidelity_extension_lock.json')['tolerance_rule']['hidden_relative_L2']['ceiling']
    write(ROOT/'results/calibration_ceiling_repro_v1.json',dict(result=result,planning_ceiling=ceiling,
        ceiling_pass=result['hidden']['relative_L2']<=ceiling,
        scope='Calibration only. No confirmation difference or benchmark accuracy enters this test.'))
    print('Hidden relative L2:',result['hidden']['relative_L2'],'Planning ceiling:',ceiling,
        'Next-token JSD nats:',result['logits']['JSD_nats'],flush=True)
    assert result['hidden']['relative_L2']<=ceiling,'Calibration hidden-state ceiling exceeded. Diagnose before any confirmation promotion.'

if __name__=='__main__':
    main()
