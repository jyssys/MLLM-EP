import json
from poc_dllm_attn_moe_pipeline.contract import ROOT


def test_freeze_and_no_output_selection():
    frozen=json.loads((ROOT/'results/main/design_freeze.json').read_text())
    rows=frozen['rows']
    assert len(rows)==32
    assert len([r for r in rows if r['split']=='heldout'])==24
    assert len([r for r in rows if r['early_kill']])==12
    assert len(set(r['problem_id'] for r in rows))==32
    assert all(r['generation_budget']==64 for r in rows)
