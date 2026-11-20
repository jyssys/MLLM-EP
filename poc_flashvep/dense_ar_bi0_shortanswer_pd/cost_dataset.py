"""Select cost calibration inputs by prespecified length ranks only."""
import json
from .common import ROOT,write,digest


def main():
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    tasks=['hotpotqa','musique','narrativeqa','qmsum','qasper'];parts={}
    for task in tasks:
        parts[task]=sorted([r for r in data if r['task']==task],key=lambda r:(r['input_tokens'],r['prompt_id']))
        assert len(parts[task])==40
    train=[parts[task][index] for index in (0,13,26,39) for task in tasks]
    heldout=[parts[task][index] for index in (6,20,33) for task in tasks]
    assert not ({r['prompt_id'] for r in train}&{r['prompt_id'] for r in heldout})
    for name,value in (('cost_calibration20.json',train),('cost_validation15.json',heldout)):
        path=ROOT/'data'/name
        if path.exists():assert json.loads(path.read_text())==value
        else:write(path,value)
    manifest=ROOT/'configs/frozen_cost_input_split_v1.json'
    value=dict(parent_dataset_SHA256=digest(ROOT/'data/longbench_shortanswer200_v2.json'),
        train_ranks=[0,13,26,39],validation_ranks=[6,20,33],sort=['input_tokens','prompt_id'],
        train_ids=[r['prompt_id'] for r in train],validation_ids=[r['prompt_id'] for r in heldout],
        selection='No performance, output length or answer score was used.',
        scope='Real requests from the frozen200. Disjoint cost fit/validation prompts, not a new Main workload.')
    if manifest.exists():assert json.loads(manifest.read_text())==value
    else:write(manifest,value)


if __name__=='__main__':main()
