"""Add measured receiver shapes without changing frozen training costs."""
from .common import *
from .analyze import records,percentile


def main():
    target=ROOT/'configs/revised_cost_model_v2.json'
    if target.exists():print('V2_COST_ALREADY_FROZEN');return
    model=json.loads((ROOT/'configs/revised_cost_model_v1.json').read_text())
    added=[]
    for batch in (8,16):
        model['D_suffix_ms'][str(batch)]={'0':0.}
        model['D_commit_ms'][str(batch)]={'0':0.}
        for layers in (2,4,8,12):
            name=f'cost_Source4_Destination{batch}_Last{layers}_N{2*batch}_v1'
            folder=ROOT/'raw'/name
            assert json.loads((folder/'dispatcher_result.json').read_text())['complete'],name
            frames=json.loads((folder/'CUDA_event_enclosures_decode.json').read_text())['rows']
            selected=[r for r in frames if r['prefilling'] and r.get('cut')==36-layers and len(r['requests'])==batch]
            assert len(selected)>=2,name
            requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
            commits=[1000*(max(requests[rid]['first_token'] for rid in r['requests'])-r['end_monotonic']) for r in selected]
            model['D_suffix_ms'][str(batch)][str(layers)]=max(r['elapsed_ms'] for r in selected)
            model['D_commit_ms'][str(batch)][str(layers)]=max(0.,*commits)
            added.append(dict(name=name,source_batch=4,destination_batch=batch,layers=layers,samples=len(selected),
                suffix_max_event_ms=model['D_suffix_ms'][str(batch)][str(layers)],
                commit_max_observed_ms=model['D_commit_ms'][str(batch)][str(layers)],
                source_SHA256=digest(folder/'CUDA_event_enclosures_decode.json')))
    model.update(revised_revision=2,v1_SHA256=digest(ROOT/'configs/revised_cost_model_v1.json'),
        added_receiver_calibration=added,v2_freeze_epoch=time.time(),future_confirmation_information=False,
        scope='Measured receiver shapes and previous source costs. Small-sample maxima are empirical estimates, not WCET bounds. Serialized handoff is enforced by policy revision2.')
    write(target,model);print('V2_COST_FROZEN',len(added))


if __name__=='__main__':main()
