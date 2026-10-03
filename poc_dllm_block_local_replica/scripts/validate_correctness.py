"""Validate reuse, physical EP4 ownership, and destination-accounting invariants."""
import json
import numpy as np
from poc_dllm_block_local_replica.contract import *
from poc_dllm_ep_comm.scripts.analyze import request


def main():
    evidence=ROOT/'raw/physical_revalidation/ep4'
    own=[json.loads((evidence/f'ownership.rank{r}.json').read_text()) for r in range(4)]
    prior=[json.loads((SOURCE/f'raw/main_v2/ownership.rank{r}.json').read_text()) for r in range(4)]
    bytes_by_layer={}
    for rank,doc in enumerate(own):
        assert doc['mask']==MASK and doc['physical_gpu']==GPUS[rank]
        assert doc['actual_cuda_uuid'].removeprefix('GPU-')==UUIDS[rank]
        assert doc['ownership']==prior[rank]['ownership']
        for layer in doc['ownership']:
            assert layer['owned']==list(range(rank*64,(rank+1)*64))
            assert layer['dispatcher']=='DeepEPDispatcher'
            assert layer['kernel']=='UnquantizedFusedMoEMethod'
            size=sum(w['bytes'] for w in layer['weights'].values())//64
            assert size==EXPERT_BYTES
            bytes_by_layer[layer['layer']]=size
    probes=[]
    for directory in sorted(evidence.glob('*_b32')):
        r,_,_,_=request(directory)
        orig=json.loads((SOURCE/'raw/main_v2'/directory.name/'trace.rank0.json').read_text())
        current=json.loads((directory/'trace.rank0.json').read_text())
        assert current['output_ids']==orig['output_ids']
        assert [x['new_positions'] for x in current['acceptances']]==[x['new_positions'] for x in orig['acceptances']]
        for rank in range(4):
            with np.load(directory/f'trace.rank{rank}.npz') as z, np.load(SOURCE/'raw/main_v2'/directory.name/f'trace.rank{rank}.npz') as old:
                assert set(z.files)==set(old.files)
                for key in z.files:
                    if key.endswith('_weights'):assert np.allclose(z[key],old[key],rtol=1e-5,atol=1e-6)
                    else:assert np.array_equal(z[key],old[key]),key
        probes.append(dict(request=directory.name,output_acceptance_router_parity=True,
            physical_buffer_validation_checks=r['physical_buffer_validation_checks'],semantic_parity=r['semantic_parity']))
    new_json(ROOT/'results/baseline_validation.json',dict(status='PASS',probes=probes,
        tensor_bytes_per_layer=bytes_by_layer,bytes_per_expert=EXPERT_BYTES,
        topology=own[0]['topology'],actual_expert_sharding=True,DeepEP_sparse_dispatch_combine=True,
        current_vs_prior_source_ID_weight_buffer_parity=True))
    print('TRUE_EP4_REVALIDATED',len(probes),flush=True)


if __name__=='__main__':main()
