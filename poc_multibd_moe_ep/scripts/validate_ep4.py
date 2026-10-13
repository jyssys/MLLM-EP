"""Validate actual full-model proof traces, not mapped single-GPU routes."""
import json
import argparse
from poc_multibd_moe_ep.contract import *
from poc_multibd_moe_ep.scripts.analyze_conversion import ep_metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True)
    args=parser.parse_args()
    probes=[]
    for r in [1,2]:
        d=ROOT/f'artifacts/R{r}_ep4_held_out_paired_{args.session}'
        assert json.loads((d/'status.json').read_text())['success']
        result=json.loads((d/'request0_trace_repeat0.json').read_text())
        ownership=[json.loads((d/f'ownership.rank{k}.json').read_text()) for k in range(4)]
        for k,data in enumerate(ownership):
            assert len(data['sparse_layers'])==19
            assert str(data['actual_cuda_uuid']).removeprefix('GPU-')==UUIDS[k]
            for layer in data['sparse_layers']:
                assert layer['owner_ids']==list(range(k*64,(k+1)*64))
                assert layer['w13']==[64,1024,2048] and layer['w2']==[64,2048,512]
                assert layer['tensor_bytes']==402653184
                assert layer['backend']=='DeepEPDispatcher'
        layers=ep_metrics(d/'request0_trace_repeat0')
        t=result['trajectories'][0]
        def active_count(trajectory):
            return max(sum(b['is_active'] and not b['is_dummy'] for b in step['block_trace'])
                       for step in trajectory['trajectory'])
        active=active_count(t)
        # A short first problem need not admit a second real block. Validate
        # MultiBD transitions over the frozen cohort, not by counting dummies.
        cohort_active=max(active_count(json.loads(p.read_text())['trajectories'][0])
                          for p in d.glob('request*_trace_repeat0.json'))
        assert cohort_active>=r
        probes.append(dict(R=r,forward_layers_validated=len(layers),sparse_layers=19,
            owner_experts_per_rank=64,physical_ranks=GPUS,
            first_probe_maximum_real_active_blocks=active,
            cohort_maximum_real_active_blocks=cohort_active,
            output_tokens=len(t['token_ids']),termination=t['completion_reason'],
            path_and_destination_dedup_row_checks='PASS',
            timing_lane='heavy trace, NOT HEADLINE',
            multi_bd_state_machine='official untouched scheduler/sampler/cache',
            no_DMax=True))
    validity=json.loads((ROOT/'results'/args.session/'validity.json').read_text())
    write(ROOT/'results'/args.session/'physical_ep4_validation.json',dict(status='PASS',probes=probes,
        measurement_session=args.session,mask=MASK,
        supported_upstream_ep=False,bounded_bridge=['Config EP guard removal','owner-local received-ID correction'],
        stage0_real_generation=True, stage1_sharded_real_dispatch_owner_execution_combine=True,
        instrumentation_parity='PASS' if validity['ep4_held_out_pass_all'] else 'FAIL',
        full_model_performance_claim=False))
    print(probes)


if __name__=='__main__':main()
