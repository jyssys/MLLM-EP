"""Record runtime differences without inventing unmeasured kernel costs."""
import json
import numpy as np
from .common import ROOT,write,digest
from .historical_bridge import HISTORICAL,SEEDS


def current(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    rows=json.loads((folder/'NIXL_host_setup_decode.json').read_text())['rows']
    measured=[r for r in rows if isinstance(r.get('request_id'),str) and r['request_id'].split('-')[0].startswith('q')]
    summary={}
    for kind in ('make_transfer_host','submit_transfer_host','read_blocks_host'):
        selected=[r for r in measured if r['kind']==kind]
        summary[kind]=dict(calls=len(selected),mean_host_ms=float(np.mean([1000*(r['end']-r['start']) for r in selected])) if selected else None)
        if selected and kind=='make_transfer_host':
            summary[kind].update(input_descriptors_mean=float(np.mean([r['input_descriptors'] for r in selected])),
                submitted_descriptors_mean=float(np.mean([r['submitted_descriptors'] for r in selected])),
                descriptor_reduction_calls=sum(r['submitted_descriptors']<r['input_descriptors'] for r in selected),
                valid_candidate_byte_records=sum(r.get('bytes') is not None for r in selected))
    return dict(name=name,BI=cfg['batch_invariant'],KV_block=cfg['KV_block_tokens'],
        coalescing=cfg.get('nixl_manual_coalesce',False),summary=summary,
        SDK_agent_configurations=[r for r in rows if r['kind']=='NIXL_agent_configuration'],
        raw_SHA256=digest(folder/'NIXL_host_setup_decode.json'),evidence='MEASURED_HOST_BOUNDARIES',
        scope='Only q-request API enclosures. Read/make/submit overlap and cannot be added as exclusive time. Not pure DMA or GEMM duration.')


def main():
    historical=[]
    for seed in SEEDS:
        for axis in ('Layer','Token'):
            name=f'heldout_confirmation_{axis}_Last8_N72_O16_q6_s{seed}_B1_W32_v1';folder=HISTORICAL/'raw'/name
            cfg=json.loads((folder/'configuration.json').read_text())
            norms={role:json.loads((folder/f'phase_stable_norm_{role}.json').read_text()) for role in ('prefill','decode')}
            assert all(r['counters']['split_normalization_calls']==r['counters']['padded_prefill_calls']==0 for r in norms.values())
            runtimes={role:json.loads((folder/f'effective_runtime_{role}.json').read_text()) for role in ('prefill','decode')}
            coalescing=json.loads((folder/'manual_KV_coalescing.json').read_text())['rows']
            graphs=json.loads((folder/'graph_replay_decode.json').read_text())
            historical.append(dict(name=name,axis=axis,seed=seed,norm_correction=norms,
                engine_runtime={role:{k:v for k,v in value.items() if k!='KV_tensors'} for role,value in runtimes.items()},
                replay_shape_counts=graphs['counts'],
                coalescing_all_recorded_calls=dict(calls=len(coalescing),
                    input_descriptors_mean=float(np.mean([r['input_descriptors'] for r in coalescing])),
                    submitted_descriptors_mean=float(np.mean([r['submitted_descriptors'] for r in coalescing])),
                    scope='Includes warm-up because this old observer did not record a request ID. Never label this a measured-only mean.'),
                source_window=cfg['source_outstanding_limit'],prospective_reservation=cfg['common_prospective_KV_reservation'],
                config_SHA256=digest(folder/'configuration.json'),evidence='HISTORICAL_TRACE_ANALYSIS'))
    current_rows=[]
    for path in sorted((ROOT/'raw').glob('HistBridgeCurrent*/dispatcher_result.json')):
        if json.loads(path.read_text()).get('complete'):current_rows.append(current(path.parent.name))
    write(ROOT/'analysis/historical_current_runtime_transport_audit.json',dict(historical=historical,current=current_rows,
        falsification=dict(phase_norm_correction='Counter0 for all six historical Layer/Token starts. No correction operation was invoked.',
            prefill_graph_advantage='Historical engine mode is NONE/Compile0 for P and FULL_DECODE_ONLY/Compile0 for D. Replay shapes and the separate suffix runner source remain disclosed. No prefill-graph acceleration is assumed.'),
        scope='A source/runtime audit. Unknown historical SDK thread settings are not guessed. '
        'Different host boundaries do not become GPU-exclusive service costs. Coalescing and block interventions have separate outcome/statistical reports.'))
    print('RUNTIME_BRIDGE_AUDIT',len(historical),len(current_rows))


if __name__=='__main__':main()
