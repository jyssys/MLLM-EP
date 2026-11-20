"""Change descriptor grouping without changing KV bytes or model math."""
import json
import time
from .common import ROOT,save,write
from .qualification import configuration,records
from .run import execute
from .campaign import serving


def main():
    design=ROOT/'configs/frozen_exact_coalescing_screen_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),KV_block_tokens=16,NIXL_threads=4,official_UCX_init=True,
            B=8192,S=1024,state_N=8,performance_N=32,QPS=2.35,arrival_seed=2026112033,
            intervention='Existing exact paired-address descriptor coalescing only. No packing, extra KV copy, model or scheduler change.',
            gate='Full valid KV and P hidden equality, strict same-geometry Native output, lifetime retirement.',
            scope='Independent transport diagnostic. Current KV128 Layer/Token/Native primary arms remain unchanged. '
            'Do not silently adopt a new transport only for one method. N32 cannot qualify full200 performance.'))
    name='qual_Native_ExactCoalesce_KV16_N8_BI0_v1'
    cfg=configuration(name,'Native','qualification8.json',restore_official_UCX_init=True,
        nixl_manual_coalesce=True,numerical_probe=True,numerical_probe_sync=True,
        phase='EXACT_DESCRIPTOR_COALESCING_CORRECTNESS')
    assert execute(save(cfg)),name
    folder=ROOT/'raw'/name
    baseline=ROOT/'raw/qual_Native_OfficialInit_ShortAnswer_N8_BI0_v2'
    expected=json.loads((baseline/'prompt_state_prefill.json').read_text())['requests']
    p=json.loads((folder/'prompt_state_prefill.json').read_text())['requests']
    d=json.loads((folder/'prompt_state_decode.json').read_text())['requests']
    expected_output={r['sample_id']:r['output_ids'] for r in records(baseline/'requests.jsonl')}
    actual_output={r['sample_id']:r['output_ids'] for r in records(folder/'requests.jsonl')}
    rows=[dict(request_id=f'q{i}',full_KV_exact=expected[f'q{i}']['KV']==p[f'q{i}']['KV']==d[f'q{i}']['KV'],
        P_hidden_exact=expected[f'q{i}']['final_hidden_SHA256']==p[f'q{i}']['final_hidden_SHA256'],
        greedy_exact=expected_output[i]==actual_output[i]) for i in range(8)]
    lifetime={file.name:json.loads(file.read_text())['passed'] for file in folder.glob('*lifetime*.json')}
    passed=all(r['full_KV_exact'] and r['P_hidden_exact'] and r['greedy_exact'] for r in rows) and bool(lifetime) and all(lifetime.values())
    write(ROOT/'analysis/BI0_exact_coalescing_N8_state_v1.json',dict(passed=passed,rows=rows,lifetime=lifetime,evidence='CORRECTNESS_ONLY'))
    assert passed
    results=[]
    for enabled in (False,True):
        name=f'CoalescingAB_{int(enabled)}_Native_KV16_N32_q2.35_BI0_v1'
        value=serving(name,'Native','qualification32.json',32,2.35,2026112033,
            nixl_manual_coalesce=enabled,restore_official_UCX_init=True,phase='EXACT_DESCRIPTOR_COALESCING_INTERVENTION')
        setup=json.loads((ROOT/'raw'/name/'NIXL_host_setup_decode.json').read_text())['rows']
        window=json.loads((ROOT/'raw'/name/'window.json').read_text())
        selected=[r for r in setup if r['kind'] in ('make_transfer_host','submit_transfer_host','NIXL_native_telemetry') and
            window['start']<=r.get('start',r.get('time',0))<=window['end']]
        results.append(dict(name=name,enabled=enabled,metrics=value,transport_rows=selected))
        write(ROOT/'analysis/BI0_exact_coalescing_N32_screen_v1.json',dict(results=results,evidence='DIAGNOSTIC',
            scope='One independent start per arm. Exact-map byte coalescing, no model or scheduling change. '
            'Warmup excluded. Natural BI0 outputs/batches can differ. No primary full200 adoption or causal kernel/DMA attribution.'))
    write(ROOT/'state/exact_coalescing_screen_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/exact_coalescing_screen_hold.json',dict(time=time.time(),error=repr(error)));raise
