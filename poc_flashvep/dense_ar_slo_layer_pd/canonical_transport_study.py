"""Qualify fresh-block ordering and test the observed descriptor explosion."""
import fcntl
from .common import *
from .configuration import base
from .method_discovery_v2 import method_settings
from .coalesced_costs import build,parse
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent
from .stages import state_gate
from .analyze import records,percentile,metrics


def qualify(count,whole=False):
    options=method_settings(group=8 if whole else 4,wait=True)
    options.update(async_decode=True,canonical_new_KV_block_order=True)
    name=f'qual_canonical_blocks_N{count}_whole{int(whole)}_full_state_v1'
    cfg=base(name,'FORCED_QUALIFICATION',count=count,cap=16,request_caps=[2]*count,
        arrival_offsets=[0.]*count,arrival_rate=None,
        forced_layers=[12] if whole else [0,2,4,8,12,0,4,8],
        verify_layer_execution=True,hidden_verify_hash=True,verify_native_prefix_KV=True,
        full_prompt_state_witness=True,full_prompt_state_witness_roles=['prefill','decode'],
        correctness_only=True,clean_timing=False,negotiation_timeout_s=120,**options)
    assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
    work=state_gate(name,count);assert work['passed'];folder=ROOT/'raw'/name
    expected=independent();source=json.loads((folder/'full_prompt_state_prefill.json').read_text())['requests']
    dest=json.loads((folder/'full_prompt_state_decode.json').read_text())['requests']
    imports={r['external_request_id']:r for p in folder.glob('extension_decode_*.jsonl')
        for r in records(p) if r['event']=='NATIVE_IMPORTED_PREFIX_IDENTITY'}
    checks={}
    for rid,cut in work['cuts'].items():
        actual=source[rid] if cut==36 else dest[rid]
        checks[rid]=dict(KV_equal=actual['KV']==expected[rid]['KV'],
            Hidden_equal=actual['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'],
            Native_import_equal=imports.get(rid,{}).get('passed',False) if cut==36 else True)
    passed=len(checks)==count and all(all(v.values()) for v in checks.values())
    write(ROOT/'analysis'/(name+'_state_gate.json'),dict(name=name,passed=passed,checks=checks,
        work=work,evidence='CORRECTNESS_ONLY'))
    assert passed
    return name


def calibration(batch,enabled,tag):
    cfg=build(batch);cfg.update(name=f'canonical_cost_B{batch}_{tag}_N{batch*15}_v1',
        canonical_new_KV_block_order=enabled,async_decode=True,
        phase='CANONICAL_BLOCK_ORDER_EVENT_HOST_CALIBRATION_NOT_CLEAN_SERVING')
    name=save(cfg);assert execute(name,restore=False);assert against_previous(name)['passed']
    rows=parse(name);tx=[]
    for p in (ROOT/'raw'/name).glob('native_decode_*.json'):
        for x in json.loads(p.read_text())['NIXL_stats_intervals']:
            d=x['data']
            tx.extend(dict(descriptors=n,duration_ms=1000*t) for n,t in zip(d['num_descriptors'],d['transfer_duration']))
    value=dict(name=name,canonical=enabled,batch=batch,rows=rows,
        median_descriptors=percentile([r['descriptors'] for r in tx],.5),
        mean_descriptors=sum(r['descriptors'] for r in tx)/len(tx),
        operation_duration_p95_ms=percentile([r['duration_ms'] for r in tx],.95),
        handoff_p95_ms=percentile([r['handoff_to_D_stream_start_ms'] for r in rows],.95),
        evidence='MEASURED',scope='Isolated event/Host calibration. Operation telemetry is not pure wire time or online speedup.')
    write(ROOT/'analysis'/(name+'_calibration.json'),value)
    print('CANONICAL_CALIBRATION',name,value['mean_descriptors'],value['handoff_p95_ms'],flush=True)
    return value


def model(values):
    rows=[r for value in values for r in value['rows']]
    original=json.loads((ROOT/'configs/coalesced_cost_model_v2.json').read_text())
    for key,field in {'P_prefix_ms':'P_prefix_host_ms','handoff_ms':'handoff_to_D_stream_start_ms',
            'D_suffix_ms':'D_suffix_event_ms','D_commit_ms':'D_commit_host_ms'}.items():
        original[key]={str(batch):{str(k):percentile([r[field] for r in rows if r['batch']==batch and r['layers']==k],.95)
            for k in (0,2,4,8,12)} for batch in sorted({r['batch'] for r in rows})}
    original.update(model_version='CANONICAL_NEW_BLOCK_ORDER_ASYNC_V4',numerical_tables_changed=True,
        primitive_calibration_names=[v['name'] for v in values],
        primitive_sample_counts={str(b):3 for b in sorted({r['batch'] for r in rows})},
        primitive_files={v['name']:digest(ROOT/'raw'/v['name']/'source.zip') for v in values},
        borrowed_components=['decode_step_by_batch_p95_ms','negotiation_idle_ms','negotiation_busy_ms','uncertainty'],
        scope='Three isolated groups per cut and actual batch. Fresh block ordering and standard async Decode. Stream events include launch gaps and waits. Borrowed Decode bounds remain explicitly conservative lookups, not a recalibrated request-max guarantee or WCET. Validate residual coverage on later discovery and unseen starts.')
    write(ROOT/'configs/canonical_cost_model_v4.json',original)


def main():
    try:
        gates=[qualify(32),qualify(8,True)]
        a=calibration(8,False,'A1');b=calibration(8,True,'B');c=calibration(8,False,'A2')
        ratio=b['handoff_p95_ms']/((a['handoff_p95_ms']+c['handoff_p95_ms'])/2)
        descriptor_ratio=b['mean_descriptors']/((a['mean_descriptors']+c['mean_descriptors'])/2)
        worth=ratio<.9 and descriptor_ratio<.5
        write(ROOT/'analysis/canonical_transport_ABA.json',dict(A1=a,B=b,A2=c,
            handoff_p95_ratio=ratio,descriptor_mean_ratio=descriptor_ratio,continue_calibration=worth,
            gates=gates,evidence='MEASURED',scope='One A/B/A mechanism study, not a confidence interval or online speedup. Same request data, async execution, registration and ownership. Only the fresh-block return order changes.'))
        if not worth:return
        values=[b]
        for batch in (1,2,3,4,16):values.append(calibration(batch,True,'CAL'))
        model(values)
        write(ROOT/'analysis/canonical_transport_full_state_gate.json',dict(passed=True,gates=gates,
            model='canonical_cost_model_v4.json',evidence='CORRECTNESS_ONLY'))
        from .expired_deadline_recovery import options
        for label,arm,rate in (('Native','Native',4.8),('Static','STATIC',5.),('Dynamic','DIRECT',5.)):
            if arm=='Native':
                changes=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,async_decode=True,
                    nixl_manual_coalesce=True,nixl_descriptor_sort=False)
            else:changes=options(True)
            changes.update(canonical_new_KV_block_order=True,cost_model='canonical_cost_model_v4.json',
                static_layers=12,migration_fraction=1.,phase='CANONICAL_PD_SHORT_SERVING_DISCOVERY')
            name=f'canonical_PD_{label}_N128_O16_q{rate}_v1'
            cfg=base(name,arm,count=128,cap=16,rate=rate,**changes)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            print('CANONICAL_PD_PILOT',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)
        write(ROOT/'state/canonical_transport_study_complete.json',dict(time=time.time(),continued=True))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
