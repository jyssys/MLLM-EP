"""Require exact new-setting states before any performance sweep."""
import argparse
import fcntl
import json
import time
from .common import ROOT,PREVIOUS,QUIET_LOCK,bind_legacy,resources,save,write
from .prepare import qualification as configuration
from .run import execute


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def gate(name, reference):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    ref=json.loads((ROOT/'raw'/reference/'axis_state_prefill.json').read_text())['requests']
    bind_legacy()
    from .analysis import output_gate
    output=output_gate(name)
    method=cfg['evaluation_arm']
    checks={}
    if method=='Native':
        states=json.loads((folder/'axis_state_prefill.json').read_text())['requests']
        historical=json.loads((PREVIOUS/'raw/unseen_Native_reference_N128_O200_repeat1_v1/full_prompt_state_prefill.json').read_text())['requests']
        imported={r['external_request_id']:r for p in folder.glob('extension_decode_*.jsonl') for r in records(p)
                  if r.get('event')=='NATIVE_IMPORTED_PREFIX_IDENTITY'}
        imported_state=(json.loads((folder/'imported_native_prompt_KV.json').read_text())['requests']
                        if cfg.get('source_chunk_packing') else {})
        for rid,state in states.items():
            checks[rid]=dict(KV=state['KV']==historical[rid]['KV'],
                hidden=state['final_hidden_SHA256']==historical[rid]['final_hidden_SHA256'],
                imported_KV=(imported_state.get(rid,{}).get('KV')==state['KV']) if cfg.get('source_chunk_packing') else imported.get(rid,{}).get('passed') is True)
    elif method=='Token':
        src=json.loads((folder/'axis_state_prefill.json').read_text())['requests']
        dst=json.loads((folder/'axis_state_decode.json').read_text())['requests']
        from ..dense_ar_layer_token_pd.token_axis import suffix_for
        for rid,state in dst.items():
            suffix=suffix_for(cfg,rid)
            hashes=ref[rid]['partition_hashes'][str(suffix)]
            checks[rid]=dict(KV=state['KV']==ref[rid]['KV'],
                P_hidden=src[rid]['final_hidden_SHA256']==hashes['prefix_SHA256'],
                D_hidden=state['final_hidden_SHA256']==hashes['suffix_SHA256'],
                P_end=src[rid]['query_end']==8000-suffix,
                D_start=state['first_query']==8000-suffix,D_end=state['query_end']==8000)
    else:
        roles=('replica0','replica1') if method=='Agg' else ('decode',)
        for role in roles:
            states=json.loads((folder/f'full_prompt_state_{role}.json').read_text())['requests']
            for rid,state in states.items():
                checks[rid]=dict(KV=state['KV']==ref[rid]['KV'],
                    hidden=state['final_hidden_SHA256']==ref[rid]['final_hidden_SHA256'])
    visits=[]
    definitions=[]
    for p in folder.glob('extension_*.jsonl'):
        for row in records(p):
            if row.get('event')=='ACTUAL_ATTENTION_LAYER_EXECUTION' and row.get('measured'):
                visits.append(row)
            if row.get('event')=='CAPTURED_FULL_GRAPH_LAYERS':
                definitions.append(row)
    visits_pass=bool(visits) and all(r['passed'] for r in visits)
    if cfg['ar_cudagraph_mode']!='NONE':
        visits_pass=visits_pass and bool(definitions) and all(r['passed'] and r['actual']==list(range(36)) for r in definitions)
    independent=[]
    if method=='Agg' and cfg.get('independent_model_layer_witness'):
        for p in folder.glob('independent_layer_witness_*.json'):
            value=json.loads(p.read_text())
            independent.append(value)
        independent_calls=[r for v in independent for r in v['actual_calls']]
        independent_defs=[r for v in independent for r in v['captured_definitions']]
        visits_pass=(len(independent)==2 and bool(independent_calls) and bool(independent_defs)
            and all(r['passed'] and r['actual']==list(range(36)) for r in independent_calls+independent_defs))
    lifetime={p.name:json.loads(p.read_text()).get('passed') for p in folder.glob('KV_lifetime_*.json')}
    passed=(output['passed'] and len(checks)==cfg['count'] and all(all(v.values()) for v in checks.values())
        and len(lifetime)==2 and all(lifetime.values()) and visits_pass)
    result=dict(name=name,reference=reference,passed=passed,checks=checks,output=output,
        actual_layer_visits_pass=visits_pass,actual_measured_forwards=len(visits),
        full_graph_definitions=len(definitions),lifetime=lifetime,
        independent_decoder_layer_witnesses=independent,
        actual_worker_gate=json.loads((folder/'unified_worker_gate.json').read_text())['passed'],
        evidence='CORRECTNESS_ONLY',
        scope='Exact valid full prompt KV and final hidden against the new unified Native reference. Token segments are compared separately. Layer transport hashes and actual layer visits are checked by the worker. All generated outputs match the preserved original BF16 greedy reference. CPU state copies perturb latency.')
    write(ROOT/'analysis'/(name+'_state_gate.json'),result)
    assert passed,{k:v for k,v in result.items() if k!='checks'}
    print('STATE_GATE_PASS',name,flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=8)
    p.add_argument('--native-version',type=int,default=2)
    p.add_argument('--candidate-version',type=int,default=1)
    p.add_argument('--agg-version',type=int)
    args=p.parse_args()
    values=[]
    try:
        reference=configuration('Native',args.count,version=args.native_version)
        existing=ROOT/'configs'/(reference['name']+'.json')
        if existing.exists():
            reference=json.loads(existing.read_text())
        assert execute(save(reference)),reference['name']
        values.append(gate(reference['name'],reference['name']))
        for method in ('Layer','Token','Agg'):
            for layers in ((4,8,12) if method!='Agg' else (12,)):
                version=args.agg_version if method=='Agg' and args.agg_version is not None else args.candidate_version
                cfg=configuration(method,args.count,layers,version=version)
                assert execute(save(cfg)),cfg['name']
                values.append(gate(cfg['name'],reference['name']))
        family=('qualification_reserved' if reference.get('common_prospective_KV_reservation')
                else 'qualification_chunked' if reference.get('source_chunk_packing') else 'qualification')
        write(ROOT/'analysis'/f'{family}_N{args.count}.json',dict(time=time.time(),passed=True,
            names=[v['name'] for v in values],reference=reference['name'],evidence='CORRECTNESS_ONLY'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
