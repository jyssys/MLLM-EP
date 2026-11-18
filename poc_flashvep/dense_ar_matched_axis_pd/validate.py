"""Check actual inputs, frozen worker bundles and final matrix coverage."""
import argparse
import collections
import hashlib
import json
import time
import zipfile
from .common import ROOT, MODEL_REVISION, digest, preserve, write
from .host_paths import records
from .runtime_freeze import check


def frozen_bundle(path,expected):
    with zipfile.ZipFile(path) as bundle:
        observed={name:hashlib.sha256(bundle.read(
            'poc_flashvep/dense_ar_matched_axis_pd/'+name)).hexdigest() for name in expected}
        site=hashlib.sha256(bundle.read('sitecustomize.py')).hexdigest()
    assert observed==expected,'Frozen inference adapter differs inside the executed bundle'
    assert site==expected['site_bootstrap.py'],'Worker startup uses another bootstrap'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    preserve();check()
    data=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    expected=json.loads((ROOT/'state/frozen_new_inference_sources_v3.json').read_text())['SHA256']
    conditions={};checks=[]
    for start in data['starts']:
        name=start['name'];folder=ROOT/'raw'/name
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        assert cfg==json.loads((folder/'configuration.json').read_text()),name
        provenance=json.loads((folder/'input_provenance.json').read_text())
        assert provenance['model_revision']==MODEL_REVISION,name
        assert provenance['dataset_SHA256']==digest(ROOT/'data'/cfg['dataset']),name
        assert provenance['SLO_SHA256']==digest(ROOT/'configs/frozen_SLO_regimes.json'),name
        assert provenance['config_SHA256']==digest(ROOT/'configs'/(name+'.json')),name
        assert provenance['GPU_mapping']==cfg['gpus'],name
        assert cfg['execution_contract_version']==3
        receipt=json.loads((folder/'dispatcher_result.json').read_text())
        assert receipt['complete'] and receipt['normal_worker_exit'] and receipt['timing_isolation_passed'],name
        workers=json.loads((folder/'unified_worker_gate.json').read_text())
        assert workers['passed'] and len(workers['workers'])==2,name
        command=json.loads((folder/'command.json').read_text())
        assert command['source_archive_SHA256']==digest(folder/'source.zip'),name
        frozen_bundle(folder/'source.zip',expected)
        rows=sorted(records(folder/'requests.jsonl'),key=lambda r:r['sample_id'])
        assert [r['sample_id'] for r in rows]==list(range(cfg['count'])),name
        assert all(r['input_tokens']==8000 and r['output_tokens']==len(r['output_ids'])==cfg['request_caps'][r['sample_id']] for r in rows),name
        for row in rows:
            events=row['token_events']
            assert [e['count'] for e in events]==list(range(1,row['output_tokens']+1)),name
            assert all(b['time']>=a['time'] for a,b in zip(events,events[1:])),name
        identity=hashlib.sha256(json.dumps([(r['prompt_id'],r['input_ids'],r['output_ids'])
            for r in rows],separators=(',',':')).encode()).hexdigest()
        key=(start['stage'],start['cap'],start['QPS'],start['seed'],start['count'],start['admission'])
        old=conditions.setdefault(key,identity)
        assert old==identity,'Actual prompt/output identity differs between paired methods: '+name
        value=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
        assert value['output_gate']['passed'] and value['unified_worker_gate'],name
        if start['method']!='Agg':
            assert value['actual_query_coverage_gate']['passed'] and value['per_request_P_query_coverage']['passed'],name
            assert value['common_reservation_gate']['passed'],name
        if start['method'] in ('Layer','Token'):
            assert value['exact_GEMM_row_gate'],name
            assert value['work_accounting']['completed_migrated_requests']==cfg['count'],name
            padded={}
            for role in ('prefill','decode'):
                shapes=json.loads((folder/f'measured_geometry_{role}.json').read_text())['counts']
                layers=(36-start['layers'] if role=='prefill' else start['layers']) if start['method']=='Layer' else 36
                padded[role]=sum(json.loads(shape)['padded_query_tokens']*count*layers
                    for shape,count in shapes.items() if role=='prefill' or json.loads(shape)['prefilling'])
            work=value['work_accounting']
            assert padded['prefill']==work['P_GEMM_token_layer_rows'],(name,padded)
            assert padded['decode']==work['D_GEMM_token_layer_rows'],(name,padded)
        else:
            padded=None
        checks.append(dict(name=name,passed=True,actual_request_identity_SHA256=identity,
            client_one_event_per_output_token=True,
            source_archive_SHA256=command['source_archive_SHA256'],
            actual_padded_prompt_token_layer_rows=padded))
    stages=collections.Counter(r['stage'] for r in data['starts'])
    if args.final:
        from .monitor import snapshot
        current=snapshot()
        assert current['ready_for_final_analysis'] and not current['critical_failures'],current
        manifest=json.loads((ROOT/'state/matrix_manifest.json').read_text())
        confirmed=json.loads((ROOT/'state/confirmation_manifest.json').read_text())
        names={r['name'] for r in data['starts']}
        from .recovery import resolve_name,resolve_configuration,record
        planned=[resolve_name(n) for n in manifest['expected']]
        repeats=[resolve_name(n) for n in confirmed['expected']]
        assert len(planned)==len(set(planned))==232 and set(planned)<=names
        assert len(repeats)==len(set(repeats))==120 and set(repeats)<=names
        for original,replacement in record().get('replacements',{}).items():
            old=json.loads((ROOT/'configs'/(original+'.json')).read_text())
            assert resolve_configuration(old)==json.loads((ROOT/'configs'/(replacement+'.json')).read_text())
            assert original not in names and replacement in names
        supplement=json.loads((ROOT/'state/admission_confirmation_manifest.json').read_text())
        assert len(supplement['expected'])==28 and set(supplement['expected'])<=names
        assert stages['admission_control']==52
        if (ROOT/'state/last2_supplement_cancelled.json').exists():
            assert not any(n.startswith('last2_') for n in names),'Cancelled supplemental runs must not enter the primary matrix'
        else:
            added=json.loads((ROOT/'state/last2_supplement_manifest.json').read_text())
            assert len(added['discovery'])==58 and len(added['confirmation'])==30
            assert len(added['expected'])==88 and set(added['expected'])<=names
            for n,o in [(n,16) for n in (8,32,128)]+[(8,200)]:
                assert json.loads((ROOT/'analysis'/f'last2_qualification_N{n}_O{o}.json').read_text())['passed']
        exported=json.loads((ROOT/'analysis/tabular_export_manifest.json').read_text())
        assert exported['request_rows']==sum(r['count'] for r in data['starts'])
        assert exported['starts']==len(data['starts'])
    write(ROOT/'analysis'/('final_integrity_validation.json' if args.final else 'partial_integrity_validation.json'),
        dict(time=time.time(),passed=True,final=args.final,starts=len(checks),stages=dict(stages),
            paired_conditions=len(conditions),runs=checks,
            scope='Check actual request IDs, prompt tokens, output tokens, worker B/S gates, work coverage, KV reservations and frozen sources inside each executed ZIP. This does not turn finite cohorts into sustainable capacity or qualify failed GPU profiling. Independent full-state qualification remains separate.'))
    print('INTEGRITY_VALIDATION',len(checks),'FINAL' if args.final else 'PARTIAL',flush=True)


if __name__=='__main__':
    main()
