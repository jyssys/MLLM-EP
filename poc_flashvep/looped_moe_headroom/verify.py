"""Integrity checks for actual model runs. Missing evidence is never a zero result."""
import json
from collections import defaultdict
from .common import ROOT,MODELS,digest,write
from .analyze import jsonl

def main():
    checks={}
    for phase in ['p1','calibration','heldout','timing','generation_fidelity','generation_quality']:
        plan=json.loads((ROOT/f'plan_{phase}.json').read_text())
        missing=[j['id'] for j in plan['jobs'] if not (ROOT/'raw'/j['id']/'complete.json').exists()]
        assert not missing,(phase,missing)
        checks[f'{phase}_jobs_complete']=len(plan['jobs'])
    p0=json.loads((ROOT/'plan_p0_diagnostic_v5.json').read_text())
    for job in p0['jobs']:
        path=ROOT/'raw'/job['id']
        c=json.loads((path/'complete.json').read_text())
        assert c['status']=='PASS'
        r=json.loads((path/'baseline_validation.json').read_text())
        assert r['no_op']['max_logit_error']==0 and r['q0_max_error']==0 and r['exact_tokens']
    checks['native_model_baselines']=3
    count=q0=matched=rounding_records=0
    groups=defaultdict(dict)
    for p in (ROOT/'raw').glob('*_p2_*_shard*_v1/quality.jsonl'):
        if not (p.parent/'complete.json').exists():
            continue
        for r in jsonl(p):
            count+=1
            assert r['finite'] and r['token_denominator']==r['length']-1
            if r['arm']=='B0':
                continue
            assert r['total_edges']==r['executed_edges']+r.get('omitted_edges',0)
            assert 0<=r['effective_removed_fraction']<=1
            groups[(r['model'],r['split'],r['id'])][r['arm']]=r
            if r['arm']=='q0':
                # Legacy CUDA FP32 mean of all-true values can equal 1 - 2^-24.
                # This does not relax a model-logit or loss tolerance.
                assert r['delta_nll']==0 and abs(r['top1_agreement']-1)<=2**-23 and r['kl_reference_to_variant']==0
                rounding_records+=r['top1_agreement']!=1
                q0+=1
    for key,arms in groups.items():
        assert len(arms)==27,(key,len(arms))
        for q in [.05,.1,.2,.3]:
            treatment=arms[f'O1_drop_{q}_renorm0']
            control=arms[f'N1_random_drop_{q}']
            assert treatment['omitted_edges']==control['omitted_edges']
            assert treatment['total_edges']==control['total_edges']
            matched+=1
    checks.update(full_model_quality_records=count,q0_sequences=q0,matched_random_budgets=matched,
        legacy_q0_float_mean_rounding_records=rounding_records)
    generation=0
    for p in (ROOT/'raw').glob('*_generation_*/generation_quality.jsonl'):
        if not (p.parent/'complete.json').exists():
            continue
        for r in jsonl(p):
            assert r['ids'] and len(r['ids'])<=r['cap']
            assert r['truncated']==(r['finish_reason']=='length')
            if r['arm']=='q0':
                assert r['token_exact'],'q0 generation differs'
            generation+=1
    checks['free_running_sequences']=generation
    strong=0
    for p in (ROOT/'raw').glob('*_independent_control_shard*_v*/quality.jsonl'):
        assert (p.parent/'complete.json').exists(),p
        cases=defaultdict(dict)
        for r in jsonl(p):
            cases[r['id']][r['arm']]=r
        for case,arms in cases.items():
            for q in [.05,.1,.2,.3]:
                treatment=arms[f'O1_drop_{q}_renorm0']
                control=arms[f'N1_independent_site_drop_{q}']
                assert treatment['omitted_edges']==control['omitted_edges'],(case,q)
                assert treatment['total_edges']==control['total_edges']
                strong+=1
    checks['independent_site_random_budgets']=strong
    assert strong==1920,strong
    for plan_name in ['micro_v2','heavy_profile','strict_foil_load']:
        plan=json.loads((ROOT/f'plan_{plan_name}.json').read_text())
        for job in plan['jobs']:
            path=ROOT/'raw'/job['id']
            assert (path/'complete.json').exists(),path
            assert json.loads((path/'complete.json').read_text())['status']=='PASS',path
        checks[f'{plan_name}_jobs_complete']=len(plan['jobs'])
    for model in MODELS:
        path=ROOT/'raw'/f'{model}_heavy_profile_v1'
        assert json.loads((path/'official_generation_api_parity.json').read_text())['pass_exact']
        clean=json.loads((path/'generation_clean_control.json').read_text())
        observed=json.loads((path/'generation_expert_events.json').read_text())
        assert observed['generated_ids']==clean['ids']
        assert observed['steps']==16
    checks['official_generation_api_parity']=3
    strict=json.loads((ROOT/'raw/foil1_strict_load_final_v1/metadata.json').read_text())
    assert not any(strict['loading_info'].values()),strict['loading_info']
    checks['foil_strict_load_no_missing_or_unexpected_tensors']=True
    for p in (ROOT/'raw').glob('*_micro_v2_start*/expert_microbench.jsonl'):
        for r in jsonl(p):
            assert r['q0_native_output_parity']
    contention=list((ROOT/'raw').glob('ifm_ouro_contention_*/complete.json'))
    assert len(contention)==27,len(contention)
    checks['contention_runs_complete']=len(contention)
    for p in (ROOT/'raw').glob('*/complete.json'):
        request=json.loads((p.parent/'request.json').read_text())
        assert request['model'] in MODELS
        assert int(request['gpu']) in range(8)
        complete=json.loads(p.read_text())
        if 'parameter_storage_and_versions_unchanged' in complete:
            assert complete['parameter_storage_and_versions_unchanged']
    manifest=ROOT/'workload_manifest.json'
    if not manifest.exists():
        manifest=ROOT/'workload_manifest_public.json'
    frozen=json.loads(manifest.read_text())['rows']
    sets={split:{r['id'] for r in frozen if r['split']==split} for split in ['smoke','calibration','heldout']}
    assert not sets['calibration']&sets['heldout'] and not sets['smoke']&sets['heldout']
    checks['tune_test_disjoint']=True
    write(ROOT/'validation.json',dict(status='PASS',checks=checks))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    main()
