"""Evaluate an exact fixed-cohort variant only after full qualification."""
import json
import time
from .common import ROOT,SOURCE,save,write,digest
from .batch_matched import ARMS,CONFIRM_SEEDS,performance_configuration
from .campaign import run_job
from .fixed_cohort_audit import audit


def configuration(method,layers,qps,seed,stage,count):
    cfg=performance_configuration(method,layers,16,qps,seed,stage,count)
    cfg.update(name=cfg['name'].replace('_P1D1_v1','_P1Prep1Decode2_v3'),
        D_admission_batch=2,matched_D_cohort_size=2,cohort_runtime_version=3,
        entry_module='poc_flashvep.dense_ar_matched_axis_pd.fixed_matched_cohort_v3',
        matched_request_batches=dict(P_prefill=1,D_prefill=1,D_decode=2),
        first_token_release='AFTER_ALL_COHORT_FIRST_NOTICES_EXIST_WITH_ORIGINAL_GENERATION_TIMESTAMPS')
    cfg.pop('matched_request_batch',None)
    return cfg


def matrix():
    values=[]
    for index,qps in enumerate((2.,6.,10.)):
        for method,layers in ARMS[index:]+ARMS[:index]:
            values.append(configuration(method,layers,qps,2026111810,'cohort2_discovery',72))
    for rep,seed in enumerate(CONFIRM_SEEDS):
        for method,layers in ARMS[rep:]+ARMS[:rep]:
            values.append(configuration(method,layers,6.,seed,'cohort2_confirmation',144))
    return values


def main():
    configs=matrix()
    path=ROOT/'state/cohort2_online_design.json'
    value=dict(names=[c['name'] for c in configs],total=42,
        discovery_starts=21,confirmation_starts=21,outputs=[16],
        actual_batches=dict(P_prefill=1,D_prefill=1,D_decode=2),B=8192,S=1024,
        discovery_count=72,confirmation_count=144,discovery_QPS=[2.,6.,10.],confirmation_QPS=6.,
        discovery_seed=2026111810,confirmation_seeds=list(CONFIRM_SEEDS),
        runtime_adapter_hashes=json.loads((ROOT/'state/fixed_cohort2_v3_qualification_design.json').read_text())['runtime_adapter_hashes'],
        conditional_on='Every Version3 State and actual Geometry gate must pass before performance.',
        fixed_SLO_SHA256=digest(ROOT/'configs/frozen_SLO_regimes.json'),
        scope='Fixed cohorts, not a Dynamic Cut scheduler. Match both axes at Source prep1, D suffix prep1 and Decode2. Keep actual first-token holding time in TTFT and all generation timestamps unchanged. Native keeps cached initial Decode2 and original P-token reuse. Use the same traces as Batch1 but report this stage separately. Never pool different batches or N72/N144 into one CI or curve. A desired crossover is not assumed.')
    if path.exists():
        assert {k:v for k,v in json.loads(path.read_text()).items() if k!='epoch'}==value
    else:
        write(path,dict(epoch=time.time(),**value))
    try:
        until=time.time()+3600
        while not (ROOT/'state/fixed_cohort2_v3_qualification_complete.json').exists():
            assert not (ROOT/'state/fixed_cohort2_v3_qualification_stopped.json').exists(),'Stop before unqualified performance'
            assert time.time()<until,'Qualification did not complete within the conditional budget'
            time.sleep(15)
        qualified=json.loads((ROOT/'state/fixed_cohort2_v3_qualification_complete.json').read_text())
        assert qualified['passed']
        for name in qualified['names']:
            assert audit(name)['passed']
        for index,cfg in enumerate(configs):
            assert all(digest(SOURCE/name)==expected for name,expected in value['runtime_adapter_hashes'].items())
            run_job(cfg)
            audit(cfg['name'])
            write(ROOT/'state/cohort2_online_progress.json',dict(epoch=time.time(),
                completed=index+1,total=len(configs),last=cfg['name']))
            print('COHORT2_ONLINE_PROGRESS',index+1,len(configs),cfg['name'],flush=True)
        write(ROOT/'state/cohort2_online_complete.json',dict(epoch=time.time(),total=len(configs),
            scope='Exact fixed-cohort trials completed. This does not establish a crossover, sustainable capacity or a policy contribution.'))
    except BaseException as error:
        write(ROOT/'state/cohort2_online_stopped.json',dict(epoch=time.time(),error=repr(error)))
        raise


if __name__=='__main__':
    main()
