"""Close the original numerical task before releasing the RCC-PD GPU gate."""
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,environment,write
from .resources import base
from .run import execute
from .summarize import rows,comparison


def main():
    state=ROOT/'state/final_release_checks.json'
    qualification=json.loads((ROOT/'state/huginn_adapter_qualification.json').read_text())
    if qualification['status']!='FINITE_QUALIFICATION_PASS':raise RuntimeError('Expanded Huginn qualification is not ready')
    devices=[4,5];results=[]
    try:
        base.stop(devices)
        common=dict(model='huginn',variant='down_norm_adapter',workload='diagnostic',
                    seed=2026103001,gpus=devices,timeout=1800,diagnostic=True)
        jobs=[]
        for index,arm in enumerate(('PD_G1','B')):
            jobs.append(dict(common,name=f'huginn_release_KV_{arm}_{index}',arm=arm,dataset='W1',
                count=4,cap=8,rate=0.,capacity=8,env={'ABC_VERIFY_KV':'1'}))
        for index,arm in enumerate(('PD_G1','B')):
            jobs.append(dict(common,name=f'huginn_adapter_order8_capacity3_{arm}_{index}',arm=arm,
                dataset='W1_reversed8',count=8,cap=128,rate=0.,capacity=3,
                env={'B_NUMERICAL_PRESERVE_SOURCE_SEEDS':'1'}))
        jobs.append(dict(common,name='huginn_adapter_native_HF_triage8_Native_0',arm='Native',dataset='correctness',
             count=8,cap=32,rate=0.,capacity=1,seed=20261029,gpus=[4],env={}))
        write(ROOT/'configs/final_release_plan.json',jobs)
        for job in jobs:
            write(state,dict(status='RUNNING',job=job['name'],results=results,time=time.time()))
            value=execute(job);results.append(dict(name=job['name'],**value))
            if value['status']!='COMPLETE':raise RuntimeError('A required release probe failed: '+job['name'])
        name='teacher_huginn_adapter_geometry16_v10'
        command=[PYTHON,'-m','poc_flashvep.looped_pd_b_numerical_diagnosis.teacher_replay',
            '--case','huginn_closed_loop','--sample','56','--failure-manifest','configs/new_seed_failure_manifest.json',
            '--sizes','1,2,3,4,8,16','--variant','down_norm_adapter','--track-all','--kv-hash','--name',name]
        with (ROOT/'logs'/f'{name}.log').open('a') as log:
            value=subprocess.run(command,cwd=REPO,env=environment([4]),stdout=log,stderr=subprocess.STDOUT,timeout=900)
        if value.returncode:raise RuntimeError('The geometry release probe failed')
        replay=json.loads((ROOT/'raw'/name/'result.json').read_text());hashes=replay['prefill_KV_hashes']
        assert not replay['first_observed_difference'] and len({r['top1'] for r in replay['outputs']})==1
        assert all(r['status']=='COMPLETE' and r['all_clone_rows_logit_exact'] for r in replay['outputs'])
        assert all(r['entries']==hashes[0]['entries'] and r['final_hidden_sha256']==hashes[0]['final_hidden_sha256'] for r in hashes)
        checks=[]
        for job in jobs[:2]:
            folder=ROOT/'raw'/job['name'];closed=json.loads((folder/'closed_stats.json').read_text())
            tids={r['transfer_id'] for r in closed['metrics'] if r['request_id'].startswith('r0_')}
            values=[r for p in folder.glob('KV_checks-*.jsonl') for line in p.read_text().splitlines()
                    if (r:=json.loads(line))['tid'] in tids]
            assert len(values)==4 and all(r['passed'] for r in values)
            checks.append(dict(run=job['name'],N=len(values),passed=sum(r['passed'] for r in values)))
        paired=comparison(jobs[0]['name'],jobs[1]['name']);assert paired['exact']==paired['N']==4
        reference=rows(ROOT/'raw/huginn_down_norm_adapter_qual_N8_seed2026103001_PD_G1_0')
        order=[]
        for job in jobs[2:4]:
            actual={r['source_sample_id']:r for r in rows(ROOT/'raw'/job['name']).values()}
            ids=sorted(reference.keys()&actual.keys())
            exact=sum(reference[i]['output_ids']==actual[i]['output_ids'] and reference[i]['seed']==actual[i]['seed'] for i in ids)
            assert exact==len(ids)==8
            order.append(dict(run=job['name'],N=8,exact=exact))
        write(state,dict(status='INTERNAL_CORRECTNESS_RELEASE_PASS',results=results,KV_checks=checks,
            B_output_check=paired,order_checks=order,geometry_rows=[r['rows'] for r in replay['outputs']],
            original_HF_equivalence='SEPARATE_EIGHT_PROMPT_TRIAGE_ONLY',answer_accuracy='NOT_MEASURED',time=time.time()))
    except BaseException as error:
        write(state,dict(status='BLOCKED_RELEASE_CHECK',error=repr(error),results=results,time=time.time()))
        raise
    finally:
        base.restore(devices)


if __name__=='__main__':main()
