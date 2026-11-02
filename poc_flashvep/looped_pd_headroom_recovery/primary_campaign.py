"""Run bounded primary-length screens after original recovery controls."""
import copy
import statistics
import subprocess
from .common import *
from .initial_campaign import run,gate
from .prepare import put
from .search_plan import make


def request_rows(name):
    path=ROOT/'raw'/name/'requests.jsonl'
    return [r for r in (json.loads(s) for s in path.read_text().splitlines()) if not r['warmup']] if path.exists() else []


def subset_parity(reference,candidate):
    x={r['sample_id']:r for r in request_rows(reference)}
    y={r['sample_id']:r for r in request_rows(candidate)}
    common=set(x)&set(y)
    bad=[i for i in common if any(x[i][k]!=y[i][k] for k in ('input_ids','seed','output_ids','exit_depths','finish_reason'))]
    result=dict(reference=reference,candidate=candidate,compared=len(common),failed=bad,
        passed=bool(common) and not bad,evidence='MEASURED_COMMON_INPUT_SEED_TOKEN_OUTPUT',
        scope='Only the common requests are compared. Not original-HF fidelity.')
    write(ROOT/'raw'/candidate/'subset_output_parity.json',result)
    return result['passed']


def freeze_slo(record):
    from ..looped_pd_bottleneck_crossover.summary_tables import percentile
    from ..looped_pd_bottleneck_crossover.analyze_serving import token_intervals
    values=[]
    for name in record['calibration']:
        rows=request_rows(name)
        if len(rows)!=16:return False
        itls=[x for r in rows for x in token_intervals(r)[0]]
        values.append(dict(name=name,TTFT_ms=2*percentile([r['ttft_ms'] for r in rows],.95),
            TPOT_ms=2*percentile([r['tpot_ms'] for r in rows],.95),ITL_ms=2*percentile(itls,.99)))
    path=ROOT/'configs/frozen_SLO.json'
    slos=json.loads(path.read_text()) if path.exists() else {}
    key=record['model']+'_L'+str(record['length'])
    if key in slos:return True
    slos[key]=dict(TTFT_ms=min(v['TTFT_ms'] for v in values),TPOT_ms=min(v['TPOT_ms'] for v in values),
        ITL_ms=min(v['ITL_ms'] for v in values),requests=16,calibration=[v['name'] for v in values],
        rule='Two times serial calibration percentiles. Use the tighter per-metric limit across Atomic and C2048. Freeze before held-out pilot serving.',
        frozen_epoch=time.time(),calibration_values=values)
    write(path,slos)
    return True


def capacity(name):
    path=ROOT/'raw'/name/'cohorts.jsonl'
    values=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    values=[r for r in values if r.get('status')=='COMPLETE']
    rates=[r['job']['batch']/r['prefill_host_service_s'] for r in values]
    return max(rates,default=0.)


def main():
    plan=make()
    while True:
        state=json.loads((ROOT/'state/campaign.json').read_text())
        if state['stage']=='INITIAL_CONTROLS_COMPLETE_REVIEW_REQUIRED':break
        if state['stage']=='CONTROLLER_FAILED_REVIEW_REQUIRED':raise RuntimeError('Initial controller requires review')
        time.sleep(5)
    results=[]
    extra=plan['extra_ouro_control']
    if run(extra['gate']) and gate(extra['gate'],'recovery_ouro_8192_64_original_gate8_v2'):
        run(extra['pilot'])
    for record in plan['primaries']:
        model=record['model']
        write(ROOT/'state/primary_campaign.json',dict(stage='PRIMARY_CORRECTNESS',model=model,time=time.time()))
        valid=[]
        for name in record['gate']:
            okay=run(name) and gate(name,valid[0] if valid else None)
            if okay:valid.append(name)
            else:
                results.append(dict(model=model,status='BLOCKED_CORRECTNESS',name=name))
                break
        if len(valid)!=2:continue
        for name in record['capacity']:
            write(ROOT/'state/primary_campaign.json',dict(stage='PRIMARY_P_CAPACITY',model=model,name=name,time=time.time()))
            run(name)
        for name in record['calibration']:
            write(ROOT/'state/primary_campaign.json',dict(stage='PRIMARY_CALIBRATION',model=model,name=name,time=time.time()))
            run(name)
        if not freeze_slo(record):
            results.append(dict(model=model,status='CALIBRATION_FAILED'))
            continue
        pilot_reference=None
        for name in record['pilots']:
            write(ROOT/'state/primary_campaign.json',dict(stage='PRIMARY_QPS_PILOT',model=model,name=name,time=time.time()))
            okay=run(name)
            if okay:
                okay=subset_parity(valid[0],name)
                if pilot_reference is not None:
                    okay=okay and subset_parity(pilot_reference,name)
                else:pilot_reference=name
                configure()
                from ..looped_pd_atomic_prefill import evidence
                evidence.ROOT=ROOT
                okay=okay and evidence.full_work(name)['passed']
            results.append(dict(model=model,name=name,complete=okay))
            write(ROOT/'state/primary_results.json',results)
            if not okay:
                write(ROOT/'state'/f'primary_blocked_{model}.json',dict(time=time.time(),name=name,status='REVIEW_REQUIRED'))
                break
        subprocess.run([str(PYTHON),'-m','poc_flashvep.looped_pd_headroom_recovery.analyze'],cwd=REPO,check=True)
    write(ROOT/'state/primary_campaign.json',dict(stage='PRIMARY_MATRIX_COMPLETE_REVIEW_REQUIRED',time=time.time(),results=results))
    restore_if_idle()


if __name__=='__main__':main()
