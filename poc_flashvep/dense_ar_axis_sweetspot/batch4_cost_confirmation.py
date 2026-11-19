"""Separate unseen-batch cost bias from a failure of the frozen state rule."""
import fcntl
import json
import statistics
import time
from .common import ROOT,QUIET_LOCK,DEADLINE,bind_legacy,preserve,write,digest,resources
from .research_campaign import performance,freeze,run
from .model_validation import extract
from .decisions import record


def freeze_model(configs):
    original=ROOT/'configs/frozen_action_service_costs_v2.json'
    source=json.loads(original.read_text())
    model=json.loads(json.dumps(source['model']))
    grouped={a:[] for a in ('Layer','Token')}
    provenance=[]
    for cfg in configs:
        name=cfg['name'];_,_,cost=extract(ROOT,name)
        grouped[cfg['evaluation_arm']].append(cost)
        provenance.append(dict(name=name,cost_seconds=cost,
            configuration_SHA256=digest(ROOT/'configs'/(name+'.json')),
            request_log_SHA256=digest(ROOT/'raw'/name/'requests.jsonl')))
    for row in model['models']:
        if row['layers']==8 and row['batch']==4 and row['output']==16:
            costs=grouped[row['method']]
            row.update(cost_seconds={k:statistics.median(v[k] for v in costs) for k in costs[0]},
                training_starts=len(costs),status='NEW_BATCH4_INDEPENDENT_CALIBRATION',
                assumption='Three fresh calibration starts. No future confirmation metric is fitted.')
    model['training'].extend(provenance)
    model['scope']='Keep v2 rules and all old costs. Replace only Last8/Output16/Decode4 action costs using three fresh calibration starts per axis. The need for this revision was discovered on v2 seeds1930/1931. Evaluate v3 only on new seeds1970/1971/1972. No SLO, inference, batching or score equation changes.'
    path=ROOT/'configs/frozen_action_service_costs_v3.json'
    value=dict(based_on_v2_SHA256=digest(original),model=model)
    if path.exists():
        old=json.loads(path.read_text());assert old['model']==model and old['based_on_v2_SHA256']==value['based_on_v2_SHA256']
    else:
        write(path,dict(frozen_epoch=time.time(),**value))
    write(ROOT/'state/service_model_v3_frozen.json',dict(time=time.time(),path=str(path),SHA256=digest(path),
        evaluation_stage='batch4_knob_confirmation',training=provenance,
        scope='A separate revised cost model. Do not relabel failed v2 extrapolation tests as held-out v3 success.'))


def main():
    bind_legacy();preserve()
    calibration=[];confirmation=[]
    for rep,seed in enumerate((2026111960,2026111961,2026111962)):
        axes=('Layer','Token') if rep%2==0 else ('Token','Layer')
        calibration.extend(performance(a,8,16,6.,seed,batch=4,count=72,
            stage='new_batch_cost_calibration') for a in axes)
    for rep,seed in enumerate((2026111970,2026111971,2026111972)):
        for q in ((3.,6.) if rep%2==0 else (6.,3.)):
            arms=[('Native',0),('Layer',8),('Token',8)]
            arms=arms[rep:]+arms[:rep]
            confirmation.extend(performance(a,m,16,q,seed,batch=4,count=72,
                stage='batch4_knob_confirmation') for a,m in arms)
    freeze('new_batch_cost_calibration',calibration,
        'The first two prespecified v2 Decode4 tests show Token P service near227ms versus the extrapolated239ms. Distinguish a wrong unseen-batch cost table from wrong state variables. Measure three separate fresh calibration starts per axis at actual1/1/4. Do not use v2 test metrics as new training costs.')
    freeze('batch4_knob_confirmation',confirmation,
        'Freeze a separate v3 Lookup before reading any new seed1970/1971/1972 outcome. Same scalar score equations and frozen SLO. Compare new QPS3 with QPS6 under identical prep1/Decode4. Three paired independent starts, including Native for Decode5% checks. Keep v2 failures visible.')
    record('D0003','BATCH_AWARE_COST_DISCRIMINATOR',
        '새 Batch에서의 예측 실패는 비용표 외삽 오차일 수 있다.',
        '같은 비용표를 보정해도1~2변수의 상태 표현이 충분하지 않다.',
        'Batch4 새 Calibration6개 후 비용표v3만 동결한다. 새 Seed1970/1971/1972의 QPS3·6에서 원래 규칙을 검사한다.',
        'Batch-aware 비용표가 P 단계 순위를 복원한다면 고부하 Cycle 예측이 개선될 수 있다. 저부하에는 First-path와 Backlog가 중요할 수 있다.',
        '기존 v2의 첫 두 Batch4 실행에서 Token P Service는 약226~227ms다. v2는 약239ms를 사용했다.',
        '기존 규칙과 v2 결과를 변경하지 않는다. v3의 새 Held-out 검증과 잔여 예산을 검사한다.',
        'TRACE_ANALYSIS')
    while not (ROOT/'state/batch_extension_complete.json').exists():
        assert not (ROOT/'state/batch_extension_stopped.json').exists()
        assert time.time()<DEADLINE-3600
        time.sleep(10)
    try:
        assert (ROOT/'state/batch4_state_complete.json').exists()
        if time.time()>DEADLINE-2.5*3600:
            write(ROOT/'state/batch4_cost_confirmation_deferred.json',dict(time=time.time(),status='NOT_RUN',
                reason='The new calibration plus confirmation would threaten the final analysis/publication reserve.'))
            return
        if not run('new_batch_cost_calibration',calibration):return
        freeze_model(calibration)
        if run('batch4_knob_confirmation',confirmation):
            write(ROOT/'state/batch4_cost_confirmation_complete.json',dict(time=time.time(),
                scope='Separate v3 calibration and fresh-rate/seed confirmation. Scientific interpretation remains pending.'))
    except BaseException as error:
        write(ROOT/'state/batch4_cost_confirmation_stopped.json',dict(time=time.time(),error=repr(error)))
        raise
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            owner=resources()
            if not owner.processes():owner.restore(list(range(8)))
        preserve()


if __name__=='__main__':
    main()
