"""Preserve prior derived CIs, record the correction and restart our monitor."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,preserve,digest
from .monitor import alive
from .resampling import mean_interval,METHOD
from .decisions import record


def main():
    preserve()
    target=ROOT/'state/bootstrap_CDF_correction.json'
    assert not target.exists(),'Preserve the first correction record'
    previous=json.loads((ROOT/'state/monitor_process.json').read_text())
    pid=previous['pid']
    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'poc_flashvep.dense_ar_axis_sweetspot.monitor' in command
    assert Path(f'/proc/{pid}').stat().st_uid==os.getuid()
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if not alive(pid):break
        time.sleep(.05)
    assert not alive(pid)
    write(ROOT/'state/monitor_v1_before_CDF_correction.json',previous)
    saved=[]
    for name in ('independent_start_statistics.json','service_queue_causal_decomposition.json',
            'frozen_model_validation_and_oracle.json','recorded_observable_knob_validation.json'):
        source=ROOT/'analysis'/name
        if not source.exists():continue
        dest=ROOT/'analysis/bootstrap_CDF_correction_prior_derived'/name
        dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
        shutil.copyfile(source,dest)
        saved.append(dict(path=str(dest.relative_to(ROOT)),SHA256=digest(dest)))
    from ..dense_ar_layer_token_pd.statistics_report import mean_interval as old_interval
    fixture=[.9197886734342793,.93284887555283,1.0000237259695057]
    old,new=old_interval(fixture),mean_interval(fixture)
    assert old[1]<1<=new[1]
    write(target,dict(time=time.time(),method=METHOD,old_CI=old,new_CI=new,fixture=fixture,
        draw_count=27,endpoint_probability=1/27,tail_probability=.025,prior_derived=saved,
        scope='The original CI interpolated the sorted27 exact bootstrap means as if they were27 measured observations. Endpoint mass1/27 exceeds2.5%. New inverse empirical-CDF quantiles retain that mass. This changes new derived CIs, not raw measurements, frozen historical reports, SLOs or cost models. Nominal bootstrap coverage remains limited by only three independent starts.'))
    record('S0001','BOOTSTRAP_CDF_CORRECTION',
        'Exact Bootstrap의 재표본 확률질량을 유지해야 한다.',
        '27개 Exact Draw에 선형 분위수 보간을 적용해도 동일한 CI가 나온다.',
        '독립 Start3개의 모든27개 재표본을 검사한다. 극단 Draw의 확률1/27과 Tail2.5%를 비교한다.',
        '확률3.70%인 최대 Draw를97.5% 분위수에서 제거하면 안 된다.',
        '기존 보간 CI 상한0.98547은 신규 empirical-CDF에서1.00002가 됐다. 원시 값은 같다.',
        '신규 분석 전체에 같은 수정 방법을 적용한다. 이전 파생 Index를 별도로 보존한다.',
        'STATISTICAL_METHOD_CORRECTION')
    log=ROOT/'logs/monitor_v2.log'
    with log.open('ab') as stream:
        process=subprocess.Popen([PYTHON,'-m','poc_flashvep.dense_ar_axis_sweetspot.monitor'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/monitor_process.json',dict(time=time.time(),pid=process.pid,module='monitor',
        log=str(log),version=2,reason='Reload corrected analysis functions. No GPU worker is stopped.'))
    preserve();print('BOOTSTRAP_CDF_CORRECTED_MONITOR_RESTARTED',process.pid,old,new,flush=True)


if __name__=='__main__':
    main()
