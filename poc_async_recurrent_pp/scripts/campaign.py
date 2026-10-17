"""Quality-gated controller. It never starts a distributed approximation engine."""
import subprocess
import time
from poc_async_recurrent_pp.contract import *
from poc_async_recurrent_pp.scripts.aggregate import summarize


def job(gpus,name,module,*args):
    command=[PYTHON,'-m','poc_async_recurrent_pp.scripts.gpu_session','--gpus',gpus,
        '--output','outputs/'+name,'--timeout','21600','--',PYTHON,'-m',module,*map(str,args)]
    print('START',name,flush=True)
    result=subprocess.run(command,cwd=PROJECT)
    if result.returncode:
        raise RuntimeError('Preserved failed run: '+name)
    print('FINISH',name,flush=True)


def main():
    target=ROOT/'outputs/baseline_v1/status.json'
    deadline=time.monotonic()+7200
    while not target.exists():
        guard=ROOT/'outputs/baseline_v1/guard_status.json'
        if guard.exists() and not read(guard)['success']:
            raise RuntimeError('BASELINE_INVALID: baseline guard reported failure')
        if time.monotonic()>deadline:
            raise TimeoutError('Baseline certification did not finish')
        time.sleep(5)
    assert read(target)['pass_gate']
    # The exact task-sanity job can continue after a CPU controller restart.
    tasks=ROOT/'outputs/baseline_tasks_v1'
    if tasks.exists():
        while not (tasks/'guard_status.json').exists():
            time.sleep(5)
        assert read(tasks/'guard_status.json')['success']
    else:
        job('4','baseline_tasks_v1','poc_async_recurrent_pp.scripts.check_baseline_tasks')
    job('4','diagnostics_v2','poc_async_recurrent_pp.scripts.diagnostics')
    job(MASK,'pilot_cap2048_v1','poc_async_recurrent_pp.scripts.quality_wave','--cap',2048,'--per-dataset',16)
    wave=ROOT/'outputs/pilot_cap2048_v1'
    rows=summarize(wave)
    write(wave/'paired_summary.json',dict(rows=rows,bootstrap_resamples=10000,sample_unit='Prompt'))
    if any(row['truncated'] for row in rows):
        # Freeze a common, higher cap and identical prompt batches across all arms.
        job(MASK,'pilot_cap8192_v1','poc_async_recurrent_pp.scripts.quality_wave','--cap',8192,'--per-dataset',16)
        wave=ROOT/'outputs/pilot_cap8192_v1'
        rows=summarize(wave)
        write(wave/'paired_summary.json',dict(rows=rows,bootstrap_resamples=10000,sample_unit='Prompt'))
    clear_failure=all(row['delta_pp_CI95'][1]<-10 for row in rows if row['arm']!='exact')
    verdict='QUALITY_FAIL' if clear_failure else 'INCONCLUSIVE'
    write(ROOT/'outputs/campaign_status.json',dict(verdict=verdict,final_wave=str(wave),
        full_quality_benchmark='NOT_RUN',distributed_latency_oracle='NOT_RUN',distributed_prototype='NOT_RUN',
        reason='All W1/W0 candidates materially fail paired pilot screening.' if clear_failure else
               'Pilot does not justify a quality claim. Further fixed-protocol evaluation is required.',
        quality_GO_requires=dict(gsm8k_n=1319,math500_n=500),summary=rows))
    print('CAMPAIGN',verdict,flush=True)


if __name__=='__main__':
    main()
