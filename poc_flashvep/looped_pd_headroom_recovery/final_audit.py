"""Check actual fixed-depth work and observed native Mixed batches after GPU work."""
from .common import *
from .analyze import records,table


def main():
    configure()
    from ..looped_pd_atomic_prefill import evidence
    evidence.ROOT=ROOT
    work=[];mixed=[];clocks=[];failed=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        c=json.loads(path.read_text())
        if not isinstance(c,dict) or 'name' not in c:continue
        folder=ROOT/'raw'/c['name'];terminal=folder/'dispatcher_result.json'
        if not terminal.exists():continue
        status=json.loads(terminal.read_text())
        if all((folder/f).exists() for f in ('before_measured.json','after_measured.json','requests.jsonl')):
            check=evidence.full_work(c['name'])
            row=dict(model=c['model'],length=c['length'],cap=c['cap'],complete_job=status['complete'],**check)
            work.append(row)
            if status['complete'] and not check['passed']:failed.append(c['name'])
        window=json.loads((folder/'window.json').read_text()) if (folder/'window.json').exists() else None
        for p in folder.glob('native_mixed_execution_*.json'):
            v=json.loads(p.read_text());total=sum(r['submissions'] for r in v['histogram'])
            joint=sum(r['submissions'] for r in v['histogram'] if r['mixed'])
            examples=[r for r in v['examples'] if window and window['start']<=r['time']<=window['end']]
            mixed.append(dict(name=c['name'],model=c['model'],worker=p.stem.rsplit('_',1)[1],
                complete_job=status['complete'],total_submissions_including_warmup=total,
                true_mixed_submissions_including_warmup=joint,measured_window_example_count=len(examples),
                measured_example_rows=[r['logical_rows'] for r in examples],examples=examples,
                evidence='MEASURED_NATIVE_SUBMISSION_SHAPES_AND_ACTUAL_SPAN_PHASES',
                scope='Histogram includes startup. Example timestamps prove actual measured-window shared-core batches where present. A simultaneous all-prefill correctness cohort need not contain any mixed batch.'))
        for p in folder.glob('cuda_only_clock_*.json'):
            start=json.loads(p.read_text());role=start['role']
            end=next((r for r in records(folder/f'cuda_only_events_{role}.jsonl') if r['event']=='METRIC_WINDOW_END'),None)
            if end is None:continue
            drift=(end['epoch_ns']-start['epoch_ns'])/1e9-(end['monotonic']-start['monotonic'])
            clocks.append(dict(name=c['name'],role=role,observation_seconds=end['monotonic']-start['monotonic'],
                epoch_minus_monotonic_relation_drift_seconds=drift,start_clock_pair_uncertainty_seconds=start['uncertainty_s'],
                end_clock_pair_uncertainty_seconds=end['uncertainty_s'],
                evidence='MEASURED_PAIRED_HOST_CLOCK_ANCHORS',scope='Check alignment drift separately from recorded device kernel duration. Do not infer a missing clock anchor from the first kernel.'))
    table('full_depth_checks.csv',work)
    table('native_mixed_batch_evidence.csv',mixed)
    table('profiler_clock_alignment_checks.csv',clocks)
    write(ROOT/'state/final_work_audit.json',dict(time=time.time(),checks=len(work),failed_complete_jobs=failed,
        passed=not failed,scope='Actual request recurrent rows. Excludes padded projection rows, native graph-capture warm executions, and separately recorded startup requests.'))
    preserve()
    print(json.dumps(dict(checks=len(work),failed=failed,mixed_workers=len(mixed),clock_pairs=len(clocks))))
    if failed:raise RuntimeError('Completed-job fixed-depth accounting failed. Do not publish a success verdict.')


if __name__=='__main__':main()
