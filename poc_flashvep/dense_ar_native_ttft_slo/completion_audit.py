"""Check terminal evidence without requiring a favorable scientific result."""
import argparse
import json
import re
import time
from pathlib import Path
from .common import ROOT,SOURCE,REPO,REPORT_NAMES,write,digest,preserve
from .failure_analysis import save_csv


def audit(final=False):
    preserve();requirements=[];inputs={};pending=[]
    def read(relative):
        path=ROOT/relative
        if not path.exists():return None
        inputs[relative]=digest(path)
        return json.loads(path.read_text())
    def record(identifier,title,okay,detail):
        requirements.append(dict(id=identifier,requirement=title,status='AUDITED' if okay else 'PENDING',detail=detail))
        if not okay:pending.append(identifier)
    raw=read('analysis/historical_main400_metric_reproduction.json')
    historical=read('analysis/historical_independent_metric_audit.json')
    record('A1','Reproduce the15 requested historical Main400 starts',bool(raw and raw['passed'] and len(raw['starts'])==15),
        'Published rounding and each-start metrics must match. This is raw recalculation,not a new GPU replication.')
    record('A2','Independently audit all209 historical timing starts',bool(historical and historical['passed'] and len(historical['results'])==209),
        'All25 metrics and request gate predicates stay frozen.')
    failures=read('analysis/request_failure_attribution.json')
    record('B','Eight failure masks,token weighting,all gate counterfactuals',bool(failures and failures['passed']),
        'Accounting and Shapley values are metric attribution,not runtime causal percentages.')
    traces=read('analysis/request_critical_paths.json');policies=read('analysis/policy_credit_batch_audit.json')
    causal=read('analysis/native_relative_causal_comparisons.json');delivery=read('analysis/worker_client_decode_gap_analysis.json')
    record('C','Exact request host paths,policy states and worker/client gaps',
        all(value and value['passed'] for value in (traces,policies,causal,delivery)),
        'Absent GPU boundaries remain NOT_INSTRUMENTED. Host-path and delivery identities do not identify exclusive GPU causes.')
    work=read('analysis/native_relative_actual_work_audit.json')
    reused=read('analysis/reused_one_factor_controls.json')
    record('C2','Actual prompt work and read-only reuse of prior one-factor controls',
        bool(work and work['passed'] and reused and reused['passed'] and len(reused['runs'])==15),
        'Inherited target metadata is not proof of matched work. Reused Heldout controls are not new Main400 measurements.')
    profile=read('analysis/serial_native_relative_kernel_analysis_v2.json')
    profile_terminal=bool(profile and len(profile['results'])==12 and all(row['evidence'] in
        ('ISOLATED_KERNEL_MEASUREMENT','DIAGNOSTIC','FAILED','NOT_RUN') for row in profile['results']))
    record('D0','Every prospective serial CPC has a terminal result',profile_terminal,
        'Counter/observer failures remain diagnostic. No favorable retry. Qualified paired N<3 has no start-level CI.')
    intervention=read('analysis/controlled_intervention_analysis.json')
    record('D1','Current one-factor effects and both Native decode gates',bool(intervention and intervention['passed']),
        'Natural/equal replay stay separate. Exact Common-D Native can differ across arms. Source-window mismatch stays explicit.')
    for identifier,plan_file,complete_file,expected in (
        ('D2','main400_native_relative_policy_confirmation_v2.json','main400_native_relative_complete_v2.json',39),
        ('D3','heldout192_native_relative_confirmation_v1.json','heldout192_native_relative_complete_v1.json',30)):
        plan=read('jobs/'+plan_file);complete=read('state/'+complete_file)
        terminal=False;detail=[]
        if plan and complete:
            lookup={row['name']:row for row in complete['results']}
            for block in plan['blocks']:
                for label,name in block['arms'].items():
                    row=lookup.get(name)
                    if row is None:
                        previous=[r for r in complete['results'] if r['label']==label and r['start']<block['start'] and
                            (r['status']=='FAILED' or r['status']=='COMPLETE' and not r.get('quality_passed',False))]
                        detail.append(dict(name=name,status='NOT_RUN_BY_PRIOR_FAILURE' if previous else 'MISSING',
                            blocking_names=[r['name'] for r in previous]))
                    else:detail.append(dict(name=name,status=row['status'],quality_passed=row.get('quality_passed'),reason=row.get('reason')))
            terminal=len(detail)==expected and all(row['status'] in
                ('COMPLETE','FAILED','NOT_RUN','NOT_RUN_BY_PRIOR_FAILURE') for row in detail)
        record(identifier,'Preserve every proposed cell in '+plan_file,terminal,
            dict(expected=expected,cells=detail,scope='Terminal status does not imply quality,decode safety or performance success.'))
    gates=[]
    for path in sorted((ROOT/'state').glob('ns_*_composition_v1.json')):
        proof=json.loads(path.read_text());gates.append(dict(name=path.name,passed=proof['passed']))
        inputs[str(path.relative_to(ROOT))]=digest(path)
    record('D4','Preserve all13 fresh composition gates',len(gates)==13,gates)
    dataset=read('analysis/dataset_identity_and_overlap_audit.json')
    arrival=read('analysis/arrival_seed_and_trace_audit.json');bridge=read('analysis/historical_current_runtime_bridge_audit.json')
    record('E','Frozen datasets,arrival offsets and explicit runtime bridge',
        all(value and value['passed'] for value in (dataset,arrival,bridge)),
        'Main/Heldout/Discovery and disjoint calibration sets are audited. Old/new runtime changes are not a one-factor comparison.')
    cpu=read('analysis/cpu_regression_audit.json')
    fresh={str(path.relative_to(REPO)):digest(path) for path in SOURCE.glob('*.py')}
    cpu_fresh=bool(cpu and cpu['passed'] and {row['path']:row['SHA256'] for row in cpu['source_files']}==fresh)
    record('F0','CPU tests and exact current source hashes',cpu_fresh,
        'The symptom reproduction remains expected red. Metric/correctness tests must be green.')
    for identifier,filename in (('F1','configuration_and_gate_inventory.json'),('F2','raw_metric_recalculation.json'),
            ('F3','gpu_budget_accounting.json')):
        value=read('analysis/'+filename)
        record(identifier,filename,bool(value and value['passed'] and (not final or value['final'])),
            'Final audit requires no live or interrupted owned work. Failed and NOT_RUN configurations remain listed.')
    manifest=read('analysis/figure_manifest.json');extra=read('analysis/intervention_figure_manifest.json')
    regen=read('analysis/figure_regeneration_audit.json')
    record('G0','Ten requested figures and separate serial/intervention plots',
        bool(manifest and manifest['passed'] and len(manifest['figures'])==10 and extra and extra['passed']),
        'Every figure has PNG,PDF and CSV. No missing cell receives an interpolated value.')
    record('G1','Exact PNG/PDF/CSV regeneration',bool(regen and regen['passed']),
        'Compare generated bytes from unchanged source tables. Metadata timestamps are excluded from PDF output.')
    reports=[]
    for name in REPORT_NAMES:
        path=ROOT.parent/name
        if not path.exists():reports.append(dict(name=name,status='MISSING'));continue
        text=path.read_text();broken=[]
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in link or link.startswith('#'):continue
            target=link.split('#')[0]
            if target and not (path.parent/target).exists():broken.append(link)
        reports.append(dict(name=name,status='PRESENT' if not broken else 'BROKEN_LINK',broken=broken,SHA256=digest(path)))
    record('G2','Five requested reports and local artifact links',
        len(reports)==5 and all(row['status']=='PRESENT' for row in reports),reports)
    verdict=read('analysis/scientific_verdict.json')
    record('H','Explicit scientific verdict and measurement limitations',bool(verdict and verdict.get('complete')),
        'An operationally complete campaign can retain PARTIAL_ROOT_CAUSE and failed scientific gates. No speedup is required.')
    result=dict(time=time.time(),passed=not pending,final=final,requirements=requirements,pending=pending,
        input_files_SHA256=inputs,report_audit=reports,
        scope='This audit checks evidence coverage and terminal execution states. It does not convert a negative result into scientific success. '
            'Resource restoration,lossless archiving and remote-SHA verification follow this science closeout gate.')
    write(ROOT/'state/science_completion_audit.json',result)
    save_csv(ROOT/'figures/science_completion_requirements.csv',requirements)
    if final:assert result['passed'],pending
    print('SCIENCE_COMPLETION_AUDIT',result['passed'],pending,flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();audit(args.final)
