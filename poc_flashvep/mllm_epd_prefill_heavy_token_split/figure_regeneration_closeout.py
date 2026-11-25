"""Reproduce required figures in guarded fresh output folders after serving."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

from .common import ROOT,REPO,PYTHON,digest,preserve,resources,write
from .delivery_requirement_inventory import FIGURES
from .resource_closeout import require_final_closeout

JOBS=[
    dict(module='main100_result_figures',args=[],drop=['main100_actual_','main100_C37_failure_startmeans','main100_result_figures_v0']),
    dict(module='baseline_confirmation_analysis',args=[],drop=['main100_C14_','main100_baseline_confirmation_C14_']),
    dict(module='token_confirmation_analysis',args=[],drop=['token_confirmation_C37_']),
    dict(module='token_discovery_analysis',args=[],drop=['token_discovery_C34_']),
    dict(module='profile_analysis',args=['ec_LOW_CUPTI_C25_CPC_q0.5_v0','--proxy-name','ec_LOW_CUPTI_C25_proxy'],
        drop=['ec_LOW_CUPTI_C25_CPC_q0.5_v0_idle_']),
    dict(module='qualified_low_load_idle_report',args=[],drop=['low_load_C25_evidence_v0']),
    dict(module='idle_scope_figures',args=['--version','v1'],drop=['idle_scope_activity_v1','idle_scope_figure_audit_v1']),
    dict(module='suffix_quantum_cost_report',args=['--tag','suffix_QUANTA_C41R3'],
        drop=['suffix_QUANTA_C41R3_summary_v0','suffix_QUANTA_C41R3_real_forward_events_v0',
            'suffix_QUANTA_C41R3_request_geometry_v0','suffix_QUANTA_C41R3_physical_gap_fit_oracle_v0',
            'suffix_QUANTA_C41R3_empty_schedule_events_v0','suffix_QUANTA_C41R3_report_audit_v0']),
    dict(module='interference_probe_analysis',args=['--tag','interference_PROBE_C40R4'],
        drop=['interference_PROBE_C40R4_starts_v0','interference_PROBE_C40R4_failure_categories_v0',
            'interference_PROBE_C40R4_probes_v0','interference_PROBE_C40R4_memory_v0',
            'interference_PROBE_C40R4_paired_ratios_v0','interference_PROBE_C40R4_audit_v0']),
    dict(module='parallelism_feasibility_figure',args=[],drop=['parallelism_feasibility_structural_v0'],
        input_report='mllm_epd_dual_side_parallelism_feasibility_v1.md'),
]


def allowed_write(sandbox,path):
    if isinstance(path,int):return
    resolved=Path(os.fsdecode(path)).resolve();sandbox=sandbox.resolve()
    if resolved!=sandbox and sandbox not in resolved.parents:
        raise PermissionError('Figure reproduction cannot write outside its fresh output folder: '+str(resolved))


def analysis_link_allowed(job,name):
    if name.startswith(tuple(job['drop'])):return False
    # This generator calls the actual metric writer. Its derived outputs must
    # be new files, never links that resolve to frozen original metrics.
    if job['module']=='baseline_confirmation_analysis' and name.endswith('_metrics.json'):return False
    return True


def install_write_guard(sandbox):
    """Guard Python file writes, not an arbitrary external-process sandbox."""
    flags=os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND
    def audit(event,args):
        if event=='open':
            path,mode,options=args
            if int(options or 0)&flags or mode and any(value in mode for value in 'wax+'):
                allowed_write(sandbox,path)
        elif event in ['os.mkdir','os.remove','os.rmdir']:
            allowed_write(sandbox,args[0])
        elif event in ['os.rename','os.link','os.symlink']:
            allowed_write(sandbox,args[0]);allowed_write(sandbox,args[1])
    sys.addaudithook(audit)


def CSV_rows(path,namespace=None):
    with path.open(newline='') as stream:
        reader=csv.reader(stream);rows=list(reader)
    if namespace is not None:
        rows=[[cell.replace(str(namespace),str(ROOT)) for cell in row] for row in rows]
    return rows


def compare_bundle(original,generated,stem,csv_path=None):
    before=original/'figures'/f'{stem}.png';after=generated/'figures'/f'{stem}.png'
    original_csv=original/csv_path if csv_path else original/'figures'/f'{stem}.csv'
    generated_csv=generated/csv_path if csv_path else generated/'figures'/f'{stem}.csv'
    with Image.open(before) as left,Image.open(after) as right:
        pixels=left.size==right.size and left.convert('RGBA').tobytes()==right.convert('RGBA').tobytes()
    csv_equal=CSV_rows(original_csv)==CSV_rows(generated_csv,generated)
    pdf=generated/'figures'/f'{stem}.pdf';content=pdf.read_bytes()
    pdf_basic=content.startswith(b'%PDF-') and b'%%EOF' in content[-1024:]
    return dict(stem=stem,CSV_exact_after_output_path_normalization=csv_equal,PNG_pixels_exact=pixels,
        PDF_header_and_EOF=pdf_basic,PDF_pixel_render_verification='NOT_MEASURED_NO_INSTALLED_RENDERER',
        original_PNG_SHA256=digest(before),regenerated_PNG_SHA256=digest(after),
        original_CSV_SHA256=digest(original_csv),regenerated_CSV_SHA256=digest(generated_csv),
        regenerated_PDF_SHA256=digest(pdf),passed=csv_equal and pixels and pdf_basic,
        scope='Actual figure re-execution. PNG pixels and CSV values are compared. PDF metadata and rendered pixels are not claimed equal.')


def worker(job,sandbox):
    require_final_closeout()
    assert ROOT/'verification' in sandbox.resolve().parents
    job=JOBS[job];namespace=sandbox/'evidence'
    sys.dont_write_bytecode=True
    os.environ['MPLCONFIGDIR']=str(sandbox/'mplconfig')
    os.environ['XDG_CACHE_HOME']=str(sandbox/'cache')
    os.environ['TMPDIR']=str(sandbox/'tmp')
    for path in [namespace/'analysis',namespace/'figures',sandbox/'mplconfig',sandbox/'cache',sandbox/'tmp']:
        path.mkdir(parents=True,exist_ok=True)
    for name in ['data','raw','configs','state','references']:(namespace/name).symlink_to(ROOT/name,target_is_directory=True)
    for path in (ROOT/'analysis').iterdir():
        if path.is_file() and analysis_link_allowed(job,path.name):
            (namespace/'analysis'/path.name).symlink_to(path)
    if 'input_report' in job:(sandbox/job['input_report']).symlink_to(ROOT.parent/job['input_report'])
    install_write_guard(sandbox)
    module=importlib.import_module('poc_flashvep.mllm_epd_prefill_heavy_token_split.'+job['module'])
    if job['module']=='baseline_confirmation_analysis':
        from . import metrics
        metrics.ROOT=namespace
    module.ROOT=namespace;sys.argv=[job['module']]+job['args'];module.main()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='v0')
    parser.add_argument('--worker',type=int);parser.add_argument('--sandbox',type=Path)
    args=parser.parse_args()
    if args.worker is not None:
        assert 0<=args.worker<len(JOBS) and args.sandbox is not None
        worker(args.worker,args.sandbox);return
    assert args.version.startswith('v') and args.version[1:].isdigit();preserve()
    require_final_closeout()
    owner=resources();burners=owner.burners();processes=owner.processes()
    assert set(burners)==set(range(8)),'Run figure regeneration after final burner restoration.'
    assert {r['gpu'] for r in processes}==set(range(8))
    assert all(Path('/proc',str(r['pid'])).stat().st_uid==os.getuid() and
        os.getsid(r['pid'])==burners[r['gpu']] for r in processes)
    output=ROOT/'verification'/f'figure_regeneration_{args.version}';assert not output.exists();output.mkdir(parents=True)
    records=[];comparisons=[];errors=[]
    required={(stem,csv_path) for _,_,stems,_,csv_path in FIGURES for stem in stems}
    found=set()
    for index,job in enumerate(JOBS):
        sandbox=output/job['module'];sandbox.mkdir()
        command=[str(PYTHON),'-m',__package__+'.figure_regeneration_closeout','--worker',str(index),'--sandbox',str(sandbox)]
        log=sandbox/'execution.log'
        with log.open('xb') as stream:result=subprocess.run(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
        records.append(dict(module=job['module'],command=command,returncode=result.returncode,log_SHA256=digest(log)))
        if result.returncode:
            errors.append(dict(module=job['module'],kind='REGENERATION_FAILED',log=str(log)));continue
        generated=sandbox/'evidence'
        for stem,csv_path in sorted(required,key=lambda row:row[0]):
            if not (generated/'figures'/f'{stem}.png').exists():continue
            try:
                comparison=compare_bundle(ROOT,generated,stem,csv_path);comparisons.append(dict(module=job['module'],**comparison))
                if comparison['passed']:found.add((stem,csv_path))
                else:errors.append(dict(module=job['module'],kind='FIGURE_MISMATCH',stem=stem))
            except Exception as error:errors.append(dict(module=job['module'],stem=stem,error=type(error).__name__+': '+str(error)))
        print('FIGURE_REGENERATION_JOB',index,job['module'],result.returncode,flush=True)
        preserve()
    missing=sorted((stem,str(csv_path)) for stem,csv_path in required-found)
    target=ROOT/'analysis'/f'figure_regeneration_closeout_{args.version}.json';assert not target.exists()
    write(target,dict(time=time.time(),jobs=records,comparisons=comparisons,errors=errors,missing=missing,
        passed=not errors and not missing,original_raw_and_figures_unchanged=True,
        source_SHA256=digest(__file__),fresh_output_folder=str(output),
        scope='Required figures only. Python writes are confined to fresh output folders. '
            'This is not a general OS sandbox. Read-only preservation Git commands are the only intended external subprocesses in generators. '
            'No benchmark, model execution, profiler or statistical unit was changed. PDF checks are explicitly limited.'))
    preserve();assert not errors and not missing,'Preserve failed regeneration and inspect exact differences without replacing frozen figures.'
    print('REQUIRED_FIGURE_REGENERATION_PASS',len(comparisons),flush=True)


if __name__=='__main__':main()
