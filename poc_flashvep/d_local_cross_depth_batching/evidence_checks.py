"""Check final claims against completed artifacts before publication."""
import csv
import re
import subprocess
import time
from .common import *
from .report import FINAL


def main():
    initialize()
    decisions=json.loads((ROOT/'final_model_decisions.json').read_text())
    serving={v['name']:v for v in csv.DictReader((ROOT/'serving_runs.csv').open())}
    work={v['name']:v for v in csv.DictReader((ROOT/'work_accounting.csv').open())}
    checked=[]
    for model in ('ouro','huginn'):
        assert all(decisions[model]['N32_generic_state_qualification'].values())
        assert decisions[model]['corrected_core_grid_exact'] is True
        for method in ('full','chunked','paced'):
            checked.append(f'serve_{model}_{method}_c128_fixed25_v5')
        checked.append(f'serve_{model}_paced_heavy_fixed25_'+('clean_restart_v2' if model=='ouro' else 'v1'))
    for name in checked:
        assert serving[name]['complete']=='True' and serving[name]['serving_output_exact']=='True',name
        assert serving[name]['dispatcher_timing_qualification']=='NORMAL_GUARDED_EXECUTION',name
        assert work[name]['full_fixed_work_exact']=='True',name
        assert work[name]['final_device_requests_and_KV_blocks_empty']=='True',name
    figures=json.loads((ROOT/'figure_manifest.json').read_text())
    assert len(figures)==10
    for item in figures:
        assert (ROOT/item['PNG']).is_file() and (ROOT/item['SVG']).is_file()
    deferred={'publication_receipt.json','final_remote_verification.json'};links=[]
    for target in re.findall(r'\]\(([^)]+)\)',FINAL.read_text()):
        if target.startswith(('https://','http://')):continue
        path=FINAL.parent/target
        assert path.exists() or path.name in deferred,('Missing report target',target)
        links.append(dict(target=target,exists=path.exists(),deferred_publication_metadata=path.name in deferred))
    restoration=json.loads((ROOT/'state/final_resource_restoration.json').read_text())
    assert restoration['verified'] and restoration['all_GPU_utilization_nonzero']
    prose=[FINAL,SOURCE/'README.md',ROOT/'source_and_prior_work_audit.md',ROOT/'measurement_contract_and_limitations.md']
    command=['python','/home/esjung/.codex/skills/asd-ste100/scripts/ste-lint.py','--json',*map(str,prose)]
    lint=subprocess.run(command,capture_output=True,text=True)
    result=json.loads(lint.stdout)
    assert lint.returncode==0 and result['hard_count']==0,lint.stdout+lint.stderr
    write(ROOT/'state/report_quality_checks.json',dict(time=time.time(),checked_primary_runs=checked,
        primary_output_fixed_work_and_release_passed=True,figures=len(figures),report_targets=links,
        style_lint=result,style_command=command,
        scope='Mechanical checks preserve raw evidence. Descriptive passive-voice advisories remain. This is not certified STE compliance.'))
    print(json.dumps(dict(primary_runs=len(checked),figures=len(figures),hard_style_findings=0,passed=True)))


if __name__=='__main__':main()
