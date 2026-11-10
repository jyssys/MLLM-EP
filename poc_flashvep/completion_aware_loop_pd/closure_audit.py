"""Check source, preservation, run inventory, and report closure before publication."""
import subprocess
import sys
from .common import *
from .publish import preserved,git
from .serving_audit import csv_write


def main():
    configure();preserved()
    runs=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.is_dir():continue
        cfgpath=ROOT/'configs'/(folder.name+'.json')
        cfg=json.loads(cfgpath.read_text()) if cfgpath.exists() else {}
        receipt=folder/'dispatcher_result.json'
        record=json.loads(receipt.read_text()) if receipt.exists() else dict(status='NO_TERMINAL_RECEIPT')
        mapping=folder/'worker_map.json'
        source=folder/'new_campaign_source.zip'
        runs.append(dict(name=folder.name,model=cfg.get('model'),phase=cfg.get('phase'),
            count=cfg.get('count'),arrival_seed=cfg.get('arrival_seed'),
            configuration_sha256=digest(cfgpath) if cfgpath.exists() else None,
            worker_mapping=json.loads(mapping.read_text()) if mapping.exists() else None,
            source_archive_sha256=digest(source) if source.exists() else None,
            receipt=record))
    write(ROOT/'run_inventory.json',runs);csv_write(ROOT/'run_inventory.csv',runs)
    configs=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        value=json.loads(path.read_text())
        if 'name' not in value:continue
        receipt=ROOT/'raw'/value['name']/'dispatcher_result.json'
        configs.append(dict(name=value['name'],configuration_sha256=digest(path),
            model=value.get('model'),phase=value.get('phase'),count=value.get('count'),
            arrival_seed=value.get('arrival_seed'),
            status=json.loads(receipt.read_text()).get('status') if receipt.exists() else 'UNEXECUTED_CONFIGURATION',
            scope='A prepared configuration is not a completed experiment. Superseded and unselected configurations remain preserved.'))
    write(ROOT/'configuration_inventory.json',configs);csv_write(ROOT/'configuration_inventory.csv',configs)
    command=[sys.executable,'-m','unittest','discover','-s','poc_flashvep/completion_aware_loop_pd','-t','.',
        '-p','test_*.py','-v']
    tests=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
    inherited=subprocess.run([sys.executable,'-m','unittest','poc_flashvep.idle_intervention_dynamic_loop.test_protocol','-v'],
        cwd=REPO,capture_output=True,text=True)
    compiled=subprocess.run([sys.executable,'-m','compileall','-q',str(SOURCE)],cwd=REPO,capture_output=True,text=True)
    result=dict(time=time.time(),unit_tests=dict(command=command,returncode=tests.returncode,
        output=tests.stdout+tests.stderr),inherited_protocol_tests=dict(returncode=inherited.returncode,
        output=inherited.stdout+inherited.stderr),compile=dict(returncode=compiled.returncode,
        output=compiled.stdout+compiled.stderr),active_branch=git('branch','--show-current'),
        preservation_passed=True,source_hashes={str(p.relative_to(REPO)):digest(p) for p in SOURCE.rglob('*')
            if p.is_file() and p.suffix in ('.py','.cc')},
        source_scope='The final source hash list differs from earlier per-run snapshots. Each run archives its actual implementation.')
    write(ROOT/'state/final_source_verification.json',result)
    assert tests.returncode==inherited.returncode==compiled.returncode==0
    print(json.dumps(dict(runs=len(runs),new_tests_pass=True,inherited_tests_pass=True,compile_pass=True,preservation_pass=True)))


if __name__=='__main__':
    main()
