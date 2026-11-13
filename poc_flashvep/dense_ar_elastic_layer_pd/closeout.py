"""Close GPU work and verify fresh generation on every approved burner."""
import argparse,ast,fcntl,io,re,unittest
from .common import *


def close():
    preserve();active=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal() or int(proc.name)==os.getpid():continue
        try:
            words=(proc/'cmdline').read_bytes().split(b'\0')
            if proc.stat().st_uid==os.getuid() and b'-m' in words:
                index=words.index(b'-m')
                if words[index+1].startswith(b'poc_flashvep.dense_ar_elastic_layer_pd.'):
                    active.append(dict(pid=int(proc.name),module=words[index+1].decode()))
        except (OSError,IndexError):pass
    assert not active,('An own campaign launcher remains active',active)
    missing=[p.name for p in (ROOT/'raw').iterdir() if p.is_dir() and (p/'command.json').exists() and not (p/'dispatcher_result.json').exists()]
    assert not missing,('An executed GPU job has no terminal receipt',missing)
    with QUIET_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);owner=resources();before=owner.processes();owned=owner.burners()
        assert all(os.getsid(r['pid'])==owned.get(r['gpu']) for r in before),'An experiment or unrelated GPU process remains'
        write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=time.time(),
            reason='Every executed job has a terminal receipt. New GPU starts are prohibited.'))
        log=ROOT/'burner.log';offset=log.stat().st_size if log.exists() else 0
        owner.restore(list(range(8)))
    pattern=re.compile(r'\[GPU (\d+)\] rounds=(\d+) generated_tokens=\d+')
    deadline=time.monotonic()+420;latest={}
    while time.monotonic()<deadline:
        for gpu,generation in pattern.findall(log.read_bytes()[offset:].decode(errors='replace')):latest[int(gpu)]=int(generation)
        owner=resources();owned=owner.burners();actual=owner.processes()
        gpu_set={r['gpu'] for r in actual if os.getsid(r['pid'])==owned.get(r['gpu'])}
        if gpu_set==set(range(8)) and set(latest)==set(range(8)):break
        # Restore only missing, unoccupied GPUs. This retains the GPU0 retry.
        if len(owned)<8:owner.restore(list(range(8)))
        time.sleep(2)
    assert gpu_set==set(range(8)) and set(latest)==set(range(8)),dict(actual=actual,owned=owned,rounds=latest)
    baseline=dict(latest);deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        for gpu,generation in pattern.findall(log.read_bytes()[offset:].decode(errors='replace')):latest[int(gpu)]=int(generation)
        if all(latest[g]>baseline[g] for g in range(8)):break
        time.sleep(2)
    assert all(latest[g]>baseline[g] for g in range(8)),'Require fresh completed generation on every GPU'
    (ROOT/'state/burner_closeout.log').write_bytes(log.read_bytes())
    write(ROOT/'state/resource_restoration_verified.json',dict(time=time.time(),passed=True,
        owned_burner_sessions=owner.burners(),actual_GPU_processes=owner.processes(),
        baseline_generation=baseline,fresh_generation=latest,
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv'],text=True),
        scope='Fresh completed generation and verified own sessions on all8 GPUs. NVML is a restoration check, not physical idle metrology.'))
    preserve();print('ALL_EIGHT_BURNERS_VERIFIED',flush=True)


def verify():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists();preserve()
    suite=unittest.defaultTestLoader.discover(str(SOURCE),top_level_dir=str(REPO));stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    parsed=[]
    for path in sorted(SOURCE.glob('*.py')):
        ast.parse(path.read_text(),filename=str(path));parsed.append(dict(path=str(path.relative_to(REPO)),SHA256=digest(path)))
    for path,expected in json.loads((ROOT/'state/frozen_predecessor_hashes.json').read_text()).items():assert digest(REPO/path)==expected
    for path,expected in json.loads((ROOT/'state/installed_runtime_sources.json').read_text())['hashes'].items():assert digest(path)==expected
    assert digest(ROOT/'configs/frozen_SLO_regimes.json')=='c8aa94213e4f4ce54821d01ba985a8e9e88a20566a3e8d1f3cc8060d1b3926d9'
    from .publish import DOCUMENTS
    missing=[]
    for name in DOCUMENTS:
        path=REPO/'poc_flashvep/reports'/name;assert path.exists(),name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):continue
            if not (path.parent/target.split('#')[0]).exists():missing.append(dict(document=name,target=target))
    write(ROOT/'state/final_verification.json',dict(time=time.time(),tests_run=result.testsRun,
        tests_passed=result.wasSuccessful(),test_output=stream.getvalue(),source_AST_checks=parsed,
        missing_local_links=missing,preservation=preservation(),frozen_SLO_unchanged=True,
        frozen_predecessor_files_unchanged=True,installed_runtime_unchanged=True,
        passed=result.wasSuccessful() and not missing))
    print(stream.getvalue());assert result.wasSuccessful() and not missing


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['close','verify'])
    (close if parser.parse_args().action=='close' else verify)()
