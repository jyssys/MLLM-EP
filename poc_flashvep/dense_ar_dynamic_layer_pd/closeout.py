"""Close GPU work and verify the approved all-device burner state."""
import argparse,ast,fcntl,io,re,unittest
from .common import *


def close():
    preserve()
    launchers=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal() or int(proc.name)==os.getpid():continue
        try:
            words=(proc/'cmdline').read_bytes().split(b'\0')
            if proc.stat().st_uid==os.getuid() and b'-m' in words:
                i=words.index(b'-m')
                if words[i+1].startswith(b'poc_flashvep.dense_ar_dynamic_layer_pd.'):
                    launchers.append(dict(pid=int(proc.name),module=words[i+1].decode()))
        except (OSError,IndexError):pass
    assert not launchers,('An own campaign launcher remains active',launchers)
    active=[]
    for path in (ROOT/'raw').iterdir():
        if path.is_dir() and not (path/'dispatcher_result.json').exists():active.append(path.name)
    assert not active,('No unreceipted GPU namespace may be hidden at closeout',active)
    with QUIET_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);owner=resources()
        before=owner.processes();burners=owner.burners()
        assert all(os.getsid(r['pid'])==burners.get(r['gpu']) for r in before),'An experiment or unrelated GPU process remains'
        write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=time.time(),reason='All planned GPU stages reached a recorded terminal status. New GPU starts are prohibited.'))
        owner.restore(list(range(8)))
    log_path=ROOT/'burner.log';log_offset=log_path.stat().st_size if log_path.exists() else 0
    deadline=time.monotonic()+420;latest={}
    pattern=re.compile(r'\[GPU (\d+)\] rounds=(\d+) generated_tokens=\d+')
    while time.monotonic()<deadline:
        text=log_path.read_bytes()[log_offset:].decode(errors='replace')
        latest={}
        for gpu,generation in pattern.findall(text):latest[int(gpu)]=int(generation)
        owner=resources();actual=owner.processes();owned=owner.burners()
        active_gpus={r['gpu'] for r in actual if os.getsid(r['pid'])==owned.get(r['gpu'])}
        if active_gpus==set(range(8)) and all(g in latest for g in range(8)):break
        # Retry only missing, unoccupied GPUs. Never start a duplicate worker.
        if len(owned)<8:owner.restore(list(range(8)))
        time.sleep(2)
    assert active_gpus==set(range(8)),dict(actual=actual,owned=owned)
    baseline=dict(latest);deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        text=log_path.read_bytes()[log_offset:].decode(errors='replace')
        for gpu,generation in pattern.findall(text):latest[int(gpu)]=int(generation)
        if all(latest.get(g,-1)>baseline.get(g,-1) for g in range(8)):break
        time.sleep(2)
    assert all(latest.get(g,-1)>baseline.get(g,-1) for g in range(8)),'Require fresh generation on every GPU'
    snapshot=ROOT/'state/burner_closeout.log';snapshot.write_text((ROOT/'burner.log').read_text(errors='replace'))
    write(ROOT/'state/resource_restoration_verified.json',dict(time=time.time(),owned_burner_sessions=owner.burners(),
        actual_GPU_processes=owner.processes(),baseline_generation=baseline,fresh_generation=latest,
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv'],text=True),
        passed=True,scope='Fresh completed generation and verified own process sessions on all8 GPUs. NVML utilization is a resource check, not kernel idle metrology.'))
    preserve();print('ALL_EIGHT_BURNERS_VERIFIED')


def verify():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists();preserve()
    suite=unittest.defaultTestLoader.discover(str(SOURCE),top_level_dir=str(REPO))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
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
    write(ROOT/'state/final_verification.json',dict(time=time.time(),tests_run=result.testsRun,tests_passed=result.wasSuccessful(),
        test_output=stream.getvalue(),source_AST_checks=parsed,missing_local_links=missing,preservation=preservation(),
        frozen_SLO_unchanged=True,frozen_predecessor_files_unchanged=True,installed_runtime_unchanged=True,
        passed=result.wasSuccessful() and not missing,scope='CPU and artifact checks do not substitute for separate GPU correctness and performance gates.'))
    print(stream.getvalue());assert result.wasSuccessful() and not missing


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['close','verify']);a=p.parse_args()
    (close if a.action=='close' else verify)()


if __name__=='__main__':main()
