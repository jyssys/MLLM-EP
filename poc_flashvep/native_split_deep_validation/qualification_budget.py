"""End an equal-budget qualification round only at an idle controller boundary."""
import signal
from .common import *


def main():
    initialize()
    declaration = ROOT / 'state/equal_candidate_budget_amendment.json'
    if not declaration.exists():
        assert not list((ROOT / 'raw').glob('c_*')), 'Declare the budget before any clean cut selection'
        write(declaration, dict(time=time.time(), candidates_per_axis=2,
            mandatory_methods_per_model=8,
            evidence='The first cache4 Huginn Native N8/N32 dispatches took 287.774 and approximately 614 seconds. They are heavy state diagnostics, not serving latency.',
            reason='Repeating the expanded cache qualification and every third candidate risks displacing the required three-start confirmation in both models. Keep two candidates per axis uniformly before any clean selection.',
            omitted_third_candidates=dict(huginn=['token1792', 'loop24', '2d_c31_k1792'],
                ouro=['token4032', 'loop1', '2d_c3_k3840']),
            scope='BUDGET_REDUCED_EQUAL_CANDIDATE_SET. No correctness or SLO threshold changes. Omitted candidates are NOT_RUN_TIME_BUDGET, not performance failures.'))
    owners = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            words = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'poc_flashvep.native_split_deep_validation.campaign' in words:
                pid = int(proc.name)
                assert proc.stat().st_uid == os.getuid() and os.getsid(pid) == pid
                owners.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            pass
    assert len(owners) == 1, owners
    controller = owners[0]
    path = ROOT / 'state/qualification.json'
    while time.time() < LONG_START_CUTOFF:
        if (ROOT / 'state/qualification_exception.json').exists():
            raise RuntimeError('Qualification has an exception. Do not synthesize a terminal gate.')
        value = json.loads(path.read_text())
        if value['status'] == 'QUALIFICATION_FINISHED':
            return
        counts = [len(value['models'][model]['qualification']) for model in ('ouro', 'huginn')]
        done = all(len(row.get('checks', [])) >= 2 or
                   (row.get('checks') and not row['checks'][-1]['passed'])
                   for model in ('ouro', 'huginn') for row in value['models'][model]['qualification'].values())
        active = json.loads((ROOT / 'state/active.json').read_text())
        if min(counts) >= 8 and done and not active.get('name'):
            # The dispatcher already cleaned its owned GPU session. Only the
            # CPU controller is signaled before it can start the third round.
            proc = Path('/proc', str(controller))
            assert proc.stat().st_uid == os.getuid() and os.getsid(controller) == controller
            assert b'poc_flashvep.native_split_deep_validation.campaign' in (proc / 'cmdline').read_bytes().split(b'\0')
            os.kill(controller, signal.SIGTERM)
            write(ROOT / 'state/qualification_before_budget_finish.json', value)
            value.update(status='QUALIFICATION_FINISHED', equal_candidate_budget=2,
                third_candidates='NOT_RUN_TIME_BUDGET', GPU_worker_signaled=False)
            write(path, value)
            write(ROOT / 'state/qualification_budget_finish.json', dict(time=time.time(), controller=controller,
                GPU_worker_signaled=False, methods_per_model=counts,
                scope='All completed raw gates remain unchanged. Only the next unstarted equal candidate round is omitted.'))
            print('Equal two-candidate qualification round finished', flush=True)
            return
        time.sleep(.02)


if __name__ == '__main__':
    main()
