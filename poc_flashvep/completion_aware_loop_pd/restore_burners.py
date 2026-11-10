"""Restore approved workers and check actual generation on every GPU."""
import contextlib
import io
from .common import *


def main():
    configure()
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists(),'Close scientific work before final restoration'
    from ..looped_pd_autonomous_headroom.common import resources
    from ..idle_intervention_dynamic_loop import check_burners
    check_burners.ROOT=ROOT;check_burners.configure=configure
    owner=resources();owner.restore(list(range(8)))
    started=time.monotonic();retry_done=False
    while time.monotonic()-started<240:
        attempt_started=time.time()
        with contextlib.redirect_stdout(io.StringIO()):check_burners.main()
        final=ROOT/'state/final_burner_verification.json'
        result=json.loads(final.read_text()) if final.exists() else {}
        if result.get('time',0)>=attempt_started and result.get('all_eight_verified'):
            print('ALL_EIGHT_BURNERS_VERIFIED',flush=True);return
        pending=json.loads((ROOT/'state/pending_burner_verification.json').read_text())
        print(json.dumps(dict(Burner_verification='STARTUP_PENDING',
            wrappers=len(pending['wrappers']),resident_GPUs=sum(bool(v) for v in pending['engines'].values()),
            elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
        if time.monotonic()-started>150 and not retry_done:
            # A healthy worker remains untouched. Retry only an approved
            # wrapper without any resident engine after the startup allowance.
            missing=[g for g in range(8) if not pending['engines'].get(str(g),pending['engines'].get(g))]
            wrappers=owner.burners()
            owned_missing=[g for g in missing if g in wrappers]
            if owned_missing:owner.stop(owned_missing)
            if missing:owner.restore(missing)
            retry_done=True
        time.sleep(5)
    raise RuntimeError('Final Burner generation verification did not pass on all eight GPUs')


if __name__=='__main__':
    main()
