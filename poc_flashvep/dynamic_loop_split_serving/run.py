"""Reuse the existing exclusive resource guard and bounded worker cleanup."""
import argparse
import fcntl
import zipfile
from .common import *


class BudgetBeforeLaunch(RuntimeError):
    pass


def fits_launch_budget(cfg, now):
    return (now < LONG_START_CUTOFF and now < GPU_DEADLINE - 180
            and now + cfg.get('timeout', 300) + cfg.get('startup_timeout', 240) < GPU_DEADLINE)


def execute(name):
    initialize()
    assert name.replace('_', '').isalnum(), 'A run name must be a simple immutable namespace'
    # Controllers can request the same predeclared cell. A shared namespace
    # lock prevents a second controller from treating a live run as abandoned.
    with (ROOT / 'state' / (name + '.namespace.lock')).open('a+') as namespace:
        fcntl.flock(namespace, fcntl.LOCK_EX)
        folder = ROOT / 'raw' / name
        receipt = folder / 'dispatcher_result.json'
        if receipt.exists():
            return bool(json.loads(receipt.read_text()).get('complete'))
        if folder.exists() and not (folder / 'dispatcher_result.json').exists():
            # Older controllers predate the namespace lock. Their GPU work and
            # finalization still hold the existing quiet lane. Wait for it.
            with (ROOT / 'state/quiet_lane.lock').open('a+') as lane:
                fcntl.flock(lane, fcntl.LOCK_EX)
                if not (folder / 'dispatcher_result.json').exists():
                    raise RuntimeError('Unfinished namespace requires evidence review: ' + name)
        try:
            return _execute_locked(name)
        except RuntimeError as error:
            # A legacy controller can create this namespace after our first
            # check, without taking the newer namespace lock. Its quiet lane
            # still protects the entire GPU run and receipt finalization.
            legacy_errors = {
                'Unfinished namespace requires explicit evidence review: ' + name,
                'Namespace became unfinished while waiting for the lane: ' + name,
            }
            if str(error) not in legacy_errors or not folder.exists():
                raise
            with (ROOT / 'state/quiet_lane.lock').open('a+') as lane:
                fcntl.flock(lane, fcntl.LOCK_EX)
                if receipt.exists():
                    return bool(json.loads(receipt.read_text()).get('complete'))
            raise


def _execute_locked(name):
    initialize()
    from ..looped_pd_autonomous_headroom import run as native
    original = native.environment
    def environment(gpus):
        cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
        if not fits_launch_budget(cfg, time.time()):
            raise BudgetBeforeLaunch('The queued job no longer fits the fixed launch budget')
        env = original(gpus)
        folder = ROOT / 'raw' / name
        parent_paths = sorted(list(PARENT_SOURCE.rglob('*.py')) + list(PARENT_SOURCE.rglob('*.cc')))
        with zipfile.ZipFile(folder / 'parent_campaign_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in parent_paths:
                archive.write(path, str(path.relative_to(REPO)))
        write(folder / 'parent_campaign_source_hashes.json',
              {str(path): digest(path) for path in parent_paths})
        return env
    native.environment = environment
    try:
        from ..fluidtoken_loop_split import run as inherited
        return inherited.execute(name)
    except BudgetBeforeLaunch as error:
        write(ROOT / 'raw' / name / 'dispatcher_result.json', dict(returncode=None,
            complete=False, status='NOT_RUN_TIME_BUDGET', reason=str(error), time=time.time(),
            scope='Rechecked after acquiring the quiet lane and before launching a worker. No GPU model execution started.'))
        return False
    finally:
        native.environment = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    raise SystemExit(0 if execute(args.name) else 1)
