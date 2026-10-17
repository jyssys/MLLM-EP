"""Reuse the scoped burner guard and its pending-restore protection."""
import sys
import zipfile
from poc_async_recurrent_pp.contract import *
from poc_looped_multigpu_kv_placement.scripts import gpu_session as shared
from poc_ouro_reasoning_fidelity_multigpu import guard_cleanup


def main():
    shared.ROOT, shared.PROJECT, shared.ASSETS = ROOT, PROJECT, ASSETS
    guard_cleanup.ASSETS = ASSETS
    requested = list(map(int, sys.argv[sys.argv.index('--gpus')+1].split(',')))
    assert set(requested) <= set(GPUS)
    output = (ROOT/sys.argv[sys.argv.index('--output')+1]).resolve()
    assert output.is_relative_to(ROOT/'outputs') and not output.exists()
    expected = sys.argv[sys.argv.index('--')+1:]
    groups = []
    prior_write=shared.write
    def snapshot(path,value):
        prior_write(path,value)
        if path.name=='run_contract.json':
            with zipfile.ZipFile(path.parent/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
                for source in sorted(ROOT.rglob('*.py')):
                    archive.write(source,arcname=str(source.relative_to(ROOT)))
    shared.write=snapshot
    original = shared.subprocess.Popen
    shared.subprocess.Popen = guard_cleanup.registered_launch(original, expected, output, groups)
    try:
        shared.main()
    finally:
        shared.subprocess.Popen = original
        if output.exists():
            guard_cleanup.finish(groups, requested, output, shared.contexts)


if __name__ == '__main__':
    main()
