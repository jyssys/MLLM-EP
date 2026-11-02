"""Rebuild derived figures from frozen summaries before owned publication."""
import argparse
import signal
from .common import *
from .guard import processes


def unfinished(pid):
    path=Path('/proc')/str(pid)
    try:return path.exists() and (path/'stat').read_text().split()[2]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=int,required=True)
    parser.add_argument('--archive-child',type=int,required=True);args=parser.parse_args()
    parent=Path('/proc')/str(args.parent)
    assert parent.stat().st_uid==os.getuid()
    assert b'poc_flashvep.looped_pd_atomic_prefill.closeout' in (parent/'cmdline').read_bytes().split(b'\0')
    marker=ROOT/'state/final_figure_review_20261102.json';result=dict(time=time.time(),status='DERIVED_REVIEW_IN_PROGRESS',parent=args.parent)
    os.kill(args.parent,signal.SIGSTOP)
    try:
        assert not processes('publish'),'Do not change derived artifacts during an active publication'
        write(marker,result)
        deadline=time.monotonic()+600
        while unfinished(args.archive_child):
            assert time.monotonic()<deadline,'Archive-child wait exceeded its bounded review limit'
            time.sleep(1)
        # Do not rewrite archived large CSV files. The GPU search is terminal.
        from . import analyze,figures,reporting
        analyze.serving=lambda:json.loads((ROOT/'serving_runs.json').read_text())
        analyze.capacity=lambda:json.loads((ROOT/'capacity_summary.json').read_text())
        figures.serving=analyze.serving
        figures.main();reporting.main()
        result.update(status='DERIVED_REVIEW_COMPLETE',completed_time=time.time(),
            scope='The cached measured summaries are unchanged. Single-rate bootstrap intervals now use visible vertical bars. No GPU experiment or acceptance threshold changed.')
        write(marker,result)
    except BaseException as error:
        import traceback
        result.update(status='BLOCKED_DERIVED_REVIEW',error=repr(error),traceback=traceback.format_exc());write(marker,result)
        raise
    finally:
        os.kill(args.parent,signal.SIGCONT)


if __name__=='__main__':main()
