"""Snapshot-only sitecustomize: redirect inherited workers to new evidence."""
import os

if os.environ.get('DUETSERVE_EVIDENCE_BOOTSTRAP') == '1':
    from poc_flashvep.duetserve_agg_decode.common import bind_previous
    bind_previous()
    if os.environ.get('DENSE_CONFIG'):
        import json
        import sys
        config = json.loads(open(os.environ['DENSE_CONFIG']).read())
        if config.get('nsys_bounded_capture') and '--role' not in sys.argv:
            from poc_flashvep.duetserve_agg_decode.ownership_watch import install
            install()
    from poc_flashvep.duetserve_agg_decode.iteration_trace import bootstrap
    bootstrap()
