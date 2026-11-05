"""Reuse qualified hooks before the isolated native continuation candidate."""
import os
if os.environ.get('FLUIDTOKEN_LOOP_WORKER') == '1':
    try:
        import runpy
        from poc_flashvep.native_loop_2d_oracle.common import configure, REPO
        configure()
        runpy.run_path(str(REPO / 'poc_flashvep/dynamic_loop_split_serving/bootstrap/sitecustomize.py'))
        from poc_flashvep.looped_pd_atomic_prefill.common import config
        from poc_flashvep.native_loop_2d_oracle.native_path import install
        install(config())
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
