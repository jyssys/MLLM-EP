"""Install the frozen numerical path before campaign-specific validation hooks."""
import os
if os.environ.get('FLUIDTOKEN_LOOP_WORKER') == '1':
    try:
        import runpy
        from poc_flashvep.native_split_deep_validation.common import configure, REPO
        configure()
        runpy.run_path(str(REPO / 'poc_flashvep/native_loop_2d_oracle/bootstrap/sitecustomize.py'))
        from poc_flashvep.looped_pd_atomic_prefill.common import config
        from poc_flashvep.native_split_deep_validation.scheduler_contract import install
        install(config())
        from poc_flashvep.native_split_deep_validation.exact_fingerprint import install as fingerprint
        fingerprint(config())
        if config().get('deep_graph_policy'):
            from poc_flashvep.native_split_deep_validation.graph_policy import install as graph_policy
            graph_policy(config())
        from poc_flashvep.native_split_deep_validation.wire_accounting import install as wire_accounting
        wire_accounting(config())
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
