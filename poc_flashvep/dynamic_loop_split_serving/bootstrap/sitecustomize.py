"""Install parent hooks with redirected globals before worker imports."""
import os
if os.environ.get('FLUIDTOKEN_LOOP_WORKER') == '1':
    try:
        import runpy
        from poc_flashvep.dynamic_loop_split_serving.common import configure, PARENT_SOURCE
        configure()
        import json
        from pathlib import Path
        contract = json.loads(Path(os.environ['ARC_CONFIG']).read_text())
        from poc_flashvep.dynamic_loop_split_serving.dynamic_recovery import prepare, finalize
        prepare(contract)
        runpy.run_path(str(PARENT_SOURCE / 'bootstrap/sitecustomize.py'))
        finalize(contract)
        from poc_flashvep.looped_pd_atomic_prefill.common import config
        from poc_flashvep.dynamic_loop_split_serving.warmup_boundary import install as warmup_boundary
        warmup_boundary(config())
        from poc_flashvep.dynamic_loop_split_serving.service_timing import install as service_timing
        service_timing(config())
        from poc_flashvep.dynamic_loop_split_serving.runtime_contract import install as runtime_contract
        runtime_contract(config())
        from poc_flashvep.dynamic_loop_split_serving.loop_burst import install as loop_burst
        loop_burst(config())
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
