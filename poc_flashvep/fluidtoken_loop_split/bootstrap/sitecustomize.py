"""Redirect inherited hooks before any worker imports campaign globals."""
import os
if os.environ.get('FLUIDTOKEN_LOOP_WORKER') == '1':
    try:
        from poc_flashvep.fluidtoken_loop_split.common import configure
        configure()
        if os.environ.get('FLUIDTOKEN_DEBUGGER_OWNER_PID'):
            import ctypes
            owner = int(os.environ['FLUIDTOKEN_DEBUGGER_OWNER_PID'])
            # Authorize this runner and its diagnostic children, not all processes.
            result = ctypes.CDLL(None, use_errno=True).prctl(0x59616d61, owner, 0, 0, 0)
            if result:
                raise OSError(ctypes.get_errno(), 'Cannot authorize the owned diagnostic runner')
        import runpy
        import json
        from pathlib import Path
        from poc_flashvep.fluidtoken_loop_split.collector_buffers import prepare as buffers
        buffers(json.loads(Path(os.environ['ARC_CONFIG']).read_text()))
        from poc_flashvep.fluidtoken_loop_split.probe_admission import prepare
        prepare(json.loads(Path(os.environ['ARC_CONFIG']).read_text()))
        runpy.run_path('/home/esjung/MLLM-EP-github/poc_flashvep/looped_pd_autonomous_headroom/bootstrap/sitecustomize.py')
        from poc_flashvep.looped_pd_atomic_prefill.common import config
        from poc_flashvep.fluidtoken_loop_split.split_adapter import install
        install(config())
        from poc_flashvep.fluidtoken_loop_split.token_suffix_origin import install as token_suffix_origin
        token_suffix_origin(config())
        from poc_flashvep.fluidtoken_loop_split.partition_accounting import install as partition_accounting
        partition_accounting(config())
        from poc_flashvep.fluidtoken_loop_split.nixl_control import install as nixl_control
        nixl_control(config())
        from poc_flashvep.fluidtoken_loop_split.activity_capture import install as activity_capture
        activity_capture(config())
        from poc_flashvep.fluidtoken_loop_split.device_execution import install as device_execution
        device_execution(config())
        from poc_flashvep.fluidtoken_loop_split.graph_coverage import install as graph_coverage
        graph_coverage(config())
        from poc_flashvep.fluidtoken_loop_split.probe_quantum import install as probe_quantum
        probe_quantum(config())
        from poc_flashvep.fluidtoken_loop_split.gc_observer import install as gc_observer
        gc_observer(config())
        from poc_flashvep.fluidtoken_loop_split.evidence_routing import install as evidence_routing
        evidence_routing(config())
        from poc_flashvep.fluidtoken_loop_split.host_pause_observer import install as host_pause_observer
        host_pause_observer(config())
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
