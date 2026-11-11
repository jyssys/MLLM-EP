import os
if os.environ.get('PREFILL_DISCOVERY_SNAPSHOT') and os.environ.get('FLUIDTOKEN_LOOP_WORKER')=='1':
    try:
        import importlib.util,json,runpy,sys
        from pathlib import Path
        snapshot=Path(os.environ['PREFILL_DISCOVERY_SNAPSHOT'])
        for name in ('prefill_heavy_decode_safe','loop_width_gemm_validation'):
            key='poc_flashvep.'+name;package=snapshot/'poc_flashvep'/name
            spec=importlib.util.spec_from_file_location(key,package/'__init__.py',submodule_search_locations=[str(package)])
            module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
        from poc_flashvep.prefill_heavy_decode_safe.common import configure,ROOT,digest,write
        configure();cfg=json.loads(Path(os.environ['ARC_CONFIG']).read_text())
        hashes=json.loads((ROOT/'raw'/cfg['name']/'discovery_source_hashes.json').read_text())
        assert all(digest(snapshot/path)==value for path,value in hashes.items())
        runpy.run_path(str(snapshot/'poc_flashvep/huginn_net_benefit/bootstrap/sitecustomize.py'))
        from poc_flashvep.loop_width_gemm_validation import kernel_runtime,lt
        kernel_runtime.ROOT=ROOT;lt.ROOT=ROOT
        kernel_runtime.install(cfg)
        if cfg.get('priority_serial_comparison') and cfg.get('priority_inner_core_events',True):
            from poc_flashvep.loop_width_gemm_validation import common as width_common
            width_common.ROOT=ROOT
            from poc_flashvep.loop_width_gemm_validation.core_timing import install as core_timing
            from poc_flashvep.loop_width_gemm_validation.core_graph_timing import install as core_graph_timing
            core_timing(cfg);core_graph_timing(cfg)
        from poc_flashvep.prefill_heavy_decode_safe.activity import install
        install(cfg)
        from poc_flashvep.prefill_heavy_decode_safe.observation import install as observe
        observe(cfg)
        from poc_flashvep.looped_pd_atomic_prefill import common as atomic
        assert atomic.ROOT==ROOT,('Evidence namespace did not propagate',atomic.ROOT,ROOT)
        write(ROOT/'raw'/cfg['name']/f'discovery_loaded_{os.getpid()}.json',dict(
            configuration_SHA256=digest(os.environ['ARC_CONFIG']),snapshot=str(snapshot),ROOT=str(ROOT),
            projection_policy_SHA256=cfg.get('projection_kernel_policy_SHA256'),
            LD_PRELOAD=os.environ.get('LD_PRELOAD'),
            CUPTI_mappings=[line.strip() for line in open('/proc/self/maps') if 'libcupti' in line]))
    except BaseException:
        import traceback
        traceback.print_exc();os._exit(70)
