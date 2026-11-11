import os
if os.environ.get('LOOP_WIDTH_SNAPSHOT') and os.environ.get('FLUIDTOKEN_LOOP_WORKER')=='1':
    try:
        import importlib.util,json,runpy,sys
        from pathlib import Path
        snapshot=Path(os.environ['LOOP_WIDTH_SNAPSHOT']);key='poc_flashvep.loop_width_gemm_validation'
        package=snapshot/'poc_flashvep/loop_width_gemm_validation'
        spec=importlib.util.spec_from_file_location(key,package/'__init__.py',submodule_search_locations=[str(package)])
        module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
        from poc_flashvep.loop_width_gemm_validation.common import configure,ROOT,digest
        configure();cfg=json.loads(Path(os.environ['ARC_CONFIG']).read_text())
        hashes=json.loads((ROOT/'raw'/cfg['name']/'width_source_hashes.json').read_text())
        assert all(digest(snapshot/name)==value for name,value in hashes.items())
        runpy.run_path(str(snapshot/'poc_flashvep/huginn_net_benefit/bootstrap/sitecustomize.py'))
        from poc_flashvep.loop_width_gemm_validation.kernel_runtime import install
        install(cfg)
        from poc_flashvep.loop_width_gemm_validation.core_timing import install as core_timing
        core_timing(cfg)
        from poc_flashvep.loop_width_gemm_validation.core_graph_timing import install as core_graph_timing
        core_graph_timing(cfg)
    except BaseException:
        import traceback
        traceback.print_exc();os._exit(70)
