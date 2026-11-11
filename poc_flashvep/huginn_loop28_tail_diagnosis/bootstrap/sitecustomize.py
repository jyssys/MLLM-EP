import os
if os.environ.get('LOOP28_DIAG_SNAPSHOT') and os.environ.get('FLUIDTOKEN_LOOP_WORKER')=='1':
    try:
        import importlib.util,json,runpy,sys
        from pathlib import Path
        snapshot=Path(os.environ['LOOP28_DIAG_SNAPSHOT'])
        name='poc_flashvep.huginn_loop28_tail_diagnosis'
        p=snapshot/'poc_flashvep/huginn_loop28_tail_diagnosis'
        spec=importlib.util.spec_from_file_location(name,p/'__init__.py',submodule_search_locations=[str(p)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        from poc_flashvep.huginn_loop28_tail_diagnosis.common import configure,ROOT,digest
        configure()
        cfg=json.loads(Path(os.environ['ARC_CONFIG']).read_text())
        hashes=json.loads((ROOT/'raw'/cfg['name']/'diagnosis_source_hashes.json').read_text())
        assert all(digest(snapshot/path)==value for path,value in hashes.items())
        runpy.run_path(str(snapshot/'poc_flashvep/huginn_net_benefit/bootstrap/sitecustomize.py'))
        if cfg.get('diagnosis_events'):
            from poc_flashvep.huginn_loop28_tail_diagnosis.observer import install
            install(cfg)
        if cfg.get('output_idle_capture'):
            from poc_flashvep.huginn_loop28_tail_diagnosis.output_idle import install
            install(cfg)
        if cfg.get('drain_forecast_audit'):
            from poc_flashvep.huginn_loop28_tail_diagnosis.drain_audit import install
            install(cfg)
    except BaseException:
        import traceback
        traceback.print_exc();os._exit(70)
