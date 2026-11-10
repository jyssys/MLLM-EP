"""Load the qualified runtime before the new process-local scheduler hooks."""
import os
if os.environ.get('FLUIDTOKEN_LOOP_WORKER')=='1':
    try:
        import json
        import runpy
        import sys
        import importlib.util
        from pathlib import Path
        snapshot=os.environ.get('INTERVENTION_SNAPSHOT')
        if snapshot:
            import poc_flashvep
            package=Path(snapshot)/'poc_flashvep/completion_aware_loop_pd'
            key='poc_flashvep.completion_aware_loop_pd'
            spec=importlib.util.spec_from_file_location(key,package/'__init__.py',submodule_search_locations=[str(package)])
            module=importlib.util.module_from_spec(spec);sys.modules[key]=module;spec.loader.exec_module(module)
        from poc_flashvep.completion_aware_loop_pd.common import configure,REPO,ROOT,digest,write
        configure()
        cfg=json.loads(Path(os.environ['ARC_CONFIG']).read_text())
        legacy_manifest=ROOT/'raw'/cfg['name']/'reused_idle_source_hashes.json'
        if legacy_manifest.exists():
            assert all(digest(path)==value for path,value in json.loads(legacy_manifest.read_text()).items()),'Reused frozen idle source changed'
        if cfg.get('qualification_ignore_observer_tail_latch'):
            assert cfg.get('state_probe') and cfg.get('async_prompt_state_probe')
            assert not cfg.get('clean_confirmation'),'A clean run must retain the Decode tail latch'
        assert not cfg.get('dlocal_campaign') and not cfg.get('D_local_probe_output_pacing')
        hashes=json.loads((ROOT/'raw'/cfg['name']/'new_campaign_source_hashes.json').read_text())
        assert all(digest(Path(snapshot)/Path(path).relative_to(REPO) if snapshot else path)==value
                   for path,value in hashes.items()),'Frozen worker source changed before startup'
        write(ROOT/'raw'/cfg['name']/f'loaded_intervention_source_{os.getpid()}.json',dict(hashes=hashes))
        if cfg.get('intervention_method','native')!='native':
            from poc_flashvep.native_split_deep_validation import graph_policy
            geometry=graph_policy.preferred_geometry
            def preferred(config,role):
                if role=='decode':
                    widths=config.get('D_candidate_widths',[config.get('D_tile_tokens',config['length'])])
                    return {(q*b,q) for q in widths for b in range(1,config['D_slots']+1)
                        if q*b<=config['D_prefill_graph_rows']}
                if config.get('intervention_method')=='dynamic_token':
                    points=[0]+config['boundary_checkpoints']+[config['length']]
                    # A rejection can skip a checkpoint. Include every legal
                    # prefix and delta, not only adjacent small suffixes.
                    widths={b-a for a in points for b in points if b>a}
                    allowed={(q,q) for q in widths if q<=config['P_prefill_graph_rows']}
                    if config['P_slots']>=2:
                        allowed.update((a+b,max(a,b)) for a in widths for b in widths
                            if a+b<=config['P_prefill_graph_rows'])
                    return allowed
                return geometry(config,role)
            graph_policy.preferred_geometry=preferred
        if cfg.get('deferred_prompt_state_probe') or cfg.get('async_prompt_state_probe'):
            assert cfg['state_probe'] and not cfg['verify_transfer']
            from poc_flashvep.looped_pd_headroom_recovery import state_probe
            state_probe.install=lambda config:None
        runpy.run_path(str(REPO/'poc_flashvep/native_split_deep_validation/bootstrap/sitecustomize.py'))
        if cfg.get('intervention_method','native')!='native':
            if cfg.get('use_old_scheduler'):
                from poc_flashvep.idle_intervention_dynamic_loop.runtime import install
            else:
                from poc_flashvep.completion_aware_loop_pd.runtime import install
            install(cfg)
            if cfg.get('intervention_warmup_widths'):
                from poc_flashvep.idle_intervention_dynamic_loop.warmup import install as warmup
                warmup(cfg)
        if cfg.get('sparse_device_trace'):
            from poc_flashvep.idle_intervention_dynamic_loop.sparse_trace import install as sparse
            sparse(cfg)
        if cfg.get('deferred_prompt_state_probe'):
            from poc_flashvep.idle_intervention_dynamic_loop.deferred_state import install as deferred
            deferred(cfg)
        if cfg.get('async_prompt_state_probe'):
            from poc_flashvep.idle_intervention_dynamic_loop.async_state import install as asynchronous
            asynchronous(cfg)
        if cfg.get('full_kernel_trace'):
            manifest=json.loads((ROOT/'raw'/cfg['name']/'full_collector_manifest.json').read_text())
            assert digest(manifest['binary'])==manifest['binary_sha256'],'Activity binary changed'
            from poc_flashvep.completion_aware_loop_pd.full_activity import install as activity
            activity(cfg)
        if cfg.get('no_measured_graph_capture'):
            from poc_flashvep.completion_aware_loop_pd.graph_freeze import install as freeze
            freeze(cfg)
    except BaseException:
        import traceback
        traceback.print_exc();os._exit(70)
