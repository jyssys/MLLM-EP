"""Install exact Native adapters before optional campaign extensions."""
import os
if os.environ.get('FLUIDTOKEN_LOOP_WORKER') == '1':
    try:
        import importlib.util
        import json
        import runpy
        import sys
        from pathlib import Path
        snapshot = Path(os.environ['NET_BENEFIT_SNAPSHOT'])
        import poc_flashvep
        for name in ('huginn_net_benefit', 'completion_aware_loop_pd', 'idle_intervention_dynamic_loop'):
            key = 'poc_flashvep.' + name
            package = snapshot / 'poc_flashvep' / name
            spec = importlib.util.spec_from_file_location(key, package / '__init__.py', submodule_search_locations=[str(package)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
        from poc_flashvep.huginn_net_benefit.common import configure, REPO, ROOT, digest, write
        configure()
        cfg = json.loads(Path(os.environ['ARC_CONFIG']).read_text())
        folder = ROOT / 'raw' / cfg['name']
        for manifest in ('new_campaign_source_hashes.json', 'qualified_predecessor_hashes.json'):
            hashes = json.loads((folder / manifest).read_text())
            assert all(digest(snapshot / Path(path).relative_to(REPO)) == value for path, value in hashes.items()), manifest
        write(folder / f'loaded_net_benefit_source_{os.getpid()}.json', dict(snapshot=str(snapshot), config_sha256=digest(os.environ['ARC_CONFIG'])))
        write(folder / f'runtime_environment_{os.getpid()}.json', {
            key: os.environ.get(key, 'UNSET_DRIVER_DEFAULT') for key in (
                'CUDA_DEVICE_MAX_CONNECTIONS', 'CUDA_DEVICE_MAX_COPY_CONNECTIONS',
                'CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING')})
        assert not cfg.get('dlocal_campaign') and not cfg.get('D_local_probe_output_pacing')
        if cfg.get('async_prompt_state_probe'):
            assert cfg['state_probe'] and not cfg['verify_transfer']
            from poc_flashvep.looped_pd_headroom_recovery import state_probe
            state_probe.install = lambda config: None
        if cfg.get('intervention_method', 'native') != 'native' or cfg.get('D_prefill_token_cap'):
            from poc_flashvep.native_split_deep_validation import graph_policy
            geometry = graph_policy.preferred_geometry
            def preferred(config, role):
                if role == 'decode':
                    if config.get('intervention_method', 'native') == 'native':
                        frontier = config.get('native_frontier')
                        remaining = config['length']-frontier[1] if frontier else config['length']
                        width = min(config['D_prefill_token_cap'], remaining)
                        widths = {width, remaining % width} - {0}
                    else:
                        widths = config.get('D_candidate_widths', [config.get('D_tile_tokens', config['length'])])
                    return {(q*b, q) for q in widths for b in range(1, config['D_slots']+1)
                        if q*b <= config['D_prefill_graph_rows']}
                if config.get('intervention_method') != 'dynamic_token':
                    return geometry(config, role)
                points = [0] + config['boundary_checkpoints'] + [config['length']]
                widths = {b-a for a in points for b in points if b > a}
                allowed = {(q, q) for q in widths if q <= config['P_prefill_graph_rows']}
                if config['P_slots'] >= 2:
                    allowed.update((a+b, max(a, b)) for a in widths for b in widths
                        if a+b <= config['P_prefill_graph_rows'])
                return allowed
            graph_policy.preferred_geometry = preferred
        runpy.run_path(str(REPO / 'poc_flashvep/native_split_deep_validation/bootstrap/sitecustomize.py'))
        if cfg.get('exact_integer_validation'):
            from poc_flashvep.huginn_net_benefit.integer_validation import install as integer_validation
            integer_validation(cfg)
        if cfg.get('skip_dead_partial_coda_prepare'):
            from poc_flashvep.huginn_net_benefit.partial_prepare import install as partial_prepare
            partial_prepare(cfg)
        if cfg.get('enable_prefill_uva'):
            from dataclasses import replace
            from poc_flashvep.looped_pd_abc import worker
            base_configuration = worker.configuration
            def configuration(*args, **kwargs):
                value = base_configuration(*args, **kwargs)
                value['execution_config'] = replace(value['execution_config'], prefill_uva=True)
                return value
            worker.configuration = configuration
        if cfg.get('decode_loop_grouping'):
            from poc_flashvep.huginn_net_benefit.fixed_decode import install as fixed_decode
            fixed_decode(cfg)
        if cfg.get('fixed_depth_submit'):
            from poc_flashvep.huginn_net_benefit.fixed_submit import install as fixed_submit
            fixed_submit(cfg)
        if cfg.get('intervention_method', 'native') != 'native':
            if cfg.get('monotone_early_KV_protocol'):
                from poc_flashvep.huginn_net_benefit.monotone_early_protocol import install
            else:
                from poc_flashvep.completion_aware_loop_pd.runtime import install
            install(cfg)
            if cfg.get('net_benefit_extension'):
                from poc_flashvep.huginn_net_benefit.runtime import install as extension
                extension(cfg)
            if cfg.get('track_source_device_events'):
                from poc_flashvep.huginn_net_benefit.source_events import install as source_events
                source_events(cfg)
            if cfg.get('decode_loop_grouping') or cfg.get('fixed_depth_submit'):
                from poc_flashvep.huginn_net_benefit.pending_decode import install as pending_decode
                pending_decode(cfg)
            if cfg.get('intervention_warmup_widths'):
                from poc_flashvep.idle_intervention_dynamic_loop.warmup import install as warmup
                warmup(cfg)
        if cfg.get('native_loop_event_transfer'):
            from poc_flashvep.huginn_net_benefit.early_tap import install as early_tap
            early_tap(cfg)
        if cfg.get('whole_owned_block_wire'):
            from poc_flashvep.huginn_net_benefit.block_wire import install as block_wire
            block_wire(cfg)
        if cfg.get('wire_progress_burst'):
            from poc_flashvep.huginn_net_benefit.wire_burst import install as wire_burst
            wire_burst(cfg)
        if cfg.get('independent_prompt_stream'):
            from poc_flashvep.huginn_net_benefit.stream_route import install as stream_route
            stream_route(cfg)
        if cfg.get('deferred_coda_collection'):
            from poc_flashvep.huginn_net_benefit.deferred_coda import install as deferred_coda
            deferred_coda(cfg)
        if cfg.get('explicit_decode_priority'):
            assert cfg.get('intervention_method', 'native') == 'native', (
                'The serial dynamic budget must not block a committed spatial continuation')
            from poc_flashvep.huginn_net_benefit.decode_priority import install as decode_priority
            decode_priority(cfg)
        if cfg.get('D_prefill_loop_cap') or cfg.get('D_prefill_token_cap'):
            assert cfg.get('intervention_method', 'native') == 'native', (
                'Dynamic policy costs must also respect the execution group cap')
            from poc_flashvep.huginn_net_benefit.prefill_group import install as prefill_group
            prefill_group(cfg)
        if cfg.get('sparse_device_trace'):
            from poc_flashvep.idle_intervention_dynamic_loop.sparse_trace import install as sparse
            sparse(cfg)
        if cfg.get('host_cost_trace'):
            from poc_flashvep.huginn_net_benefit.host_cost import install as host_cost
            host_cost(cfg)
        if cfg.get('nsight_graph_enclosures'):
            from poc_flashvep.huginn_net_benefit.graph_trace import install as graph_trace
            graph_trace(cfg)
        if cfg.get('async_prompt_state_probe'):
            from poc_flashvep.idle_intervention_dynamic_loop.async_state import install as state
            state(cfg)
        if cfg.get('post_decode_state'):
            from poc_flashvep.huginn_net_benefit.post_decode_state import install as final_state
            final_state(cfg)
        if cfg.get('source_submission_pipeline'):
            from poc_flashvep.huginn_net_benefit.source_pipeline import install as source_pipeline
            source_pipeline(cfg)
        if cfg.get('predictive_checkpoint_preparation'):
            assert cfg.get('intervention_method') in ('dynamic_loop', 'dynamic_tile', 'dynamic_token')
            from poc_flashvep.huginn_net_benefit.predictive_prepare import install as predictive_prepare
            predictive_prepare(cfg)
        if cfg.get('drained_completion_recovery'):
            from poc_flashvep.huginn_net_benefit.drained_recovery import install as drained_recovery
            drained_recovery(cfg)
        if cfg.get('no_measured_graph_capture'):
            from poc_flashvep.completion_aware_loop_pd.graph_freeze import install as freeze
            freeze(cfg)
        from poc_flashvep.huginn_net_benefit.resolved_config import install as resolved_config
        resolved_config(cfg)
        if cfg.get('explicit_graph_teardown'):
            from poc_flashvep.huginn_net_benefit.graph_lifetime import install as graph_lifetime
            graph_lifetime(cfg)
        from poc_flashvep.huginn_net_benefit.exit_integrity import install as exit_integrity
        exit_integrity(cfg)
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
