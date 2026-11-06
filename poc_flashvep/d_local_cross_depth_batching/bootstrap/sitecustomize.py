"""Load the qualified numerical path, then isolated campaign hooks."""
import os
if os.environ.get('FLUIDTOKEN_LOOP_WORKER') == '1':
    try:
        import runpy
        from poc_flashvep.d_local_cross_depth_batching.common import configure, REPO
        configure()
        import json
        from pathlib import Path
        contract = json.loads(Path(os.environ['ARC_CONFIG']).read_text())
        if contract.get('dlocal_service_once'):
            from poc_flashvep.dynamic_loop_split_serving import service_timing
            service_timing.install=lambda cfg: None
        if contract.get('dlocal_activity_thin'):
            from poc_flashvep.fluidtoken_loop_split import activity_capture
            activity_capture.install=lambda cfg: None
        if contract.get('freeze_dlocal_sources'):
            from poc_flashvep.d_local_cross_depth_batching.common import ROOT, digest, write
            hashes = json.loads((ROOT/'raw'/contract['name']/'new_campaign_source_hashes.json').read_text())
            for path, expected in hashes.items():
                assert digest(path)==expected, 'D-local source changed before worker startup: '+path
            write(ROOT/'raw'/contract['name']/f'loaded_dlocal_source_{os.getpid()}.json',dict(hashes=hashes,
                scope='Source SHA256 checked before process-local hooks load. Per-run source archive preserves these files.'))
        if contract.get('dlocal_method', 'native') != 'native':
            from poc_flashvep.native_split_deep_validation import graph_policy
            original_geometry = graph_policy.preferred_geometry
            def local_geometry(cfg, role):
                if role != 'decode':
                    return original_geometry(cfg, role)
                chunk = cfg['length'] if cfg['dlocal_method'] == 'full' else cfg.get('dlocal_chunk', 32)
                return {(chunk*b, chunk) for b in range(1, 9)
                        if chunk*b <= cfg['D_prefill_graph_rows']}
            graph_policy.preferred_geometry = local_geometry
        runpy.run_path(str(REPO / 'poc_flashvep/native_split_deep_validation/bootstrap/sitecustomize.py'))
        from poc_flashvep.looped_pd_atomic_prefill.common import config
        if config().get('dlocal_campaign'):
            from poc_flashvep.d_local_cross_depth_batching.hooks import install
            install(config())
            from poc_flashvep.d_local_cross_depth_batching.trace import install as trace
            trace(config())
            from poc_flashvep.d_local_cross_depth_batching.pacing import install as pacing
            pacing(config())
            from poc_flashvep.d_local_cross_depth_batching.thin_activity import install as thin_activity
            thin_activity(config())
            from poc_flashvep.d_local_cross_depth_batching.service_once import install as service_once
            service_once(config())
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
