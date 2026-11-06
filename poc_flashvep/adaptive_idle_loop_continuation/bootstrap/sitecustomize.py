"""Load the qualified native path before bounded idle-budget instrumentation."""
import os
if os.environ.get('FLUIDTOKEN_LOOP_WORKER')=='1':
    try:
        import json
        import runpy
        from pathlib import Path
        from poc_flashvep.adaptive_idle_loop_continuation.common import configure,REPO,ROOT,digest,write
        configure()
        cfg=json.loads(Path(os.environ['ARC_CONFIG']).read_text())
        assert not cfg.get('dlocal_campaign') and not cfg.get('D_local_probe_output_pacing')
        hashes=json.loads((ROOT/'raw'/cfg['name']/'new_campaign_source_hashes.json').read_text())
        assert all(digest(p)==value for p,value in hashes.items()),'Source changed before worker startup'
        write(ROOT/'raw'/cfg['name']/f'loaded_idle_source_{os.getpid()}.json',dict(hashes=hashes))
        runpy.run_path(str(REPO/'poc_flashvep/native_split_deep_validation/bootstrap/sitecustomize.py'))
        if cfg.get('idle_D_loop_quantum') is not None:
            from poc_flashvep.adaptive_idle_loop_continuation.quantum import install as quantum
            quantum(cfg)
        if cfg.get('idle_light_owner'):
            from poc_flashvep.adaptive_idle_loop_continuation.light_owner import install as light_owner
            light_owner(cfg)
        if cfg.get('idle_observer','off')!='off':
            from poc_flashvep.adaptive_idle_loop_continuation.observer import install
            install(cfg)
    except BaseException:
        import traceback
        traceback.print_exc();os._exit(70)
