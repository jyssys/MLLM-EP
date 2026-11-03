"""Install only process-local campaign hooks in fresh qualified interpreters."""
import os

if os.environ.get('AUTONOMOUS_HEADROOM_WORKER')=='1':
    try:
        from poc_flashvep.looped_pd_autonomous_headroom.common import activate,digest
        import json
        activate()
        for path,expected in json.loads(os.environ.get('AUTONOMOUS_SOURCE_HASHES','{}')).items():
            assert digest(path)==expected,'Campaign source changed before interpreter startup: '+path
        if os.environ.get('AUTONOMOUS_OPERATOR')!='1':
            from poc_flashvep.looped_pd_headroom_recovery.runtime import install
            install()
            from poc_flashvep.looped_pd_autonomous_headroom.runtime import install as campaign
            campaign()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
