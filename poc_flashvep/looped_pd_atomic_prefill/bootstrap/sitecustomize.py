"""Install process-local study hooks before a spawned RLT worker imports its engine."""
import os

if os.environ.get('ATOMIC_PD_WORKER')=='1':
    try:
        from poc_flashvep.looped_pd_atomic_prefill.common import config
        if config()['model'] in ('ouro','huginn'):
            from poc_flashvep.looped_pd_atomic_prefill.runtime import install
            install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
