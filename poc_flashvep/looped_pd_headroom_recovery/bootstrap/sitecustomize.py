"""Install recovery hooks in each fresh native worker interpreter."""
import os

if os.environ.get('HEADROOM_RECOVERY_WORKER') == '1':
    try:
        from poc_flashvep.looped_pd_headroom_recovery.common import configure
        configure()
        from poc_flashvep.looped_pd_headroom_recovery.runtime import install
        install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
