import os
if os.environ.get('ATOMIC_NATIVE_COLOCATED')=='1':
    try:
        from poc_flashvep.looped_pd_atomic_prefill.native_mixed_worker import install
        install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
