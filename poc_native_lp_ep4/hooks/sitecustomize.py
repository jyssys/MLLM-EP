import os

if os.environ.get("NATIVE_LP_ENABLE") == "1":
    try:
        from poc_native_lp_ep4.native import install
        install()
    except BaseException:
        # Python normally prints sitecustomize errors and keeps running. That
        # would silently execute stock prefill under an LP arm label.
        import traceback
        traceback.print_exc()
        os._exit(73)
