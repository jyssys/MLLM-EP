import os

if os.environ.get('ASYMMETRY_HOOK') == '1':
    from poc_dispatch_combine_asymmetry.instrumentation import install
    install()
    os.environ['ASYMMETRY_HOOK_IMPORTED'] = '1'
