import os
if os.environ.get('SD_EP_CONTROL'):
    from poc_speculative_moe_ep.instrumentation import install
    install()
    os.environ['SD_EP_HOOK_IMPORTED']='1'
