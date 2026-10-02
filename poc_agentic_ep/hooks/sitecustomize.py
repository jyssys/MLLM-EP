import os
if os.environ.get('AGENT_EP_CONTROL'):
    from poc_agentic_ep.instrumentation import install
    install()
    os.environ['AGENT_EP_HOOK_IMPORTED']='1'
