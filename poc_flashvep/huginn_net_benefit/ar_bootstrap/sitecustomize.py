"""Route the stock AR diagnostic into this campaign without RLT hooks."""
import os
if os.environ.get('NET_BENEFIT_AR') == '1':
    try:
        from poc_flashvep.huginn_net_benefit.ar_diagnostic import configure_ar
        configure_ar()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(70)
