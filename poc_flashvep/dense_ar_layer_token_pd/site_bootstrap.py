"""Snapshot-only path binding. No installed runtime files are changed."""
import os
if os.environ.get('DENSE_LAYER_TOKEN_BOOTSTRAP') == '1':
    from poc_flashvep.dense_ar_layer_token_pd.common import bind_previous
    bind_previous()
    from poc_flashvep.dense_ar_slo_layer_pd import native
    original_stage = native.stage
    def stage(cfg, role, address):
        from poc_flashvep.dense_ar_layer_token_pd.token_axis import install, install_state_witness
        install(cfg, role)
        install_state_witness(cfg, role)
        from poc_flashvep.dense_ar_layer_token_pd.credit_fast_lane import install as fast_control
        fast_control(cfg, role)
        return original_stage(cfg, role, address)
    native.stage = stage
    from poc_flashvep.dense_ar_layer_token_pd.recording import annotate
    native.append = annotate(native.append)
    # The coordinator launches stage processes with -m. Route only that exact
    # module to a wrapper that uses the already-bound native module object.
    import subprocess
    OriginalPopen = subprocess.Popen
    class Popen(OriginalPopen):
        def __init__(self, args, *positional, **keywords):
            if isinstance(args, (list, tuple)) and '-m' in args:
                args = list(args)
                index = args.index('-m') + 1
                if args[index] == 'poc_flashvep.dense_ar_slo_layer_pd.native':
                    args[index] = 'poc_flashvep.dense_ar_layer_token_pd.serving'
            super().__init__(args, *positional, **keywords)
    subprocess.Popen = Popen
