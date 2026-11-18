"""Load only campaign-local adapters in the immutable worker snapshot."""
import os
if os.environ.get('DENSE_MATCHED_AXIS_BOOTSTRAP') == '1':
    from poc_flashvep.dense_ar_matched_axis_pd.common import bind_legacy, ROOT, write
    bind_legacy()
    from poc_flashvep.dense_ar_slo_layer_pd import native
    original_stage = native.stage
    def stage(cfg, role, address):
        assert cfg['token_budget']==cfg['D_token_budget']==8192
        assert cfg['P_slots']==cfg['D_slots']==1024
        from poc_flashvep.dense_ar_matched_axis_pd.common_credits import install as common_credits
        common_credits(cfg,role)
        from poc_flashvep.dense_ar_layer_token_pd.token_axis import install, install_state_witness
        install(cfg, role)
        install_state_witness(cfg, role)
        from poc_flashvep.dense_ar_matched_axis_pd.layer_witness import install as layer_witness
        layer_witness(cfg,role)
        from poc_flashvep.dense_ar_matched_axis_pd.segmented_source import install as chunked_source
        chunked_source(cfg,role)
        from poc_flashvep.dense_ar_matched_axis_pd.profile_labels import install as profile_labels
        profile_labels(cfg,role)
        return original_stage(cfg, role, address)
    native.stage = stage
    from poc_flashvep.dense_ar_layer_token_pd.recording import annotate
    native.append = annotate(native.append)
    import subprocess
    original_Popen = subprocess.Popen
    class Popen(original_Popen):
        def __init__(self, args, *positional, **keywords):
            if isinstance(args, (list, tuple)) and '-m' in args:
                args = list(args)
                index = args.index('-m') + 1
                if args[index] == 'poc_flashvep.dense_ar_slo_layer_pd.native':
                    args[index] = 'poc_flashvep.dense_ar_matched_axis_pd.worker'
            super().__init__(args, *positional, **keywords)
    subprocess.Popen = Popen
