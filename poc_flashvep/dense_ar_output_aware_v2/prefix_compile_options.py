"""Change only the P compiler family when the experimental switch is on."""
from poc_flashvep.dense_ar_longbench400_pd.scoped_compile_options import options as original

def options(cfg,role='single'):
    # The independent teacher uses the same eager-prompt/native-decode split.
    # Its original full36 model remains unchanged; do not enable CUDA ops for
    # the compiled decode model merely to obtain CUDA prompt primitives.
    if role=='single' and cfg.get('teacher_scoped_prompt_custom_ops'):
        cfg=dict(cfg,teacher_mode0_IR_norm_priority=False)
    if role=='prefill' and cfg.get('P_compile_prefix'):
        assert cfg.get('preserve_compiled_full_model')
        cfg=dict(cfg,execution_family='INSTALLED_DEFAULT_COMPILE')
    value=original(cfg,role)
    if role=='prefill' and cfg.get('P_compile_prefix') and cfg.get('P_compiled_custom_ops_all'):
        value['compilation_config']=dict(value.get('compilation_config',{}),custom_ops=['all'])
    return value
