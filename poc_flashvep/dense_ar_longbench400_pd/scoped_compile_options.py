"""Keep normal compiled decode and an explicit independent teacher contract."""
from .decode_compile_custom_ops import options as custom_options


def options(cfg,role='single'):
    value=custom_options(cfg,role)
    if cfg.get('D_partial_IR_norm_priority'):
        assert cfg.get('axis_kind') in ('Layer','Hybrid')
        assert cfg.get('preserve_compiled_decode_only') and cfg.get('D_compiled_custom_ops_all')
        assert not cfg.get('D_mode0_IR_norm_priority')
    if role=='single' and cfg.get('teacher_mode0_IR_norm_priority'):
        assert cfg.get('reference_single') and cfg.get('teacher_eager_prefix_compiled_suffix')
        assert cfg['execution_family']=='INSTALLED_DEFAULT_COMPILE'
        assert 'compilation_config' not in value
        value['compilation_config']=dict(custom_ops=['all'])
    return value


def cpu_test():
    cfg=dict(execution_family='MATCHED_NO_COMPILE',axis_kind='Layer',
        preserve_compiled_decode_only=True,D_execution_family_override='INSTALLED_DEFAULT_COMPILE',
        D_compiled_custom_ops_all=True,D_partial_IR_norm_priority=True)
    for role in ('prefill','decode'):
        assert options(cfg,role)==custom_options(cfg,role)
    teacher=dict(execution_family='INSTALLED_DEFAULT_COMPILE',reference_single=True,
        teacher_eager_prefix_compiled_suffix=True,teacher_mode0_IR_norm_priority=True)
    assert options(teacher,'single')['compilation_config']==dict(custom_ops=['all'])
    assert 'kernel_config' not in options(cfg,'decode')
    print('SCOPED_PREFILL_POLICY_LEAVES_FULL_DECODE_DEFAULT_PASS',flush=True)


if __name__=='__main__':cpu_test()
