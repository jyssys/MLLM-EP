"""Keep the declared Mode0 normalization provider on the compiled D role."""
from .decode_compile_custom_ops import options as custom_options


def options(cfg,role='single'):
    value=custom_options(cfg,role)
    if role=='decode' and cfg.get('D_mode0_IR_norm_priority'):
        assert cfg.get('preserve_compiled_decode_only') and cfg.get('D_compiled_custom_ops_all')
        assert 'kernel_config' not in value
        value['kernel_config']=dict(ir_op_priority={
            key:['vllm_c','native'] for key in ('rms_norm','fused_add_rms_norm')})
    return value


def cpu_test():
    cfg=dict(execution_family='MATCHED_NO_COMPILE',axis_kind='Layer',
        preserve_compiled_decode_only=True,D_execution_family_override='INSTALLED_DEFAULT_COMPILE',
        D_compiled_custom_ops_all=True,D_mode0_IR_norm_priority=True)
    assert options(cfg,'prefill')==custom_options(cfg,'prefill')
    before=custom_options(cfg,'decode');after=options(cfg,'decode')
    assert set(after)-set(before)=={'kernel_config'}
    assert all(v==['vllm_c','native'] for v in after['kernel_config']['ir_op_priority'].values())
    assert dict(after,kernel_config=None)==dict(before,kernel_config=None)
    print('DECODE_IR_PRIORITY_ONLY_ROLE_OVERRIDE_PASS',flush=True)


if __name__=='__main__':cpu_test()
