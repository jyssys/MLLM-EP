"""Keep Mode0 custom operators while restoring the declared D compiler."""
from .role_compile_options import options as role_options


def options(cfg, role='single'):
    value=role_options(cfg,role)
    if role=='decode' and cfg.get('D_compiled_custom_ops_all'):
        assert cfg.get('preserve_compiled_decode_only')
        assert cfg['D_execution_family_override']=='INSTALLED_DEFAULT_COMPILE'
        assert 'compilation_config' not in value
        value['compilation_config']=dict(custom_ops=['all'])
    return value


def cpu_test():
    cfg=dict(execution_family='MATCHED_NO_COMPILE',axis_kind='Layer',
        preserve_compiled_decode_only=True,D_execution_family_override='INSTALLED_DEFAULT_COMPILE',
        D_compiled_custom_ops_all=True)
    assert options(cfg,'prefill')==role_options(cfg,'prefill')
    assert options(cfg,'decode')['compilation_config']==dict(custom_ops=['all'])
    plain=dict(cfg,D_compiled_custom_ops_all=False)
    assert options(plain,'decode')==role_options(plain,'decode')
    print('DECODE_COMPILED_CUSTOM_OPS_ROLE_LOCAL_PASS',flush=True)


if __name__=='__main__':cpu_test()
