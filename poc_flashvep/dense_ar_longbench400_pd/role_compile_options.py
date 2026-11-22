"""Change the declared decode compiler without changing prefill options."""
from ..dense_ar_bi0_shortanswer_pd.engine_options import options as original_options


def options(cfg, role='single'):
    value = original_options(cfg, role)
    override = cfg.get('D_execution_family_override')
    if role != 'decode' or override is None:
        return value
    assert override == 'INSTALLED_DEFAULT_COMPILE', override
    assert cfg['execution_family'] == 'MATCHED_NO_COMPILE'
    assert cfg.get('preserve_compiled_decode_only')
    assert cfg.get('axis_kind') in ('Layer', 'Hybrid')
    value.pop('compilation_config', None)
    return value


def cpu_test():
    base = dict(execution_family='MATCHED_NO_COMPILE', axis_kind='Layer',
                preserve_compiled_decode_only=True,
                D_execution_family_override='INSTALLED_DEFAULT_COMPILE')
    for role in ('prefill', 'single', 'replica0', 'replica1'):
        assert options(base, role) == original_options(base, role)
    before = original_options(base, 'decode')
    after = options(base, 'decode')
    assert before['compilation_config']['mode'] == 0
    assert 'compilation_config' not in after
    assert dict(before, compilation_config=None) == dict(after, compilation_config=None)
    for role in ('prefill', 'decode', 'single'):
        plain = dict(execution_family='MATCHED_NO_COMPILE')
        assert options(plain, role) == original_options(plain, role)
    print('ROLE_COMPILE_OPTIONS_ONLY_DECODE_OVERRIDE_PASS', flush=True)


if __name__ == '__main__':
    cpu_test()
